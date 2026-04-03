"""
Dependency Score Pipeline
=========================
For each Jira issue in the input JSON:
  1. Identifies seed classes (pre-existing files modified by the feature commits)
  2. Creates an isolated git worktree at sha_before (main repo is never touched)
  3. Runs `depends` against the worktree to extract the static dependency graph
  4. Builds a directed graph with NetworkX
  5. Scores every candidate class via BFS with exponential decay
  6. Tears down the worktree, writes per-issue results to an output JSON

Using a worktree instead of `git checkout` means depends can leave behind
untracked/modified files without ever dirtying the main repo working tree,
so there is no need to stash, clean, or restore HEAD at all.

Usage:
    python dependency_score_pipeline.py \
        --issues     path/to/jira_issues.json \
        --repo       path/to/cloned/git/repo \
        --depends    path/to/depends.jar \
        --output     path/to/output_dir \
        [--decay     0.5] \
        [--max-depth 4] \
        [--cache]            # reuse depends output when sha_before repeats

Requirements:
    pip install networkx
    Java 8+ on PATH  (for depends)
    git 2.5+ on PATH (for worktree support)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(repo: Path, *args, check=True):
    """Run a git command inside *repo* and return stdout as a string."""
    cmd = ["git", "-C", str(repo)] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def make_worktree(repo: Path, sha: str, worktree_dir: Path) -> Path:
    """
    Create a git worktree at *worktree_dir* checked out at *sha*.

    Worktrees share the same object store as the main repo but have a
    completely independent working tree. This means depends can write or
    leave behind any files it wants inside the worktree without ever
    touching the main repo's working tree -- no stash, no clean, no
    HEAD restore needed.
    """
    log.info("  Creating worktree %s -> %s", worktree_dir.name, sha[:10])
    git(repo, "worktree", "add", "--detach", str(worktree_dir), sha)
    return worktree_dir


def remove_worktree(repo: Path, worktree_dir: Path):
    """Remove a worktree directory and deregister it from git."""
    try:
        git(repo, "worktree", "remove", "--force", str(worktree_dir))
    except RuntimeError as e:
        log.warning("  worktree remove failed (%s) -- deleting directory manually.", e)
        shutil.rmtree(worktree_dir, ignore_errors=True)
    # Prune any stale worktree metadata left behind
    git(repo, "worktree", "prune", check=False)


# ---------------------------------------------------------------------------
# depends runner
# ---------------------------------------------------------------------------

def run_depends(depends_jar: Path, source_root: Path, out_dir: Path) -> Path:
    """
    Invoke depends on all Java sources under *source_root* and return the
    path to the produced JSON file.

    depends CLI (v0.9.7+):
        java -jar depends.jar <lang> <src-dir> <output-name> [options]

    The tool writes <output-name>-file.json (file-level graph) in *out_dir*.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = "depends_out"

    cmd = [
        "java", "-jar", str(depends_jar),
        "java",
        str(source_root),
        output_name,
        f"--dir={str(out_dir)}",
        "--granularity=file",
        "--format=json",
        "--auto-include",
    ]

    log.info("  Running depends on %s ...", source_root.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("depends exited non-zero:\n%s", result.stderr[:400])

    candidate = out_dir / f"{output_name}-file.json"
    if not candidate.exists():
        alt = out_dir / f"{output_name}.json"
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Expected depends output at {candidate} but it was not created.\n"
            f"depends stderr:\n{result.stderr[:600]}"
        )
    return candidate


# ---------------------------------------------------------------------------
# depends JSON -> NetworkX DiGraph
# ---------------------------------------------------------------------------

DEPENDENCY_WEIGHTS = {
    "Extend":      1.0,   # inheritance
    "Implement":   1.0,   # interface implementation
    "Call":        0.8,   # method invocation
    "Import":      0.6,   # import statement
    "Use":         0.6,   # type usage / field type
    "Parameter":   0.5,
    "Return":      0.5,
    "Contain":     0.4,
    "Create":      0.4,
    "Cast":        0.3,
    "Annotation":  0.2,
    "Throw":       0.2,
    "DynamicCall": 0.3,
}


def load_depends_graph(depends_json: Path, source_root: Path) -> nx.DiGraph:
    """
    Parse the depends JSON output into a NetworkX DiGraph.

    Node IDs are relative file paths from *source_root*, e.g.:
        flink-clients/src/main/java/org/apache/flink/client/RestClient.java

    *source_root* must be the directory that was passed to depends (i.e. the
    worktree path), so absolute paths in the output are stripped correctly.

    Edge weight = max coupling weight among all dependency types on that edge.
    """
    with open(depends_json, encoding="utf-8") as f:
        data = json.load(f)

    variables: list = data.get("variables", [])
    cells: list = data.get("cells", [])

    root_str = str(source_root.resolve()).replace("\\", "/").rstrip("/") + "/"

    def normalise(raw: str) -> str:
        raw = raw.replace("\\", "/")
        if raw.startswith(root_str):
            raw = raw[len(root_str):]
        return raw.lstrip("/")

    variables = [normalise(v) for v in variables]

    G = nx.DiGraph()
    for v in variables:
        G.add_node(v)

    for cell in cells:
        src_idx = cell.get("src")
        dst_idx = cell.get("dest")
        if src_idx is None or dst_idx is None:
            continue
        if src_idx >= len(variables) or dst_idx >= len(variables):
            continue

        src_node = variables[src_idx]
        dst_node = variables[dst_idx]

        dep_values: dict = cell.get("values", {})
        if not dep_values:
            continue

        weight = max(
            DEPENDENCY_WEIGHTS.get(dep_type, 0.1)
            for dep_type in dep_values
        )

        if G.has_edge(src_node, dst_node):
            G[src_node][dst_node]["weight"] = max(
                G[src_node][dst_node]["weight"], weight
            )
        else:
            G.add_edge(src_node, dst_node, weight=weight)

    log.info(
        "  Dependency graph: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Seed class extraction
# ---------------------------------------------------------------------------

def extract_seed_paths(issue: dict) -> set:
    """
    Return the set of pre-existing Java files that were modified (STATUS M)
    across all commits for this issue. Added/deleted files are excluded.
    """
    seeds = set()
    for commit in issue.get("commits", []):
        for f in commit.get("files", []):
            if f.get("STATUS") != "M":
                continue
            path = f.get("PARENT_PATH", "") or f.get("CHILD_PATH", "")
            if path.endswith(".java"):
                seeds.add(path.replace("\\", "/"))
    return seeds


# ---------------------------------------------------------------------------
# BFS dependency scoring
# ---------------------------------------------------------------------------

def compute_dependency_scores(
    G: nx.DiGraph,
    seed_paths: set,
    decay: float = 0.5,
    max_depth: int = 4,
    reverse_weight_factor: float = 0.7,
) -> dict:
    """
    Score every node in *G* relative to *seed_paths* using weighted,
    exponential-decay BFS in both directions.

    Forward BFS (seed -> candidate):
        Classes the seed uses. High score = the feature depends on this class.
        TD here can propagate into the new feature (primary contagion risk).

    Reverse BFS (candidate -> seed, via reversed graph):
        Classes that use the seed. High score = this class may break if the
        seed changes. Dampened by reverse_weight_factor.

    Score for a path: product(edge_weights) * decay^depth
    Combined score is normalised to [0, 1] within the issue.
    Seeds are fixed at 1.0.
    """
    forward_scores: dict = defaultdict(float)
    reverse_scores: dict = defaultdict(float)
    min_depth_fwd: dict = {}
    min_depth_rev: dict = {}

    G_rev = G.reverse(copy=False)

    for seed in seed_paths:
        if seed not in G:
            log.debug("    Seed not in graph: %s", seed)
            continue

        # Forward BFS
        visited_fwd = {seed: (0, 1.0)}
        queue = [(seed, 0, 1.0)]
        while queue:
            current, depth, cum_weight = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in G.successors(current):
                if neighbor == seed:
                    continue
                edge_w = G[current][neighbor].get("weight", 0.5)
                new_weight = cum_weight * edge_w * (decay ** (depth + 1))
                new_depth = depth + 1
                if neighbor not in visited_fwd or visited_fwd[neighbor][1] < new_weight:
                    visited_fwd[neighbor] = (new_depth, new_weight)
                    queue.append((neighbor, new_depth, new_weight))

        for node, (depth, weight) in visited_fwd.items():
            if node == seed:
                continue
            forward_scores[node] += weight
            if node not in min_depth_fwd or min_depth_fwd[node] > depth:
                min_depth_fwd[node] = depth

        # Reverse BFS
        visited_rev = {seed: (0, 1.0)}
        queue = [(seed, 0, 1.0)]
        while queue:
            current, depth, cum_weight = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in G_rev.successors(current):
                if neighbor == seed:
                    continue
                edge_w = G_rev[current][neighbor].get("weight", 0.5)
                new_weight = cum_weight * edge_w * reverse_weight_factor * (decay ** (depth + 1))
                new_depth = depth + 1
                if neighbor not in visited_rev or visited_rev[neighbor][1] < new_weight:
                    visited_rev[neighbor] = (new_depth, new_weight)
                    queue.append((neighbor, new_depth, new_weight))

        for node, (depth, weight) in visited_rev.items():
            if node == seed:
                continue
            reverse_scores[node] += weight
            if node not in min_depth_rev or min_depth_rev[node] > depth:
                min_depth_rev[node] = depth

    # Combine and normalise
    all_nodes = set(forward_scores) | set(reverse_scores)
    raw_combined = {
        node: forward_scores.get(node, 0.0) + reverse_scores.get(node, 0.0)
        for node in all_nodes
    }
    max_val = max(raw_combined.values()) if raw_combined else 1.0

    results = {}

    for seed in seed_paths:
        if seed in G:
            results[seed] = {
                "forward_score": 1.0,
                "reverse_score": 1.0,
                "combined_score": 1.0,
                "min_depth_forward": 0,
                "min_depth_reverse": 0,
                "is_seed": True,
            }

    for node, raw in raw_combined.items():
        results[node] = {
            "forward_score": round(forward_scores.get(node, 0.0), 6),
            "reverse_score": round(reverse_scores.get(node, 0.0), 6),
            "combined_score": round(raw / max_val, 6),
            "min_depth_forward": min_depth_fwd.get(node),
            "min_depth_reverse": min_depth_rev.get(node),
            "is_seed": node in seed_paths,
        }

    return results


# ---------------------------------------------------------------------------
# Per-issue pipeline
# ---------------------------------------------------------------------------

def process_issue(
    issue_id: str,
    issue: dict,
    repo: Path,
    depends_jar: Path,
    output_dir: Path,
    cache_dir: Path,
    decay: float,
    max_depth: int,
    use_cache: bool,
) -> dict:
    """Full pipeline for a single Jira issue. Returns the result dict."""

    sha_before = issue.get("sha_before")
    if not sha_before:
        log.warning("[%s] No sha_before -- skipping.", issue_id)
        return {"error": "missing sha_before"}

    seed_paths = extract_seed_paths(issue)
    if not seed_paths:
        log.warning("[%s] No modified pre-existing .java files -- skipping.", issue_id)
        return {"error": "no seed classes found"}

    log.info("[%s] sha_before=%s | seeds=%d", issue_id, sha_before[:10], len(seed_paths))

    # Cache check: if two issues share sha_before we reuse the depends output
    # rather than spinning up a new worktree and re-running depends.
    depends_cache_path = cache_dir / sha_before / "depends_out-file.json"

    if use_cache and depends_cache_path.exists():
        log.info("  [cache hit] sha=%s", sha_before[:10])
        depends_json = depends_cache_path
        # Paths in the cached JSON were already normalised relative to the
        # worktree root at cache time, so we use repo as the strip prefix
        # (which resolves identically for already-relative paths).
        source_root = repo
    else:
        # Create an isolated worktree so the main repo working tree is never
        # touched regardless of what depends writes or leaves behind.
        worktree_dir = output_dir / "_worktrees" / sha_before
        worktree_dir.parent.mkdir(parents=True, exist_ok=True)

        # Remove any stale worktree from a previous crashed run
        if worktree_dir.exists():
            log.info("  Removing stale worktree at %s", worktree_dir)
            remove_worktree(repo, worktree_dir)

        make_worktree(repo, sha_before, worktree_dir)
        try:
            with tempfile.TemporaryDirectory(prefix="depends_out_") as tmp:
                depends_json_tmp = run_depends(depends_jar, worktree_dir, Path(tmp))

                dest_dir = (
                    cache_dir / sha_before
                    if use_cache
                    else output_dir / "depends_cache" / sha_before
                )
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / depends_json_tmp.name
                shutil.copy2(depends_json_tmp, dest)
                depends_json = dest

        finally:
            # Always remove the worktree even if depends failed
            log.info("  Removing worktree for %s", sha_before[:10])
            remove_worktree(repo, worktree_dir)

        # worktree_dir is gone now but its resolved path string is still
        # correct for stripping absolute paths from the depends output.
        source_root = worktree_dir

    G = load_depends_graph(depends_json, source_root)

    scores = compute_dependency_scores(
        G, seed_paths, decay=decay, max_depth=max_depth
    )

    all_java_nodes = [n for n in G.nodes if n.endswith(".java")]
    seed_nodes_found = sum(1 for s in seed_paths if s in G)

    return {
        "jira_id": issue_id,
        "sha_before": sha_before,
        "seed_classes": sorted(seed_paths),
        "seed_classes_in_graph": seed_nodes_found,
        "total_java_nodes_in_graph": len(all_java_nodes),
        "scored_candidate_classes": len(scores),
        "params": {"decay": decay, "max_depth": max_depth},
        "dependency_scores": scores,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute dependency scores for Jira issues")
    parser.add_argument("--issues",        required=True, help="Path to jira_issues.json")
    parser.add_argument("--repo",          required=True, help="Path to cloned git repository")
    parser.add_argument("--depends",       required=True, help="Path to depends.jar")
    parser.add_argument("--output",        required=True, help="Output directory")
    parser.add_argument("--decay",         type=float, default=0.5, help="BFS decay factor (default 0.5)")
    parser.add_argument("--max-depth",     type=int,   default=4,   help="Max BFS depth (default 4)")
    parser.add_argument("--cache",         action="store_true",      help="Cache depends output by sha_before")
    parser.add_argument("--issues-filter", nargs="*", help="Only process these Jira IDs")
    args = parser.parse_args()

    repo        = Path(args.repo).resolve()
    depends_jar = Path(args.depends).resolve()
    output_dir  = Path(args.output).resolve()
    issues_path = Path(args.issues).resolve()
    cache_dir   = output_dir / "_depends_cache"

    if not repo.exists():
        sys.exit(f"Repo not found: {repo}")
    if not depends_jar.exists():
        sys.exit(f"depends.jar not found: {depends_jar}")
    if not issues_path.exists():
        sys.exit(f"Issues file not found: {issues_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(issues_path, encoding="utf-8") as f:
        all_issues: dict = json.load(f)

    if args.issues_filter:
        all_issues = {k: v for k, v in all_issues.items() if k in args.issues_filter}
        log.info("Filtered to %d issues: %s", len(all_issues), list(all_issues.keys()))

    # ------------------------------------------------------------------
    # Resume support: any existing per-issue file that is complete (no
    # "error" key, non-empty "dependency_scores") is loaded directly and
    # skipped in the processing loop.
    # ------------------------------------------------------------------
    all_results: dict = {}
    skipped: list = []

    for issue_id in list(all_issues.keys()):
        per_issue_path = output_dir / f"{issue_id}_dep_scores.json"
        if per_issue_path.exists():
            try:
                with open(per_issue_path, encoding="utf-8") as f:
                    existing = json.load(f)
                if "error" not in existing and existing.get("dependency_scores"):
                    all_results[issue_id] = existing
                    skipped.append(issue_id)
            except Exception:
                pass  # corrupt / partial file -- will be reprocessed

    remaining = {k: v for k, v in all_issues.items() if k not in all_results}

    log.info(
        "Issues total: %d  |  Already done (skipping): %d  |  To process: %d",
        len(all_issues), len(skipped), len(remaining),
    )

    if not remaining:
        log.info("Nothing left to process -- writing consolidated summary and exiting.")
        summary_path = output_dir / "all_dependency_scores.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log.info("Summary: %s", summary_path)
        return

    # No try/finally needed around the main loop: because we use worktrees,
    # the main repo working tree is never modified and requires no cleanup.
    failed: list = []

    for issue_id, issue in remaining.items():
        try:
            result = process_issue(
                issue_id=issue_id,
                issue=issue,
                repo=repo,
                depends_jar=depends_jar,
                output_dir=output_dir,
                cache_dir=cache_dir,
                decay=args.decay,
                max_depth=args.max_depth,
                use_cache=args.cache,
            )
            all_results[issue_id] = result

            per_issue_path = output_dir / f"{issue_id}_dep_scores.json"
            with open(per_issue_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            log.error("[%s] FAILED: %s", issue_id, e, exc_info=True)
            failed.append(issue_id)
            all_results[issue_id] = {"error": str(e)}

    summary_path = output_dir / "all_dependency_scores.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log.info("Done. Results written to: %s", output_dir)
    log.info("  Skipped (already done) : %d", len(skipped))
    log.info("  Newly processed        : %d", len(remaining) - len(failed))
    log.info("  Failed                 : %d  %s", len(failed), failed if failed else "")
    log.info("  Summary                : %s", summary_path)


if __name__ == "__main__":
    main()