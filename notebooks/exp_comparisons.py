# utils/exp_yaml_printer.py
import subprocess

import yaml
from dvc.api import open as dvc_open  # works across DVC 2/3

# Path to your YAML artifact relative to repo root
ARTIFACT = "output/top_words.yaml"


def _list_experiment_refs() -> list[dict[str, str]]:
    """
    Return a list of dicts with keys: ref, label, sha.
    Uses pure Git to enumerate DVC experiment refs.
    """
    out = subprocess.check_output(  # noqa: S603
        ["git", "for-each-ref", "refs/exps", "--format=%(refname:short) %(objectname)"],  # noqa: S607
        text=True,
    )
    exps = []
    for line in out.strip().splitlines():
        ref, sha = line.split()
        label = ref.split("/")[-1]  # short tail of the ref; not always pretty, but stable
        exps.append({"ref": ref, "label": label, "sha": sha})
    return exps


def _match_experiment(exps: list[dict[str, str]], query: str) -> dict[str, str] | None:
    """
    Match by:
      - exact SHA
      - SHA prefix
      - exact label
      - case-insensitive label contains (if unique)
    Returns the single matched experiment dict or None.
    If multiple fuzzy matches, prints choices and returns None.
    """
    query_l = query.lower()

    # exact sha or exact label
    exact = [e for e in exps if e["sha"] == query or e["label"] == query]
    if exact:
        return exact[0]

    # sha prefix
    pref = [e for e in exps if e["sha"].startswith(query)]
    if len(pref) == 1:
        return pref[0]
    if len(pref) > 1:
        print(f"Multiple experiments match SHA prefix '{query}':")
        for e in pref:
            print(f"  {e['label']}  {e['sha'][:7]}")
        return None

    # label contains (case-insensitive)
    fuzzy = [e for e in exps if query_l in e["label"].lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        print(f"Multiple experiments match label '{query}':")
        for e in fuzzy:
            print(f"  {e['label']}  {e['sha'][:7]}")
        return None

    print(f"No experiment matched '{query}'.")
    return None


def _load_yaml_at_rev(path: str, rev: str):
    with dvc_open(path, rev=rev, mode="r") as f:
        return yaml.safe_load(f)


def print_all_experiments_text(artifact: str = ARTIFACT):
    """
    Print the YAML artifact for all experiments as plain text.
    """
    exps = _list_experiment_refs()
    if not exps:
        print("No DVC experiments found (refs/exps/*).")
        return

    for i, e in enumerate(exps, 1):
        print("=" * 80)
        print(f"[{i}/{len(exps)}] EXPERIMENT: {e['label']}  ({e['sha'][:12]})")
        print("-" * 80)
        try:
            data = _load_yaml_at_rev(artifact, rev=e["sha"])
            print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        except Exception as err:
            print(f"[ERROR] Could not load {artifact} for {e['label']} ({e['sha'][:7]}): {err}")
        print()  # blank line after each


def print_experiment_yaml(exp_query: str, artifact: str = ARTIFACT):
    """
    Print the YAML artifact for a specific experiment identified by:
      - experiment name/label (from the ref tail),
      - SHA,
      - or SHA prefix.
    """
    exps = _list_experiment_refs()
    if not exps:
        print("No DVC experiments found (refs/exps/*).")
        return

    match = _match_experiment(exps, exp_query)
    if not match:
        # already printed a helpful message
        return

    print("=" * 80)
    print(f"EXPERIMENT: {match['label']}  ({match['sha'][:12]})")
    print("-" * 80)
    try:
        data = _load_yaml_at_rev(artifact, rev=match["sha"])
        print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    except Exception as err:
        print(f"[ERROR] Could not load {artifact}: {err}")


def print_experiments(exp_queries: list[str], artifact: str = ARTIFACT):
    """
    Print the YAML artifacts for multiple experiments.
    """
    for query in exp_queries:
        print_experiment_yaml(query, artifact)
        print()  # blank line after each
