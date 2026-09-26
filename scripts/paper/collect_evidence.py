#!/usr/bin/env python3
# Code lifecycle: active_support. Paper evidence and figure preparation.
# Scope and settings: paper/materials/README.md; docs/CODE_LIFECYCLE.md.
"""Collect compact historical research evidence without loading models or data.

Lifecycle: active_support. Settings: docs/CODE_LIFECYCLE.md.
Run on a Slurm CPU node. Source JSONs are copied byte for byte; derived rows
retain both source role and comparator. The collector never upgrades a screen
or a retrospective composition into a new efficacy experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess

SCENARIOS = ("M00", "M01", "M06", "M07")
CANONICAL_REPORTS = {
    "x22-cosine-erm-final.json": (("ERM", "control"), ("X22", "candidate")),
    "x14-block-specialist-final.json": (("X14_RAW", "candidate"), ("X14_ROUTE", "effective")),
    "x17-severity-factorized-final.json": (("X17_MILD", "mild"), ("X17_SEVERE", "severe"), ("X17_ROUTE", "effective")),
    "x22-x17-complete-route-final.json": (("X22_X17_HISTORICAL", "effective"),),
    "global-consistency-vs-erm-final.json": (("GLOBAL_D1", "candidate"),),
    "x8-impact-fire-route-final.json": (("X8_RAW", "candidate"), ("X8_ROUTE_VS_GLOBAL", "effective")),
    "x8-impact-fire-route-vs-erm-final.json": (("X8_ROUTE_VS_ERM", "effective"),),
    "x10-dynamic-inpaint-route-final.json": (("X10_RAW", "candidate"), ("X10_ROUTE", "effective")),
    "x8-x17-complete-route-final.json": (("X8_X17_ROUTE", "effective"),),
    "x8-x10-route-final.json": (("X8_X10_ROUTE", "effective"),),
}
RAW_METHODS = {
    "ERM": ("control", None), "X22": ("cosine_erm", None),
    "X14_RAW": ("block_specialist", None), "X17_MILD": ("block_specialist", 0.25),
    "X17_SEVERE": ("block_specialist", 0.5), "GLOBAL_D1": ("global_consistency", None),
    "X8_RAW": ("impact_consistency", None), "X10_RAW": ("dynamic_inpaint", None),
}
ROUTE_COMPONENTS = {
    "X14_ROUTE": ("ERM", "ERM", "X14_RAW", "X14_RAW"),
    "X17_ROUTE": ("ERM", "ERM", "X17_MILD", "X17_SEVERE"),
    "X22_X17_HISTORICAL": ("X22", "X22", "X17_MILD", "X17_SEVERE"),
    "X8_ROUTE_VS_GLOBAL": ("GLOBAL_D1", "X8_RAW", "GLOBAL_D1", "GLOBAL_D1"),
    "X8_ROUTE_VS_ERM": ("ERM", "X8_RAW", "GLOBAL_D1", "GLOBAL_D1"),
    "X10_ROUTE": ("ERM", "ERM", "X10_RAW", "X10_RAW"),
    "X8_X17_ROUTE": ("ERM", "X8_RAW", "X17_MILD", "X17_SEVERE"),
    "X8_X10_ROUTE": ("ERM", "X8_RAW", "X10_RAW", "X10_RAW"),
}
REPO_JSONS = (
    "docs/research/method-inventory.json", "docs/experiments/three_directions_manifest.json",
    "docs/experiments/three_directions_screen.json", "docs/experiments/three_directions_costs.json",
    "docs/experiments/three_directions_fixed_bn_diagnostic.json", "docs/experiments/three_directions_jobs.json",
    "docs/experiments/three-directions-screen.json", "docs/experiments/three-directions-jobs.json",
    "docs/experiments/three-directions-statistics-audit.json", "docs/experiments/three-directions-t1-checkpoints.json",
    "docs/experiments/three-directions-t5-checkpoints.json", "reproductions/three_directions/sources.json",
)
BASELINE_DIRECTORIES = {
    "B0": ("corrected-B0-S0-3K-21093263", "heldout-B0-2022-21098505", "heldout-B0-2023-21098506"),
    "B2": ("corrected-B2-S0-3K-21093265", "heldout-B2-2022-21098505", "heldout-B2-2023-21098506"),
    "B3": ("corrected-B3-S0-3K-21093266", "heldout-B3-2022-21099907", "heldout-B3-2023-21099907"),
    "D1_ERM": ("D1-erm-S0-3K-21102676", "heldout-D1-ERM-2022-21105328", "heldout-D1-ERM-2023-21105328"),
    "D1_KL": ("D1-kl-S0-3K-21102677", "heldout-D1-KL-2022-21105328", "heldout-D1-KL-2023-21105328"),
    "D2_STD": ("D2-standard-S0-3K-21094929", "heldout-D2-STD-2022-21122938", "heldout-D2-STD-2023-21122938"),
    "D12_SARP": ("D12-SARP-S0-3K-21122172", "heldout-D12-SARP-2022-21122938", "heldout-D12-SARP-2023-21122938"),
}
FROZEN_JOBS = {1: {2021: "21212205", 2022: "21326336", 2023: "21326337"},
               5: {2021: "21212423", 2022: "21323907", 2023: "21323908"}}
ARCHIVED_SCREEN_DIRECTORIES = {
    "B4": "corrected-B4-S0-3K-21094665", "D2_RNC": "D2-rnc-S0-3K-21094930",
    "D4": "D4-token-S0-3K-21103691", "D5": "D5-CIWC-S0-3K-21111043",
    "D6": "D6-CIRC-S0-3K-21112470", "D7": "D7-CRA-S0-3K-21114921",
    "D8": "D8-FFCA-S0-3K-21116301", "D9": "D9-CEPR-S0-3K-21117395",
    "D10": "D10-RPP-S0-3K-21120041", "D11": "D11-CRPP-S0-3K-21121610",
    "D13_STD": "D13-STD-T5-S0-3K-21149784", "D13_SARP": "D13-SARP-T5-S0-3K-21149785",
}


def read_json(path):
    if path.stat().st_size > 8_000_000:
        raise ValueError(f"Refusing a noncompact evidence file: {path}")
    return json.loads(path.read_text())


def metrics(payload):
    result = payload.get("results", {})
    if not isinstance(result, dict) or not all(s in result for s in SCENARIOS):
        return None
    result = {s: result[s].get("metrics", result[s]) for s in SCENARIOS}
    if not all(isinstance(result[s], dict) and "avg_precision" in result[s] for s in SCENARIOS):
        return None
    for s in SCENARIOS:
        value = result[s]["avg_precision"]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid AP for {s}: {value}")
    return result


def bounded_summary_paths(runs):
    """Metadata-only scan: named run families, maximum three directory levels."""
    prefixes = ("cross-history-", "corrected-", "heldout-", "D1-", "D2-", "D12-", "three-directions-")
    skip = {"work", "source", "env", "venv", "checkpoints", "hdf5", "data", "logs", "wandb"}
    for directory in sorted(runs.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(prefixes):
            continue
        for root, dirs, files in os.walk(directory):
            depth = len(Path(root).relative_to(directory).parts)
            dirs[:] = sorted(d for d in dirs if d not in skip and depth < 2)
            for name in sorted(files):
                if name == "summary.json" or name.endswith("-summary.json"):
                    yield Path(root) / name


class Store:
    def __init__(self, output):
        self.output = output
        self.sources = []
        self.by_path = {}

    def add(self, path, kind):
        path = path.resolve()
        if str(path) in self.by_path:
            return self.by_path[str(path)]
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        identifier = f"s{len(self.sources) + 1:04d}"
        # Path-independent stable payload filename; same bytes have same name.
        rel = f"sources/{digest[:16]}-{path.name}"
        target = self.output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        item = {"source_id": identifier, "original_path": str(path), "source_rel": rel,
                "sha256": digest, "bytes": len(content), "kind": kind}
        self.sources.append(item)
        self.by_path[str(path)] = item
        return item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--runs", type=Path, default=Path("/project/6085198/kulbear/wildfire/runs"))
    parser.add_argument("--archive-runs", type=Path, help="Archived corrected D-series run tree; defaults beside runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Evidence scans, hashing and numeric collection require a Slurm CPU allocation")
    args.output.mkdir(parents=True, exist_ok=True)
    archive_runs = args.archive_runs or args.runs.parent / "archive/pre-t1-cleanup-2026-09-04/runs"
    store = Store(args.output)
    gaps = []
    scan = []
    for path in bounded_summary_paths(args.runs):
        try:
            payload = read_json(path)
            values = metrics(payload)
            if values:
                scan.append((path, payload, values))
        except (ValueError, OSError) as exc:
            gaps.append({"path": str(path), "reason": str(exc)})
    discovery = {"job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
                 "summary_count": len(scan), "summaries": [
                     {"path": str(p), "keys": list(x), "history": x.get("history"), "method": x.get("method"),
                      "seed": x.get("seed"), "year": x.get("year"), "block_fraction": x.get("block_fraction"),
                      "sample_counts": {s: v[s].get("sample_count") for s in SCENARIOS}}
                     for p, x, v in scan], "gaps": gaps}
    (args.output / "discovery.json").write_text(json.dumps(discovery, indent=2) + "\n")
    if args.discover_only:
        print(json.dumps({"summary_count": len(scan), "discovery": str(args.output / "discovery.json"), "job": os.environ["SLURM_JOB_ID"]}))
        return

    metadata_index = defaultdict(list)
    for path, payload, values in scan:
        identity = (payload.get("history"), payload.get("seed"), payload.get("year"))
        metadata_index[identity].append((path, payload, values))

    def get_run_sources(history, seed, year, vector, method):
        found = []
        for path, payload, values in metadata_index[(history, seed, year)]:
            expected_method, fraction = RAW_METHODS.get(method, (None, None))
            if expected_method and payload.get("method") != expected_method:
                continue
            if expected_method == "block_specialist" and payload.get("block_fraction") != fraction:
                continue
            if all(math.isclose(float(values[s]["avg_precision"]), float(vector[s]), abs_tol=1e-12, rel_tol=0) for s in SCENARIOS):
                found.append((path, payload, values))
        return found

    rows = []
    reports = {}
    analysis = args.runs / "cross-history-analysis"
    for path in sorted(analysis.glob("*.json")):
        try:
            payload = read_json(path)
        except (ValueError, OSError) as exc:
            gaps.append({"path": str(path), "reason": str(exc)})
            continue
        src = store.add(path, "historical_comparison")
        reports[path.name] = (payload, src)
    for name, definitions in CANONICAL_REPORTS.items():
        if name not in reports:
            gaps.append({"path": str(analysis / name), "reason": "Canonical completed report unavailable"})
            continue
        payload, src = reports[name]
        report_rows = payload.get("rows", [])
        identities = [(r["history"], r["seed"], r["year"]) for r in report_rows]
        expected = {(h, s, y) for h in (1, 5) for s in (0, 1, 2) for y in (2021, 2022, 2023)}
        if set(identities) != expected or len(identities) != 18:
            raise ValueError(f"Completed report {name} is not the exact 18-cell design")
        for method, role in definitions:
            for row_index, record in enumerate(report_rows):
                vector = record[role]
                originals = get_run_sources(record["history"], record["seed"], record["year"], vector, method) if method in RAW_METHODS else []
                if method in RAW_METHODS and not originals:
                    gaps.append({"method": method, "history": record["history"], "seed": record["seed"], "year": record["year"],
                                 "reason": "Original summary identity and full AP vector not recovered; aggregate retained"})
                original_sources = [store.add(p, "original_evaluation_summary") for p, _, _ in originals]
                sidecars = []
                for p, _, _ in originals:
                    for sidecar_name in ("started.json", "completed.json", "metadata.json"):
                        sidecar = p.parent / sidecar_name
                        if sidecar.is_file() and sidecar.stat().st_size < 1_000_000:
                            sidecars.append(store.add(sidecar, "original_run_metadata")["source_id"])
                for scenario in SCENARIOS:
                    ap = float(vector[scenario])
                    if not math.isfinite(ap) or not 0 <= ap <= 1:
                        raise ValueError(f"Invalid AP in {name}")
                    counts = {values[scenario].get("sample_count") for _, _, values in originals}
                    if len(counts) > 1:
                        raise ValueError(f"Conflicting populations for {method}: {record['history']}/{record['seed']}/{record['year']}")
                    count = next(iter(counts)) if counts else None
                    if count is not None and count != {2021: 3181, 2022: 2856, 2023: 2102}[record["year"]]:
                        raise ValueError(f"Unexpected paired evaluation population in {name}")
                    full_metrics = [values[scenario] for _, _, values in originals]
                    shared_metrics = full_metrics[0] if full_metrics and all(v == full_metrics[0] for v in full_metrics) else None
                    rows.append({"method": method, "history": record["history"], "seed": record["seed"],
                                 "year": record["year"], "scenario": scenario, "ap": ap,
                                 "source_id": src["source_id"], "source_rel": src["source_rel"],
                                 "source_role": role, "source_row_index": row_index,
                                 "evidence_level": "aggregate_and_original_summary" if originals else "historical_aggregate",
                                 "sample_count": count, "original_summary_source_ids": [s["source_id"] for s in original_sources],
                                 "run_metadata_source_ids": sorted(set(sidecars)), "metrics": shared_metrics,
                                 "provenance_notes": []})

    # Resolve composition populations through the explicit component selected for
    # each scenario. Preserve the historical effective AP instead of overwriting it.
    by_key = {(r["method"], r["history"], r["seed"], r["year"], r["scenario"]): r for r in rows}
    for row in rows:
        components = ROUTE_COMPONENTS.get(row["method"])
        if not components:
            continue
        component = components[SCENARIOS.index(row["scenario"])]
        source = by_key.get((component, row["history"], row["seed"], row["year"], row["scenario"]))
        if source and math.isclose(source["ap"], row["ap"], abs_tol=1e-12, rel_tol=0):
            row["component_method"] = component
            row["sample_count"] = source["sample_count"]
            row["original_summary_source_ids"] = source["original_summary_source_ids"]
            row["run_metadata_source_ids"] = source["run_metadata_source_ids"]
            row["metrics"] = source["metrics"]
            row["evidence_level"] = "scenario_composition_from_originals" if source["original_summary_source_ids"] else "scenario_composition_from_aggregates"
        else:
            raise ValueError(f"Expected route component does not match original aggregate: {row}")

    # Verify that the canonical reports use the explicitly named controls. Do
    # not assume every file's field named 'control' means the same baseline.
    for name in CANONICAL_REPORTS:
        if name not in reports:
            continue
        expected_control = "GLOBAL_D1" if name == "x8-impact-fire-route-final.json" else "ERM"
        for record in reports[name][0]["rows"]:
            for scenario in SCENARIOS:
                control_row = by_key[(expected_control, record["history"], record["seed"], record["year"], scenario)]
                if not math.isclose(control_row["ap"], record["control"][scenario], abs_tol=1e-12, rel_tol=0):
                    raise ValueError(f"Comparator mismatch in {name}: {expected_control}")
                if "delta" in record and not math.isclose(record["effective"][scenario] - record["control"][scenario],
                                                         record["delta"][scenario], abs_tol=1e-12, rel_tol=0):
                    raise ValueError(f"Saved scenario difference is inconsistent in {name}")
            for key, selected in (("primary_delta", SCENARIOS[1:]), ("block_delta", SCENARIOS[2:])):
                calculated = sum(record["effective"][s] - record["control"][s] for s in selected) / len(selected)
                if key in record and not math.isclose(calculated, record[key], abs_tol=1e-12, rel_tol=0):
                    raise ValueError(f"Saved {key} is inconsistent in {name}")

    repo_payloads = {}
    for relative in REPO_JSONS:
        path = args.repo / relative
        if path.is_file():
            repo_payloads[relative] = (read_json(path), store.add(path, "repository_evidence"))
        else:
            gaps.append({"path": str(path), "reason": "Repository evidence unavailable"})

    def add_original_rows(method, path, payload, values, history, seed, year, evidence_level="original_summary"):
        src = store.add(path, "original_evaluation_summary")
        for scenario in SCENARIOS:
            rows.append({"method": method, "history": history, "seed": seed, "year": year,
                         "scenario": scenario, "ap": values[scenario]["avg_precision"],
                         "source_id": src["source_id"], "source_rel": src["source_rel"],
                         "source_role": "results", "evidence_level": evidence_level,
                         "sample_count": values[scenario].get("sample_count"),
                         "original_summary_source_ids": [src["source_id"]], "run_metadata_source_ids": [],
                         "metrics": values[scenario], "provenance_notes": []})

    for method, directories in BASELINE_DIRECTORIES.items():
        for year, directory in zip((2021, 2022, 2023), directories):
            path = args.runs / directory / f"results-{year}" / "summary.json"
            if path.is_file():
                payload = read_json(path)
                add_original_rows("HIST_" + method, path, payload, metrics(payload), 1, 0, year, "historical_single_run_chain")
            else:
                gaps.append({"path": str(path), "reason": "Historical baseline summary unavailable"})

    for method, directory in ARCHIVED_SCREEN_DIRECTORIES.items():
        run_directory = archive_runs / directory
        paths = sorted(run_directory.glob("*/summary.json")) if run_directory.is_dir() else []
        candidates = []
        for path in paths:
            payload = read_json(path)
            values = metrics(payload)
            if values and payload.get("year") == 2021 and all(values[s].get("sample_count") == 3181 for s in SCENARIOS):
                candidates.append((path, payload, values))
        if len(candidates) == 1:
            path, payload, values = candidates[0]
            history = 5 if method.startswith("D13") else 1
            add_original_rows("HIST_" + method, path, payload, values, history, 0, 2021,
                              "archived_corrected_single_seed_screen")
        else:
            for path, _, _ in candidates:
                store.add(path, "ambiguous_archived_screen")
            gaps.append({"method": "HIST_" + method, "path": str(run_directory),
                         "reason": "Expected one complete 2021 screen; no arbitrary selection among alternatives",
                         "candidate_paths": [str(c[0]) for c in candidates], "inspected_paths": [str(p) for p in paths]})

    for history, years in FROZEN_JOBS.items():
        for year, job in years.items():
            candidates = [(p, q, v) for p, q, v in scan if re.search(rf"(?:^|[-/]){job}(?:/|$)", str(p)) and q.get("year") == year]
            if not candidates:
                gaps.append({"method": "FROZEN", "history": history, "year": year, "job": job,
                             "reason": "Frozen-reference original summary unavailable in bounded scan"})
            elif len(candidates) != 1:
                gaps.append({"method": "FROZEN", "history": history, "year": year, "job": job,
                             "reason": "Ambiguous frozen-reference summaries", "paths": [str(c[0]) for c in candidates]})
            else:
                p, q, v = candidates[0]
                add_original_rows("FROZEN", p, q, v, history, None, year, "frozen_single_checkpoint_reference")

    # RF and TD are distinct archived campaigns; preserve the original naming
    # and controls instead of combining equally named mechanisms across them.
    rf_payload, rf_src = repo_payloads["docs/experiments/three_directions_screen.json"]
    for item in rf_payload["rows"]:
        values = metrics(item)
        if not values:
            continue
        original = Path(item["source"])
        source = original if original.is_file() else args.repo / "docs/experiments/three_directions_screen.json"
        if original.is_file():
            original_payload = read_json(original)
            original_metrics = metrics(original_payload)
            if original_metrics != values:
                raise ValueError(f"RF compiled report differs from original metrics: {original}")
        else:
            gaps.append({"path": str(original), "reason": "RF original unavailable; compiled report retained"})
        add_original_rows("RF_" + item["method"], source, item, values, item["history"], item["seed"], item["year"],
                          "archived_single_seed_screen")
    td_payload, td_src = repo_payloads["docs/experiments/three-directions-screen.json"]
    for value in td_payload.get("runs", []):
        path = Path(value)
        if path.is_file():
            payload = read_json(path)
            values = metrics(payload)
            if values:
                mode = payload.get("mode", payload.get("method", path.parent.parent.name))
                add_original_rows("TD_" + str(mode), path, payload, values, payload.get("history"), payload.get("seed", 0),
                                  payload.get("year", 2021), "archived_single_seed_screen")
        else:
            gaps.append({"path": str(path), "reason": "TD original summary unavailable; aggregate remains in repository evidence"})

    keys = [(r["method"], r["history"], r["seed"], r["year"], r["scenario"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate normalized scientific cells; refusing implicit averaging")

    # Keep all remaining scientific comparisons as distinct source records. A
    # screen's exact comparator is whatever its original file states; no global
    # ranking or new causal interpretation is inferred from a filename.
    archived_comparisons = [{"artifact": name, "source_id": src["source_id"], "source_rel": src["source_rel"],
                             "method": payload.get("method"), "canonical": name in CANONICAL_REPORTS,
                             "row_count": len(payload.get("rows", [])), "payload": payload}
                            for name, (payload, src) in reports.items() if isinstance(payload, dict)]
    inventory, _ = repo_payloads["docs/research/method-inventory.json"]
    if len(inventory["methods"]) != 100:
        raise ValueError("The retained method inventory changed; audit its coverage before collection")
    source_commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    result = {"schema_version": 1, "source_commit": source_commit,
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "execution": {"job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
                            "scope": "Historical JSON scan, copying, hashing and numeric normalization; no model execution"},
              "scenarios": list(SCENARIOS), "sources": store.sources, "rows": rows,
              "archived_comparisons": archived_comparisons, "inventory": inventory["methods"],
              "repository_evidence": {k: {"source_id": v[1]["source_id"], "source_rel": v[1]["source_rel"]} for k, v in repo_payloads.items()},
              "gaps": gaps,
              "caveats": ["Historical test years were inspected during research development.",
                          "The X22+X17 historical artifact uses X22 for M00; the current CLI uses ERM for M00.",
                          "Old summary/started files do not certify completed training steps or byte identity of evaluated data.",
                          "Original-summary matching requires recorded identity plus the four-scenario AP vector; multiple equivalent records remain explicit.",
                          "A retained positive metric is not an adopted method; lifecycle and original decision remain in the full inventory."]}
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output / "evidence.json"), "rows": len(rows), "sources": len(store.sources),
                      "summary_count": len(scan), "gaps": len(gaps), "job": os.environ["SLURM_JOB_ID"]}))


if __name__ == "__main__":
    main()
