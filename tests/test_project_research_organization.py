from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_links_to_current_evidence_and_cluster_guides() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in (
        "docs/experiments/phase0.md",
        "docs/experiments/res18_unet_t1_reproduction.md",
        "docs/research-roadmap.md",
        "docs/cluster-migration.md",
        "baseline-reproduction/",
        "related-work/",
    ):
        assert target in text


def test_roadmap_orders_completed_gates_before_new_methods() -> None:
    text = (ROOT / "docs" / "research-roadmap.md").read_text(encoding="utf-8")
    headings = [
        "Stage 0 — Cluster migration and equivalence",
        "Stage 1 — Training-contract sensitivity",
        "Stage 2 — WSTS+ learned controls",
        "Stage 3 — Controlled-missingness diagnosis",
        "Stage 4 — Matched robust baselines and main method",
        "Stage 5 — Conditional extensions",
    ]
    positions = [text.index(heading) for heading in headings]

    assert positions == sorted(positions)


def test_roadmap_preserves_claim_boundaries() -> None:
    text = (ROOT / "docs" / "research-roadmap.md").read_text(encoding="utf-8")

    assert "controlled missingness" in text
    assert (
        "does not establish natural-missingness or operational-deployment performance"
        in text
    )
    assert "released-weight executable reproducibility" in text
    assert "does not prove paper-table provenance identity" in text
    assert "Candidate, primary-source verification required" in text
