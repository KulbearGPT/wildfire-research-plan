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


def test_cluster_guide_orders_read_only_gates_before_submission() -> None:
    text = (ROOT / "docs" / "cluster-migration.md").read_text(encoding="utf-8")
    ordered = [
        "Clone the reviewed commit",
        "Create the site-local profile",
        "Qualify the environment",
        "Transfer and verify HDF5",
        "Fetch pinned upstream code and weights",
        "Run the one-batch smoke",
        "Run Fold-2 test-only equivalence",
        "Run the 500-step calibration",
    ]
    positions = [text.index(item) for item in ordered]

    assert positions == sorted(positions)


def test_cluster_docs_preserve_execution_and_environment_boundaries() -> None:
    guide = (ROOT / "docs" / "cluster-migration.md").read_text(encoding="utf-8")
    environment = (ROOT / "environments" / "README.md").read_text(encoding="utf-8")

    assert (
        "No cluster scientific job has been run by this repository reorganization"
        in guide
    )
    assert "do not run the Windows-fixed controllers on Linux" in guide
    normalized_environment = " ".join(environment.split())
    assert "numerical executable equivalence" in normalized_environment
    assert "not byte-identical hardware equivalence" in normalized_environment
    assert (
        "Freeze exact training versions only after the target-cluster smoke"
        in normalized_environment
    )


def test_cluster_docs_identify_nibi_and_its_storage_boundaries() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "cluster-migration.md").read_text(encoding="utf-8")
    environment = (ROOT / "environments" / "README.md").read_text(encoding="utf-8")

    assert "Alliance Nibi" in readme
    for literal in (
        "https://docs.alliancecan.ca/wiki/Nibi",
        "nibi.alliancecan.ca",
        "--gpus=h100:1",
        "$SLURM_TMPDIR",
        "80 GB",
    ):
        assert literal in guide
    assert "/project" in guide
    assert "/scratch" in guide
    assert "Nibi" in environment


def test_cluster_guide_contains_exact_first_day_command_boundaries() -> None:
    text = (ROOT / "docs" / "cluster-migration.md").read_text(encoding="utf-8")
    literals = (
        "profile validate configs/cluster/profile.env",
        "manifest verify \"${DATA_ROOT}\"",
        "--require-production-contract",
        "submit.sh configs/cluster/profile.env",
        "--dry-run --",
        "smoke_fold2.py",
        "official_weight_entrypoint.py",
        "--trainer.max_steps=500",
        "--do_test=false",
    )
    for literal in literals:
        assert literal in text
    assert text.count('--upstream-root "${OFFICIAL_UPSTREAM_ROOT}"') == 1
    assert text.count('--upstream-root "${DERIVED_UPSTREAM_ROOT}"') == 2


def test_cluster_docs_explain_automatic_formal_preflight_and_nested_outputs() -> None:
    guide = (ROOT / "docs" / "cluster-migration.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for literal in (
        "automatically verifies the 999-file production manifest",
        "PyTorch, CUDA, and the single visible GPU",
        "official upstream commit",
        "official weight filename and size",
        '"${RUN_DIR}/work/smoke.json"',
        '"${RUN_DIR}/work/metrics.csv"',
        "WANDB_MODE=disabled",
    ):
        assert literal in guide
    assert "sha256sum" not in guide
    assert "manifest SHA-256" not in guide
    assert "formal preflight before the scientific command" in readme


def test_new_cluster_files_do_not_embed_local_or_site_specific_secrets() -> None:
    files = [
        ROOT / "README.md",
        ROOT / "docs" / "research-roadmap.md",
        ROOT / "docs" / "cluster-migration.md",
        ROOT / "environments" / "README.md",
        ROOT / "configs" / "cluster" / "profile.example.env",
        ROOT / "scripts" / "cluster" / "clusterctl.py",
        *(ROOT / "scripts" / "cluster").glob("*.sh"),
        ROOT / "manifests" / "weights" / "README.md",
    ]
    forbidden = (
        "D:\\WildFire Project",
        "C:\\Users\\Ji",
        "def-",
        "rrg-",
        "Bearer ",
        "hf_",
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in text, f"{value!r} leaked into {path}"
