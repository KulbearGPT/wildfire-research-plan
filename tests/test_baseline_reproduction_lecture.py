"""Publication contract for the beginner-friendly baseline lecture."""

from __future__ import annotations

import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LECTURE = REPOSITORY_ROOT / "baseline-reproduction" / "index.html"
RELATED_WORK = REPOSITORY_ROOT / "related-work" / "index.html"
HOME = REPOSITORY_ROOT / "index.html"

SECTION_IDS = (
    "why-baselines",
    "vocabulary",
    "target",
    "contract",
    "official-code",
    "data-folds",
    "environment",
    "calibration",
    "one-fold",
    "released-weights",
    "verification",
    "results",
    "failures",
    "boundaries",
    "checklist",
    "summary",
)

FOLD_AP_VALUES = (
    "0.5276636481285095",
    "0.4256492853164673",
    "0.5709022879600525",
    "0.3066331744194031",
    "0.483346164226532",
    "0.3223552405834198",
    "0.5765069723129272",
    "0.4736124575138092",
    "0.4777773916721344",
    "0.4709045886993408",
    "0.3237844705581665",
    "0.47403237223625183",
)

AGGREGATE_MARKERS = (
    "0.45276400446891785",
    "0.08821731990844857",
    "0.460 +/- 0.084",
    "0.45291666666666663",
    "0.08827179460179917",
    "0.3809294228752454",
    "0.23737221211194992",
    "0.6724123309055964",
    "0.26826225717862445",
)

BEGINNER_MARKERS = (
    "What is a baseline?",
    "What is a fold?",
    "What is a checkpoint?",
    "What is average precision?",
    "Why reproduce before inventing?",
    "Scientific contract",
    "Engineering compatibility",
    "Independent verifier",
    "What this result proves",
    "What this result does not prove",
)

BOUNDARY_MARKERS = (
    "test-only evaluation of official released weights",
    "not twelve new training runs",
    "released-weight executable reproducibility",
    "does not prove paper-table provenance identity",
    "no scientific evaluation child was relaunched",
    "seven unused eager architecture exports",
    "did not modify the model, loss, optimizer, data, or metrics",
    "sampled maxima may miss transients",
    "separate future training ablation",
)


class LectureParser(HTMLParser):
    """Collect the small subset of DOM facts needed by the publication test."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.sources: list[str] = []
        self.scripts: list[str] = []
        self._script: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if attributes.get("id"):
            self.ids.append(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.hrefs.append(attributes["href"])
        if tag in {"img", "script"} and attributes.get("src"):
            self.sources.append(attributes["src"])
        if tag == "script" and not attributes.get("src"):
            self._script = []

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None


def _parse_lecture() -> LectureParser:
    parser = LectureParser()
    parser.feed(LECTURE.read_text(encoding="utf-8"))
    return parser


def test_lecture_has_the_courseware_shell_and_complete_toc() -> None:
    assert LECTURE.is_file(), "baseline lecture page is missing"
    text = LECTURE.read_text(encoding="utf-8")
    parser = _parse_lecture()

    assert "03 · Baseline reproduction" in text
    assert "From a Paper Claim to a Verified Baseline" in text
    assert "A beginner-friendly walkthrough of reproducing Res18-U-Net with T=1" in text
    assert len(parser.ids) == len(set(parser.ids)), "HTML ids must be unique"
    assert all(section_id in parser.ids for section_id in SECTION_IDS)
    toc_targets = tuple(
        href.removeprefix("#")
        for href in parser.hrefs
        if href.startswith("#") and href.removeprefix("#") in SECTION_IDS
    )
    assert toc_targets == SECTION_IDS
    for control in (
        'id="navToggle"',
        'id="searchInput"',
        'id="fontDown"',
        'id="fontUp"',
        'id="themeToggle"',
        'id="progressBar"',
        "window.print()",
        "IntersectionObserver",
    ):
        assert control in text


def test_lecture_teaches_the_full_process_without_overclaiming() -> None:
    text = LECTURE.read_text(encoding="utf-8")
    missing = [
        marker
        for marker in (*BEGINNER_MARKERS, *BOUNDARY_MARKERS)
        if marker not in text
    ]
    assert not missing, f"beginner/process boundary markers are missing: {missing}"

    for stage in (
        "Freeze the question",
        "Audit the data",
        "Rebuild the environment",
        "Calibrate the runtime",
        "Complete one full fold",
        "Evaluate all twelve released weights",
        "Verify from raw evidence",
    ):
        assert stage in text

    assert "we trained twelve models" not in text.lower()
    assert "twelve training runs reproduced" not in text.lower()


def test_lecture_publishes_exact_verified_results_and_references() -> None:
    text = LECTURE.read_text(encoding="utf-8")
    missing = [
        marker
        for marker in (*FOLD_AP_VALUES, *AGGREGATE_MARKERS)
        if marker not in text
    ]
    assert not missing, f"verified numerical markers are missing: {missing}"
    assert "Fold 3" in text and "Fold 6" in text
    assert "5749.570263385773" in text
    assert "549.7098723649979" in text
    assert "15,525--19,976 MiB" in text
    assert "res18_unet_t1_reproduction.md" in text


def test_navigation_and_all_local_resources_resolve() -> None:
    home_text = HOME.read_text(encoding="utf-8")
    assert "baseline-reproduction/" in home_text
    assert "配套课程：基线复现完整流程" in home_text
    assert "Companion courseware: the complete baseline reproduction" in home_text
    assert "英文网页讲义，从论文结果到可验证基线，解释官方代码、数据划分、运行校准、十二折评估与独立核验。" in home_text
    assert "An English web lecture that moves from a paper result to a verified baseline" in home_text
    assert "../baseline-reproduction/" in RELATED_WORK.read_text(encoding="utf-8")
    parser = _parse_lecture()
    lecture_root = LECTURE.parent
    for reference in (*parser.hrefs, *parser.sources):
        if (
            not reference
            or reference.startswith(("#", "http://", "https://", "mailto:"))
        ):
            continue
        target = (lecture_root / reference.split("#", 1)[0]).resolve()
        assert target.is_relative_to(REPOSITORY_ROOT.resolve())
        assert target.exists(), f"local lecture resource does not resolve: {reference}"


def test_inline_javascript_parses() -> None:
    parser = _parse_lecture()
    assert parser.scripts, "courseware interaction script is missing"
    for script in parser.scripts:
        completed = subprocess.run(
            ["node", "--check"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
