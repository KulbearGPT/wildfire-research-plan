"""Publication contract for the committed twelve-fold released-weight result."""

import re
from html.parser import HTMLParser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "official-weight-12fold-20260812T052555Z"
GENERATION = "generations/3c076108f46e4b519e65f8603b9d97a5"
PUBLICATION_SHA256 = "31547d503f4217d7a2654aebbb7d46f14798c9cd515beb1544e8ad8bb279a2c1"
CSV_SHA256 = "55f90d2d6fe4ff175d888873188d7db7e86f0f9f7026e52a782816973b78605b"
SUMMARY_SHA256 = "769412aa56b3d972422a58771fa3fdcd82640959557b8d7c5bd5a15ed7557ff5"
INDEPENDENT_SHA256 = "a24a444783440b1d65ee7620e26b1476ca56d654fe70c2c12b871b6a07841542"
FOLD_AP_VALUES = (
    "0.5276636481285095", "0.4256492853164673", "0.5709022879600525",
    "0.3066331744194031", "0.483346164226532", "0.3223552405834198",
    "0.5765069723129272", "0.4736124575138092", "0.4777773916721344",
    "0.4709045886993408", "0.3237844705581665", "0.47403237223625183",
)
CSV_FILENAME_AP_DELTAS = (
    "-9.77120399474618e-05",
    "-9.541130065915393e-05",
    "3.237223625185415e-05",
)
RESULT_BOUNDARY_MARKERS = (
    "0.45276400446891785",
    "0.08821731990844857",
    "Fold 3",
    "0.3066331744194031",
    "Fold 6",
    "0.5765069723129272",
    "0.460 +/- 0.084",
    "upstream.lock.json paper.target",
    "0.45291666666666663",
    "0.08827179460179917",
    "official weight manifest filename labels",
)


def _missing_markers(path: Path, markers: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [marker for marker in markers if marker not in text]


def _translation_pairs(path: Path) -> set[tuple[str, str]]:
    """Return literal Chinese/English pairs from the website translation map."""
    text = path.read_text(encoding="utf-8")
    return {
        (source, translation)
        for source, translation in re.findall(
            r'\["([^"]*)", "([^"]*)"\]', text
        )
    }


class _Element:
    def __init__(self, tag: str, attributes: dict[str, str]) -> None:
        self.tag = tag
        self.attributes = attributes
        self.children: list[_Element | str] = []


class _DocumentParser(HTMLParser):
    _VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Element("document", {})
        self.stack = [self.root]

    def handle_starttag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        element = _Element(tag, {key: value or "" for key, value in attributes})
        self.stack[-1].children.append(element)
        if tag not in self._VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attributes)
        if tag not in self._VOID_TAGS:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def _descendant_text_nodes(element: _Element) -> list[str]:
    nodes: list[str] = []
    for child in element.children:
        if isinstance(child, str):
            nodes.append(child)
        else:
            nodes.extend(_descendant_text_nodes(child))
    return nodes


def _find_publication_callout(element: _Element) -> _Element:
    classes = set(element.attributes.get("class", "").split())
    text = "".join(_descendant_text_nodes(element))
    if (
        element.tag == "div"
        and {"callout", "info"} <= classes
        and "官方发布权重十二折测试评估（2026-08-12）" in text
    ):
        return element
    for child in element.children:
        if not isinstance(child, str):
            try:
                return _find_publication_callout(child)
            except LookupError:
                pass
    raise LookupError("published twelve-fold callout is missing")


def _render_like_index_runtime(text: str, translations: dict[str, str]) -> str:
    trimmed = text.strip()
    leading = re.match(r"^\s*", text).group(0)
    trailing = re.search(r"\s*$", text).group(0)
    translated = translations.get(trimmed)
    if translated is not None:
        return (
            (leading if "\n" in leading else "")
            + translated
            + (trailing if "\n" in trailing else "")
        )
    return (
        text.replace("。", ".")
        .replace("；", ";")
        .replace("，", ",")
        .replace("：", ":")
        .replace("（", "(")
        .replace("）", ")")
        .replace("“", '"')
        .replace("”", '"')
        .replace("、", ", ")
    )


def test_experiment_evidence_publishes_the_committed_twelve_fold_result() -> None:
    """Catch removal of the released-weight result, seals, or provenance boundary."""
    path = REPOSITORY_ROOT / "docs/experiments/res18_unet_t1_reproduction.md"
    missing = _missing_markers(
        path,
        (
            CAMPAIGN,
            GENERATION,
            PUBLICATION_SHA256,
            CSV_SHA256,
            SUMMARY_SHA256,
            INDEPENDENT_SHA256,
            *FOLD_AP_VALUES,
            *CSV_FILENAME_AP_DELTAS,
            *RESULT_BOUNDARY_MARKERS,
            "test-only evaluation of released weights",
            "does not prove paper-table provenance identity",
            "official focal-alpha behavior is a separate future training ablation",
            "separate twelve-fold released-weight test aggregate",
        ),
    )
    assert not missing, f"experiment evidence is missing: {missing}"


def test_readme_publishes_twelve_fold_metrics_and_recovery_boundaries() -> None:
    """Catch loss of the reproducibility result or a misleading retry claim."""
    path = REPOSITORY_ROOT / "reproductions/wsts_res18_unet_t1/README.md"
    missing = _missing_markers(
        path,
        (
            CAMPAIGN,
            GENERATION,
            *FOLD_AP_VALUES,
            *CSV_FILENAME_AP_DELTAS,
            *RESULT_BOUNDARY_MARKERS,
            "5749.570263385773",
            "549.7098723649979",
            "Fold 0 recovery",
            "Fold 2 legacy/offline parser qualification",
            "no scientific evaluation child was relaunched",
            "test-only evaluation of released weights",
        ),
    )
    assert not missing, f"README is missing: {missing}"


def test_website_has_a_chinese_source_publication_binding() -> None:
    """Catch removal of the Chinese source result from the public site."""
    path = REPOSITORY_ROOT / "index.html"
    missing = _missing_markers(
        path,
        (
            "官方发布权重十二折测试评估（2026-08-12）",
            CAMPAIGN,
            GENERATION,
            *FOLD_AP_VALUES,
            *RESULT_BOUNDARY_MARKERS,
            "仅对发布权重进行测试评估，并非十二次新的训练运行",
            "不证明论文表格来源身份一致",
            "官方 focal-alpha 行为是未来单独训练消融",
        ),
    )
    assert not missing, f"website Chinese publication binding is missing: {missing}"


def test_website_has_the_matching_english_translation_binding() -> None:
    """Catch a Chinese-only site update that the translation map cannot render."""
    path = REPOSITORY_ROOT / "index.html"
    pairs = _translation_pairs(path)
    expected_pairs = {
        (
            "官方发布权重十二折测试评估（2026-08-12）",
            "Published twelve-fold released-weight test evaluation (2026-08-12)",
        ),
        (
            "的发布权重测试结果。仅对发布权重进行测试评估，并非十二次新的训练运行。AP（Fold 0--11）依次为：0.5276636481285095、0.4256492853164673、0.5709022879600525、0.3066331744194031、0.483346164226532、0.3223552405834198、0.5765069723129272、0.4736124575138092、0.4777773916721344、0.4709045886993408、0.3237844705581665、0.47403237223625183。",
            " is a published released-weight result. This is a test-only evaluation of released weights, not twelve new training runs. AP (Fold 0--11), in order: 0.5276636481285095, 0.4256492853164673, 0.5709022879600525, 0.3066331744194031, 0.483346164226532, 0.3223552405834198, 0.5765069723129272, 0.4736124575138092, 0.4777773916721344, 0.4709045886993408, 0.3237844705581665, 0.47403237223625183.",
        ),
        ("已提交 campaign", "Committed campaign "),
        ("、generation", ", generation "),
        ("均值", "Mean"),
        ("总体标准差", "Population std"),
        ("最小值（折）", "Min (fold)"),
        ("最大值（折）", "Max (fold)"),
        ("总计 5749.570263385773", "Total 5749.570263385773"),
        ("每折中位数 549.7098723649979", "Median per fold 549.7098723649979"),
        ("论文参考为", "The paper reference is "),
        ("，provenance 为", ", with provenance "),
        ("。文件名参考为", ". The filename reference is "),
        (
            "，provenance 为 official weight manifest filename labels，明确不证明论文表格来源身份一致。GPU 观测边界为 5,756 个样本和 15,525--19,976 MiB 的采样峰值；WDDM 子进程归因不可用，观测样本最大值可能遗漏瞬态。",
            ", with provenance official weight manifest filename labels; it is explicitly not paper-table provenance. The GPU observation boundary is 5,756 samples and sampled peaks of 15,525--19,976 MiB; WDDM child attribution is unavailable and observed sample maxima may miss transients.",
        ),
        (
            "保留的 Fold 0 recovery 仅离线恢复已有科学输出；没有重新启动科学评估子进程。Fold 2 legacy/offline parser qualification 是解析与溯源边界，不是科学重试；保留失败和一次性 continuation 边界均在证据中。该一致性支持发布权重的可执行复现，但不证明论文表格来源身份一致。官方 focal-alpha 行为是未来单独训练消融，不属于本基线主张。",
            "The preserved Fold 0 recovery only finalized existing scientific output offline; no scientific evaluation child was relaunched. The Fold 2 legacy/offline parser qualification is a parsing and provenance boundary, not a scientific retry; preserved failures and the one-time continuation boundary remain in evidence. The agreement supports released-weight executable reproducibility but does not prove paper-table provenance identity. The official focal-alpha behavior is a separate future training ablation and is not part of this baseline claim.",
        ),
    }
    assert expected_pairs <= pairs, f"website English translation pairs are missing: {expected_pairs - pairs}"


def test_website_runtime_reaches_and_cleanly_joins_every_publication_translation() -> None:
    """Exercise the real text-node trim/lookup/join contract used by index.html."""
    path = REPOSITORY_ROOT / "index.html"
    translations = dict(_translation_pairs(path))
    parser = _DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    text_nodes = _descendant_text_nodes(_find_publication_callout(parser.root))

    chinese_nodes = {
        text.strip()
        for text in text_nodes
        if re.search(r"[\u3400-\u9fff]", text)
    }
    unreachable = sorted(chinese_nodes - translations.keys())
    assert not unreachable, f"publication text nodes miss trimmed Map keys: {unreachable}"

    rendered = re.sub(
        r"\s+",
        " ",
        "".join(_render_like_index_runtime(text, translations) for text in text_nodes),
    ).strip()
    assert not re.search(r"[\u3400-\u9fff]", rendered), rendered
    assert (
        f"Committed campaign {CAMPAIGN}, generation {GENERATION} is a published "
        "released-weight result. This is a test-only evaluation of released "
        "weights, not twelve new training runs."
    ) in rendered
    assert (
        "The paper reference is 0.460 +/- 0.084, with provenance "
        "upstream.lock.json paper.target. The filename reference is "
        "0.45291666666666663 +/- 0.08827179460179917, with provenance "
        "official weight manifest filename labels"
    ) in rendered
