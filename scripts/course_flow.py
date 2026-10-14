"""Keep supporting links beside the paragraph that needs them."""
from html import escape, unescape
import re


READING_NOTES = {
    'student-start': 'Browser study needs no cluster account. For a practical, first verify access, storage, the selected environment and assets, and an agreed resource budget. Missing access does not prevent completing the labelled study exercises.',
    'student-bridge': 'Keep the official reproduction and project continuation as separate records. The project needs corrected data, training-only statistics and the actual B3 foundation with its checksum and provenance; official Fold 2 weights cannot replace B3.',
    'student-report': 'Keep planned and executed results separate. For each reported score, record the evaluated checkpoint, original metric path, population and condition. A conclusion must include regressions and the limits of the comparison.',
    'student-help': 'Name the blocker and the current task. For execution problems, retain the job ID, source revision, configuration, output path and relevant error. A queued job is not a failed run, and a failed run is not a scientific negative result.',
    'student-real-data': 'These panels show recorded observations from two potentially overlapping records. Read the date and panel legend first. Gray denotes deliberately hidden input, not observed background; persistence errors come from copying yesterday, not a trained model.',
    'evaluation-population': 'Use the same sample identities, preprocessing and missingness masks for both rows. Equal counts alone do not establish the same population. Record excluded samples and keep the comparison scope explicit.',
    'baselines': 'For this T1 comparison, identify B3 by its actual checkpoint and provenance. Its foundation selection stage is distinct from final-step selection for the later ERM continuation. B5 is a separate T5 extension.',
    'setup-recovery': 'Find the first failed stage before resubmitting. Preserve failed output and logs, repair the cause, and use a new output location for a replacement. Do not silently extend the agreed experiment budget.',
    'method-recipes': 'Use only the role required by the current task. ERM uses control; the shared recipe also includes cosine and specialists. A command preview does not submit a run, and a smoke run does not establish a performance result.',
}

def reading_key(url):
    parts = url.split('#')[0].split('/')
    return parts[-2] if parts[-1] == 'index.html' else parts[-1].removesuffix('.html')


def lesson_readings(fragment, slug, step):
    disclosures = [(m.start(), m.end()) for m in re.finditer(r'<details\b.*?</details>', fragment, re.S)]
    serial = 0

    def paragraph(match):
        nonlocal serial
        if any(start <= match.start() < end for start, end in disclosures):
            return match[0]
        links = []
        def collect(link):
            attrs, label = link.groups()
            href = re.search(r'\bhref="([^"]+)"', attrs)
            if not href or 'download' in attrs or '<img' in label:
                return link[0]
            url = unescape(href[1])
            if url.startswith('#') or '.html' not in url.split('#')[0]:
                return link[0]
            if (url, label) not in links:
                links.append((url, label))
            return '<span class="reading-label">'+label+'</span>'
        body = re.sub(r'<a\b([^>]*)>(.*?)</a>', collect, match[0], flags=re.S)
        if not links:
            return body
        serial += 1
        remaining = re.sub(r'<[^>]+>', '', re.sub(r'<span class="reading-label">.*?</span>', '', body, flags=re.S))
        if not remaining.strip(' \n\t·→'):
            body = ''
        rows = ''.join(f'<li><a data-course-reading href="{escape(url,quote=True)}">{label}</a></li>' for url,label in links)
        label = re.sub('<[^>]+>', '', links[0][1])
        notes = list(dict.fromkeys(READING_NOTES[reading_key(url)] for url,_ in links if reading_key(url) in READING_NOTES))
        default_open = ' open data-default-open' if notes else ''
        explanation = ''.join('<p>'+escape(note)+'</p>' for note in notes)
        return body+f'<details{default_open} class="task-reading" id="reading-{step}-{serial}"><summary>Supporting detail: {label}</summary>{explanation}<p class="source">Open the full reference only if you need more detail; its return button brings you back to this task.</p><ul>{rows}</ul></details>'

    return re.sub(r'<p\b[^>]*>.*?</p>', paragraph, fragment, flags=re.S)
