"""Keep lesson actions linear while preserving references as local disclosures."""
from html import escape, unescape
import re


def lesson_readings(fragment, slug, step):
    links = []
    seen = set()
    disclosures = [(m.start(), m.end()) for m in re.finditer(r'<details\b.*?</details>', fragment, re.S)]

    def collect(match):
        attrs, label = match.groups()
        if any(start <= match.start() < end for start,end in disclosures):
            return match.group(0)
        href = re.search(r'\bhref="([^"]+)"', attrs)
        if not href or 'download' in attrs or '<img' in label:
            return match.group(0)
        url = unescape(href.group(1))
        if url.startswith('#') or '.html' not in url.split('#')[0]:
            return match.group(0)
        if url not in seen:
            seen.add(url)
            links.append((url, label))
        return '<span class="reading-label">' + label + '</span>'

    body = re.sub(r'<a\b([^>]*)>(.*?)</a>', collect, fragment, flags=re.S)
    # Standalone link menus add no teaching content after collection.
    def remove_menu(match):
        remaining = re.sub(r'<span class="reading-label">(?:(?!</?span\b)[\s\S])*?</span>', '', match.group(1))
        return '' if not remaining.strip(' \n\t·→') else match.group(0)
    body = re.sub(r'<p>(.*?)</p>', remove_menu, body, flags=re.S)
    if not links:
        return body
    label = 'Practical instructions and references' if (slug, step) in {
        ('fold2','step2'),('fold2','step3'),('fold2','step4'),
        ('missingness','step4'),('erm','step5'),('schedule','step3')} else 'Help and further reading'
    rows = ''.join(f'<li><a data-course-reading href="{escape(url,quote=True)}">{text}</a></li>' for url,text in links)
    disclosure = f'<details class="step-readings" id="readings-{step}"><summary>{label} · {len(links)}</summary><p>Open a resource when needed. Its return button brings you back to this exact lesson step.</p><ul>{rows}</ul></details>'
    return body[:-len('</section>')] + disclosure + '</section>'
