import re


_BOILERPLATE = re.compile(
    r"(click here|read more|see also|last updated|all rights reserved"
    r"|privacy policy|terms of (service|use)|cookie policy"
    r"|subscribe to|sign up for|follow us on"
    r"|share this (article|post|page)"
    r"|©\s*\d{4})",
    re.IGNORECASE,
)

_HTML_TAG = re.compile(r"<[^>]+>")
_MD_HEADER = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_MD_BOLD_ITALIC = re.compile(r"\*{1,3}(.*?)\*{1,3}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_JSON_EMPTY = re.compile(r'"\w+":\s*(null|""|0|\[\]|\{\}),?\s*')
_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD_ITALIC.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _JSON_EMPTY.sub("", text)

    # Remove lines that are pure boilerplate
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not _is_boilerplate_line(stripped):
            lines.append(stripped)
    text = "\n".join(lines)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _is_boilerplate_line(line: str) -> bool:
    if len(line) < 5:
        return True
    if _BOILERPLATE.search(line):
        # Only drop if the line is mostly boilerplate (< 60 chars real content)
        return len(line) < 80
    return False
