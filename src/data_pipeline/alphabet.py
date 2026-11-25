"""Uzbek text -> 2026 Latin alphabet (ğ, ö, ş, ç).

Handles Cyrillic, the 1995 Latin alphabet (gʻ, oʻ, sh, ch) and mixed input.
"""

import re
import unicodedata

TUTUQ = "ʼ"  # ʼ
_APOSTROPHES = "'ʼ’`‘ʻ"

_URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
_APOS_BETWEEN_LETTERS = re.compile(rf"(?<=[^\W\d_])[{_APOSTROPHES}](?=[^\W\d_])")
_OLD_LATIN = re.compile(rf"([GgOo]){TUTUQ}|([SsCc])([Hh])")

_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "ts", "ч": "ç", "ш": "ş", "щ": "şç", "ъ": TUTUQ,
    "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya", "ў": "ö", "қ": "q",
    "ғ": "ğ", "ҳ": "h",
}
_CYR_VOWELS = set("аеёиоуэюяў")


def _old_latin_sub(m: re.Match) -> str:
    if m.group(1):
        return {"g": "ğ", "G": "Ğ", "o": "ö", "O": "Ö"}[m.group(1)]
    first = m.group(2)
    out = "ş" if first in "Ss" else "ç"
    return out.upper() if first.isupper() else out


def _cyrillic_to_latin(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        low = ch.lower()
        if low not in _CYR:
            out.append(ch)
            continue
        lat = _CYR[low]
        # Word-initial or post-vowel "е" is read as "ye" (ер -> yer).
        if low == "е":
            prev = text[i - 1].lower() if i > 0 else ""
            if not prev.isalpha() or prev in _CYR_VOWELS or prev in "ъь":
                lat = "ye"
        if ch.isupper() and lat:
            neighbours = text[max(i - 1, 0) : i] + text[i + 1 : i + 2]
            all_caps = len(lat) > 1 and any(c.isupper() for c in neighbours)
            lat = lat.upper() if all_caps or len(lat) == 1 else lat.capitalize()
        out.append(lat)
    return "".join(out)


def _convert_chunk(text: str) -> str:
    text = _APOS_BETWEEN_LETTERS.sub(TUTUQ, text)
    # Latin pass runs before Cyrillic so a converted "ъ" after г/о is not re-read as gʻ/oʻ.
    text = _OLD_LATIN.sub(_old_latin_sub, text)
    return _cyrillic_to_latin(text)


def to_new_latin(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    parts = _URL.split(text)
    # Odd indices are the captured URLs/emails; leave them as they are.
    return "".join(p if i % 2 else _convert_chunk(p) for i, p in enumerate(parts))


_NEW_TO_OLD = {"ğ": "gʻ", "ö": "oʻ", "ş": "sh", "ç": "ch"}


def to_old_latin(text: str) -> str:
    """2026 alphabet -> 1995 Latin, for software that still expects gʻ, oʻ, sh, ch."""
    out = []
    for i, ch in enumerate(text):
        old = _NEW_TO_OLD.get(ch.lower())
        if old is None:
            out.append(ch)
        elif not ch.isupper():
            out.append(old)
        else:
            neighbours = text[max(i - 1, 0) : i] + text[i + 1 : i + 2]
            all_caps = any(c.isupper() for c in neighbours)
            out.append(old.upper() if all_caps and old[1] == "h" else old.capitalize())
    return "".join(out)


def detect_script(text: str) -> str:
    cyr = lat = False
    for ch in text:
        if not ch.isalpha():
            continue
        if "Ѐ" <= ch <= "ӿ":
            cyr = True
        elif unicodedata.name(ch, "").startswith("LATIN"):
            lat = True
    if cyr and lat:
        return "mixed"
    return "cyrillic" if cyr else "latin" if lat else "none"
