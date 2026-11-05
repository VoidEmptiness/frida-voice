
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Числа словами перед озвучкой: 1 000 000 → «один миллион», 5-м → «пятом», 50% → «пятьдесят процентов».

Вызывается до punctuate.py: тот пересобирает текст по ``[\\w-]+`` и теряет «%» и
десятичные разделители — «3,5» доезжает до TTS как «3 5».
"""

from __future__ import annotations

import re
from functools import lru_cache

from num2words import num2words

LANGS = ("ru", "en")

# «1 000» пишут обычным, неразрывным или узким неразрывным пробелом
_SEP = "\u00a0\u202f "

_PERCENT = (("процент", "процента", "процентов"), "m")
_RUBLE = (("рубль", "рубля", "рублей"), "m")
_KOPECK = (("копейка", "копейки", "копеек"), "f")
_HRYVNIA = (("гривна", "гривны", "гривен"), "f")
_DOLLAR = (("доллар", "доллара", "долларов"), "m")
_CENT = (("цент", "цента", "центов"), "m")
_EURO = (("евро", "евро", "евро"), "n")
_POUND = (("фунт", "фунта", "фунтов"), "m")
_YEN = (("иена", "иены", "иен"), "f")
_YUAN = (("юань", "юаня", "юаней"), "m")

_PERCENT_EN = (("percent", "percent", "percent"), "m")
_DOLLAR_EN = (("dollar", "dollar", "dollars"), "m")
_CENT_EN = (("cent", "cent", "cents"), "m")
_EURO_EN = (("euro", "euro", "euros"), "n")
_POUND_EN = (("pound", "pound", "pounds"), "m")
_YEN_EN = (("yen", "yen", "yen"), "n")

# написание единицы → (формы [один, единственного-дробного, много], род для «один/два»)
_UNITS: dict[str, dict[str, tuple[tuple[str, str, str], str]]] = {
    "ru": {
        "%": _PERCENT,
        "процент": _PERCENT,
        "процента": _PERCENT,
        "процентов": _PERCENT,
        "руб": _RUBLE,
        "рубль": _RUBLE,
        "рубля": _RUBLE,
        "рублей": _RUBLE,
        "₽": _RUBLE,
        "коп": _KOPECK,
        "копейка": _KOPECK,
        "копейки": _KOPECK,
        "копеек": _KOPECK,
        "грн": _HRYVNIA,
        "гривна": _HRYVNIA,
        "гривны": _HRYVNIA,
        "гривен": _HRYVNIA,
        "$": _DOLLAR,
        "usd": _DOLLAR,
        "доллар": _DOLLAR,
        "доллара": _DOLLAR,
        "долларов": _DOLLAR,
        "€": _EURO,
        "eur": _EURO,
        "евро": _EURO,
        "£": _POUND,
        "gbp": _POUND,
        "фунт": _POUND,
        "фунта": _POUND,
        "фунтов": _POUND,
        "¥": _YEN,
        "jpy": _YEN,
        "иена": _YEN,
        "иены": _YEN,
        "иен": _YEN,
        "юань": _YUAN,
        "юаня": _YUAN,
        "юаней": _YUAN,
        "cny": _YUAN,
        "ц": _CENT,
        "цент": _CENT,
        "цента": _CENT,
        "центов": _CENT,
    },
    "en": {
        "%": _PERCENT_EN,
        "percent": _PERCENT_EN,
        "$": _DOLLAR_EN,
        "usd": _DOLLAR_EN,
        "dollar": _DOLLAR_EN,
        "dollars": _DOLLAR_EN,
        "cent": _CENT_EN,
        "cents": _CENT_EN,
        "€": _EURO_EN,
        "eur": _EURO_EN,
        "euro": _EURO_EN,
        "euros": _EURO_EN,
        "£": _POUND_EN,
        "gbp": _POUND_EN,
        "pound": _POUND_EN,
        "pounds": _POUND_EN,
        "¥": _YEN_EN,
        "jpy": _YEN_EN,
        "yen": _YEN_EN,
    },
}

# окончание порядкового → (тип, падеж, род, множественное число).
# Тип «c» — количественное: «2-х комнатная» читается «двух», а не «вторых».
_ORDINGS: dict[str, tuple[str, str, str, bool]] = {
    "й": ("o", "n", "m", False),
    "ый": ("o", "n", "m", False),
    "ий": ("o", "n", "m", False),
    "ой": ("o", "n", "m", False),
    "я": ("o", "n", "f", False),
    "ая": ("o", "n", "f", False),
    "яя": ("o", "n", "f", False),
    "е": ("o", "n", "n", False),
    "ее": ("o", "n", "n", False),
    "ое": ("o", "n", "n", False),
    "и": ("o", "n", "m", True),
    "го": ("o", "g", "m", False),
    "гого": ("o", "g", "m", False),
    "ого": ("o", "g", "m", False),
    "его": ("o", "g", "m", False),
    "ей": ("o", "g", "f", False),
    "ых": ("o", "g", "m", True),
    "их": ("o", "g", "m", True),
    "му": ("o", "d", "m", False),
    "ому": ("o", "d", "m", False),
    "ему": ("o", "d", "m", False),
    "ым": ("o", "d", "m", True),
    "им": ("o", "d", "m", True),
    "м": ("o", "p", "m", False),
    "мя": ("c", "i", "m", False),
    "ю": ("o", "a", "f", False),
    "ую": ("o", "a", "f", False),
    "х": ("c", "g", "m", False),
}

_DIGITS = {
    "ru": "ноль один два три четыре пять шесть семь восемь девять".split(),
    "en": "zero one two three four five six seven eight nine".split(),
}

_NEG = {"ru": "минус ", "en": "minus "}

# форма существительного при целом числе по последней цифре:
# 1 → «процент», 2-4 → «процента», 0 и 5-9 → «процентов»
_RU_CLASS = (2, 0, 1, 1, 1, 2, 2, 2, 2, 2)

_ORDINAL_TOKEN = {
    "ru": r"(?P<ord>\d+)\s*-\s*(?P<suf>[а-яё]+)",
    "en": r"(?P<ord>\d+)(?P<suf>st|nd|rd|th)",
}

# цепочка из трёх и более групп через точку (дата dd.mm.yyyy, IP, версия) и
# время читаются словами хуже, чем цифрами — оставляем как написано
_SKIP_TOKEN = r"(?P<skip>\d+(?:\.\d+){2,}|\d{1,2}:\d{2})"

_NUM_TOKEN = {
    "ru": (
        r"(?P<grp>\d{1,3}(?:[" + _SEP + r"\.,]\d{3})+)"
        r"|(?P<dec>\d+[,.]\d{1,3}(?![,.]?\d)|\d+\.\d{1,2}(?![,.]?\d))"
        r"|(?P<int>\d+)"
    ),
    "en": (
        r"(?P<grp>\d{1,3}(?:[" + _SEP + r",]\d{3})+)"
        r"|(?P<dec>\d+\.\d+)"
        r"|(?P<int>\d+)"
    ),
}

_HAS_DIGIT = re.compile(r"\d")
_SPACES = re.compile(r"\s{2,}")
_PATTERNS: dict[str, re.Pattern[str]] = {}


def _pattern(lang: str) -> re.Pattern[str]:
    if lang not in _PATTERNS:
        units = "|".join(
            re.escape(u) for u in sorted(_UNITS[lang], key=len, reverse=True)
        )
        prefix = r"(?P<pre>[₽$€£¥]\s*)?"
        _PATTERNS[lang] = re.compile(
            "(?<!\\w)"
            + prefix
            + r"(?P<minus>[-−])?"
            + f"(?:{_ORDINAL_TOKEN[lang]}"
            + f"|{_SKIP_TOKEN}"
            + f"|{_NUM_TOKEN[lang]})"
            + rf"(?:\s*(?P<unit>{units})\.?)?(?!\w)",
            re.IGNORECASE,
        )
    return _PATTERNS[lang]


@lru_cache(maxsize=8192)
def _n2w(
    value: int | float,
    lang: str,
    to: str = "",
    case: str = "",
    gender: str = "m",
    plural: bool = False,
) -> str:
    kwargs: dict = {}
    if to:
        kwargs["to"] = to
    if lang == "ru":  # английский конвертер не знает падежей и родов
        if case:
            kwargs["case"] = case
        if gender != "m":
            kwargs["gender"] = gender
        if plural:
            kwargs["plural"] = True
    return num2words(value, lang=lang, **kwargs)


def _plain(token: str) -> str:
    return re.sub(r"\D", "", token)


def _parse(token: str, kind: str) -> tuple[int, str | None]:
    if kind == "grp":  # разделители здесь только тысячными
        return int(_plain(token)), None
    head, _, tail = token.replace(",", ".").partition(".")
    if not tail or set(tail) <= {"0"}:
        return int(_plain(head)), None
    return int(_plain(head)), tail


def _read(token: str, kind: str, lang: str, gender: str) -> tuple[str, int, bool]:
    """(слова, целая часть, целое ли число)"""
    value, frac = _parse(token, kind)
    if kind == "int" and (len(token) > 10 or (len(token) > 1 and token[0] == "0")):
        # телефон, карта, «007»: по цифрам, а не как миллиарды
        return " ".join(_DIGITS[lang][int(d)] for d in token), value, True
    if frac is None:
        return _n2w(value, lang, gender=gender), value, True
    return _n2w(float(f"{value}.{frac}"), lang), value, False


def _ordinal(token: str, lang: str, suffix: str) -> str | None:
    value = int(_plain(token))
    if lang == "en":
        return _n2w(value, lang, to="ordinal")
    rule = _ORDINGS.get(suffix)
    if rule is None:
        return None
    kind, case, gender, plural = rule
    if value % 10 == 0 and suffix in ("е", "ее", "ое"):
        gender, plural = "m", True  # «30-е годы» → «тридцатые», а не «тридцатое»
    if kind == "c":
        if value % 10 == 0:  # «в 90-х годах» → «девяностых», а не «девяноста»
            return _n2w(value, lang, to="ordinal", case="g", plural=True)
        return _n2w(value, lang, case=case, plural=(value % 10 == 1))
    return _n2w(
        value, lang, to="ordinal", case=case, gender=gender, plural=plural
    )


def _unit(raw: str, lang: str) -> tuple[tuple[str, str, str], str] | None:
    return _UNITS[lang].get(raw.lower().strip().rstrip("."))


def _form(forms: tuple[str, str, str], lang: str, value: int, whole: bool) -> str:
    if lang == "en":
        return forms[0] if value == 1 else forms[2]
    if not whole:
        # «пять десятых рубля» — родительский единственного
        return forms[1]
    if 11 <= value % 100 <= 20:
        return forms[2]
    return forms[_RU_CLASS[value % 10]]


def _render(match: re.Match, lang: str) -> str:
    g = match.groupdict()
    if g.get("skip"):
        return match.group(0)

    head = g.get("pre") or ""
    tail = g.get("unit") or ""
    unit = _unit(tail or head, lang)
    neg = _NEG[lang] if g.get("minus") else ""

    token = g.get("ord")
    if token is not None:
        words = _ordinal(token, lang, (g.get("suf") or "").lower())
        if words is None:
            return match.group(0)
        value, whole = int(_plain(token)), True
    else:
        kind = next(k for k in ("grp", "dec", "int") if g[k])
        words, value, whole = _read(
            g[kind], kind, lang, unit[1] if unit else "m"
        )

    if unit is None:
        return f"{neg}{head}{words}{tail}"
    return f"{neg}{words} {_form(unit[0], lang, value, whole)}"


def spell_numbers(text: str, lang: str) -> str:
    """Цифры → слова. Для японского и неизвестных языков текст как есть."""
    if lang not in LANGS or not _HAS_DIGIT.search(text):
        return text
    out = _pattern(lang).sub(lambda m: _render(m, lang), text)
    return _SPACES.sub(" ", out).strip()
