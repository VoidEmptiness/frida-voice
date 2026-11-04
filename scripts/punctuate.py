
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Постановка знаков препинания перед озвучкой (ru / en)."""

from __future__ import annotations

import re

RU_MODEL = "markusiko/rubert-base-punctuation"
EN_MODEL = "felflare/bert-restore-punctuation"

EN_LABELS = {0: "", 1: ",", 2: "?", 3: ".", 4: "!"}
RU_LABELS = {
    "B-!": "!",
    "B-,": ",",
    "B-.": ".",
    "B-...": "...",
    "B-:": ":",
    "B-?": "?",
    "O": "",
}

_SENT_END = {".", "!", "?", "..."}
_PUNCT_RE = re.compile(r"[.!?:;,…、。！？]")
_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)

_DENSITY_SKIP = 1.5


def needs_punctuation(text: str) -> bool:
    """auto-режим: пропустить модель, если знаки уже расставлены."""
    text = text.strip()
    if len(text) < 12:
        return False
    marks = len(_PUNCT_RE.findall(text))
    if marks >= 2 and marks / max(len(text), 1) * 100 >= _DENSITY_SKIP:
        return False
    if _PUNCT_RE.search(text[-1]):
        return False
    return True


class Punctuator:
    def __init__(self) -> None:
        self._models: dict[str, tuple] = {}
        self._failed: set[str] = set()

    def punctuate(self, text: str, lang: str) -> str:
        if lang not in ("ru", "en"):
            return text
        if lang in self._failed:
            return text
        try:
            pair = self._models.get(lang)
            if pair is None:
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                )

                name = RU_MODEL if lang == "ru" else EN_MODEL
                print(f"[punct] загрузка {name} …")
                tok = AutoTokenizer.from_pretrained(name)
                model = AutoModelForTokenClassification.from_pretrained(name)
                model.eval()
                pair = self._models[lang] = (tok, model)
            return _restore(text, pair[0], pair[1], lang)
        except Exception as e:
            self._failed.add(lang)
            print(f"[punct] {lang} недоступен ({e}), оставляю без знаков")
            return text


def _ru_mark(id2label: dict, label_id: int) -> str:
    label = id2label[label_id]
    if label == "O":
        return ""
    if label.startswith("B-"):
        return RU_LABELS.get(label, "")
    return ""


def _en_mark(label_id: int) -> str:
    return EN_LABELS.get(label_id, "")


def _restore(text: str, tok, model, lang: str, chunk_words: int = 90) -> str:
    import torch

    words = _WORD_RE.findall(text)
    if not words:
        return text

    id2label = model.config.id2label
    out: list[str] = []
    capitalize_next = lang == "ru"

    for i in range(0, len(words), chunk_words):
        chunk = words[i : i + chunk_words]
        enc = tok(
            chunk,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1)[0].tolist()

        word_ids = enc.word_ids(batch_index=0)
        first_tok: dict[int, int] = {}
        for tid, wid in enumerate(word_ids):
            if wid is not None and wid not in first_tok:
                first_tok[wid] = tid

        for wid, word in enumerate(chunk):
            tid = first_tok.get(wid)
            if tid is None:
                mark = ""
            elif lang == "ru":
                mark = _ru_mark(id2label, preds[tid])
            else:
                mark = _en_mark(preds[tid])

            if capitalize_next and word[:1].islower():
                word = word.capitalize()
            capitalize_next = mark in _SENT_END
            out.append(word + mark)

    return " ".join(out)
