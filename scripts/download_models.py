
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Скачивает RVC-модель Silence Suzuka и базовые веса (hubert / rmvpe)."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SUZUKA_DIR = MODELS / "silence_suzuka"

SUZUKA_ZIP_URL = (
    "https://huggingface.co/maryishi/IshiMary_RVC_Models/resolve/main/"
    "SlienceSuzuka_RMVPE_600Epochs.zip"
)

SUZUKA_ZIP_FALLBACK = (
    "https://huggingface.co/sxndypz/rvc-v2-models/resolve/main/silencesuzuka.zip"
)

RVC_BASE = {
    "hubert_base.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/hubert_base.pt",
    "rmvpe.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.pt",
    "rmvpe.onnx": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.onnx",
}

PIPER_DIR = MODELS / "piper"
PIPER_JA = {
    "ja_JP-hi_fi_captain-medium.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "ja/ja_JP/hi_fi_captain/medium/ja_JP-hi_fi_captain-medium.onnx"
    ),
    "ja_JP-hi_fi_captain-medium.onnx.json": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "ja/ja_JP/hi_fi_captain/medium/ja_JP-hi_fi_captain-medium.onnx.json"
    ),
}


def download(url: str, dest: Path, label: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} уже есть")
        return

    print(f"[download] {label or dest.name}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def flatten_model_dir(model_dir: Path) -> None:
    """Гарантирует, что .pth и .index лежат прямо в model_dir."""
    pths = list(model_dir.rglob("*.pth"))
    idxs = list(model_dir.rglob("*.index"))
    if not pths:
        raise RuntimeError(f"В архиве нет .pth: {model_dir}")

    for src in pths + idxs:
        if src.parent == model_dir:
            continue
        target = model_dir / src.name
        if not target.exists():
            src.rename(target)

    for p in sorted(model_dir.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def download_suzuka() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS / "silence_suzuka.zip"

    existing = list(SUZUKA_DIR.glob("*.pth")) if SUZUKA_DIR.exists() else []
    if existing:
        print(f"[skip] Silence Suzuka уже установлена: {existing[0].name}")
        return

    urls = [SUZUKA_ZIP_URL, SUZUKA_ZIP_FALLBACK]
    last_err: Exception | None = None
    for url in urls:
        try:
            download(url, zip_path, label="Silence Suzuka RVC")
            if SUZUKA_DIR.exists():
                for child in SUZUKA_DIR.iterdir():
                    if child.is_file():
                        child.unlink()
            extract_zip(zip_path, SUZUKA_DIR)
            flatten_model_dir(SUZUKA_DIR)
            zip_path.unlink(missing_ok=True)
            print(f"[ok] Модель в {SUZUKA_DIR}")
            for f in sorted(SUZUKA_DIR.iterdir()):
                print(f"     - {f.name}")
            return
        except Exception as e:
            last_err = e
            print(f"[warn] Не удалось с {url}: {e}")
            zip_path.unlink(missing_ok=True)

    raise RuntimeError(f"Не удалось скачать модель Silence Suzuka: {last_err}")


def download_rvc_base_into_package() -> None:
    """Скачивает hubert/rmvpe в models/rvc_base и дублирует в пакет rvc-python."""
    import shutil

    copy_dir = MODELS / "rvc_base"
    copy_dir.mkdir(parents=True, exist_ok=True)

    try:
        import rvc_python

        pkg_base = Path(rvc_python.__file__).resolve().parent / "base_model"
        pkg_base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pkg_base = None

    for name, url in RVC_BASE.items():
        persistent = copy_dir / name
        download(url, persistent)
        if pkg_base is not None:
            target = pkg_base / name
            if not target.exists() or target.stat().st_size != persistent.stat().st_size:
                shutil.copy2(persistent, target)


def download_piper_ja() -> None:
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in PIPER_JA.items():
        download(url, PIPER_DIR / name, label=f"Piper JA {name}")
    print(f"[ok] Piper Japanese → {PIPER_DIR}")


def download_punctuation_models() -> None:
    """HF-модели для постановки знаков препинания (ru / en)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    for name in ("markusiko/rubert-base-punctuation", "felflare/bert-restore-punctuation"):
        print(f"[hf] {name}")
        AutoTokenizer.from_pretrained(name)
        AutoModelForTokenClassification.from_pretrained(name)
    print("[ok] Punctuation models → HF cache")


def main() -> int:
    print("=== frida-voice: загрузка моделей ===")
    download_rvc_base_into_package()
    download_suzuka()
    download_piper_ja()
    try:
        download_punctuation_models()
    except Exception as e:
        print(f"[warn] знаков препинания не скачаны: {e}", file=sys.stderr)
    print("Готово.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nОтменено.", file=sys.stderr)
        raise SystemExit(130)
