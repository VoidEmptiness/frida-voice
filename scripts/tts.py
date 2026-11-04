
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
frida-voice — локальный TTS голосом Фриды (ru / ja / en).

  текст → базовый TTS (Silero/Piper) → RVC Silence Suzuka → WAV
Без внешних API.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from punctuate import Punctuator, needs_punctuation

ROOT = Path(__file__).resolve().parent.parent

LANG_PITCH_DEFAULT = {"ru": 4, "en": 4, "ja": 2}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def resolve_device(cfg_device: str) -> str:
    if cfg_device.startswith("cuda") and torch.cuda.is_available():
        return cfg_device
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA недоступна, переключаюсь на CPU")
    return "cpu"


def detect_lang(text: str) -> str:
    """Грубая эвристика: японские знаки → ja, кириллица → ru, иначе en."""
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]", text):
        return "ja"
    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"
    return "en"


def slug_from_text(text: str, max_words: int = 3, max_len: int = 48) -> str:
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    if not tokens:
        return "untitled"
    slug = "_".join(tokens[:max_words]).lower()
    return slug[:max_len].rstrip("_")


def find_rvc_files(model_dir: Path) -> tuple[Path, Path | None]:
    pths = sorted(model_dir.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(
            f"Нет .pth в {model_dir}. Сначала:\n  docker compose run --rm download"
        )
    idxs = sorted(model_dir.glob("*.index"))
    return pths[0], idxs[0] if idxs else None


def ensure_rvc_base_weights(models_dir: Path) -> None:
    import shutil

    import rvc_python

    pkg_base = Path(rvc_python.__file__).resolve().parent / "base_model"
    pkg_base.mkdir(parents=True, exist_ok=True)
    cache = models_dir / "rvc_base"
    for name in ("hubert_base.pt", "rmvpe.pt", "rmvpe.onnx"):
        src, dst = cache / name, pkg_base / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        if src.exists() and src.stat().st_size > 0:
            print(f"[cache] {name} ← models/rvc_base")
            shutil.copy2(src, dst)


class SileroEngine:
    """Silero для русского (v4_ru) и английского (v3_en)."""

    SPECS = {
        "ru": {"language": "ru", "model_id": "v4_ru"},
        "en": {"language": "en", "model_id": "v3_en"},
    }

    def __init__(self, device: str):
        self.device = torch.device("cpu" if device == "cpu" else device)
        self._models: dict[str, torch.nn.Module] = {}

    def _load(self, lang: str) -> torch.nn.Module:
        if lang in self._models:
            return self._models[lang]
        spec = self.SPECS[lang]
        print(f"[silero] загрузка {spec['model_id']} …")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=spec["language"],
            speaker=spec["model_id"],
            trust_repo=True,
        )
        model.to(self.device)
        self._models[lang] = model
        return model

    def synthesize(
        self, text: str, lang: str, speaker: str, sample_rate: int
    ) -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            raise ValueError("Пустой текст")
        if text[-1] not in ".!?…。！？":
            text += "。" if lang == "ja" else "."
        model = self._load(lang)
        with torch.inference_mode():
            audio = model.apply_tts(
                text=text, speaker=speaker, sample_rate=sample_rate
            )
        return audio.detach().cpu().numpy().astype(np.float32), sample_rate


class PiperEngine:
    """Piper ONNX — японский (и при желании другие piper-голоса)."""

    def __init__(self):
        self._voices: dict[str, object] = {}

    def _load(self, model_path: Path):
        key = str(model_path)
        if key in self._voices:
            return self._voices[key]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Нет Piper-модели: {model_path}\n"
                f"Скачай: docker compose run --rm download"
            )
        from piper import PiperVoice

        print(f"[piper] загрузка {model_path.name} …")
        voice = PiperVoice.load(str(model_path))
        self._voices[key] = voice
        return voice

    def synthesize(
        self, text: str, model_path: Path, speaker_id: int = 0
    ) -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            raise ValueError("Пустой текст")
        voice = self._load(model_path)
        tmp_path = Path(
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        )
        try:
            kwargs: dict = {}
            try:
                from piper.config import SynthesisConfig

                kwargs["syn_config"] = SynthesisConfig(speaker_id=speaker_id)
            except Exception:
                kwargs["speaker_id"] = speaker_id

            with wave.open(str(tmp_path), "wb") as wav_file:
                if hasattr(voice, "synthesize_wav"):
                    try:
                        voice.synthesize_wav(text, wav_file, **kwargs)
                    except TypeError:
                        voice.synthesize_wav(text, wav_file)
                else:
                    try:
                        voice.synthesize(text, wav_file, **kwargs)
                    except TypeError:
                        voice.synthesize(text, wav_file)

            if tmp_path.stat().st_size < 44:
                raise RuntimeError("Piper не записал аудио (пустой WAV)")
            audio, sr = sf.read(tmp_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), int(sr)
        finally:
            tmp_path.unlink(missing_ok=True)


class SuzukaVoice:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = resolve_device(str(cfg.get("device", "cpu")))
        self.default_lang = str(cfg.get("lang", "auto")).lower()
        self.global_pitch = int(cfg.get("pitch", 4))
        self.f0method = cfg.get("f0method", "harvest")
        self.index_rate = float(cfg.get("index_rate", 0.6))
        self.protect = float(cfg.get("protect", 0.33))
        self.lang_cfg: dict = cfg.get("languages") or {}
        self.punct_mode = str(cfg.get("punctuate", "auto")).lower()

        self.models_dir = resolve_path(cfg.get("models_dir", "models"))
        self.rvc_dir = resolve_path(cfg.get("rvc_model", "models/silence_suzuka"))
        self.output_dir = resolve_path(cfg.get("output_dir", "output"))
        self.tmp_dir = resolve_path(cfg.get("tmp_dir", "tmp"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.force_lang: str | None = (
            None if self.default_lang == "auto" else self.default_lang
        )
        self.pitch = self.global_pitch

        self.silero = SileroEngine(self.device)
        self.piper = PiperEngine()
        self.punctuator = Punctuator()
        self._init_rvc()

    def _init_rvc(self) -> None:
        from rvc_python.infer import RVCInference

        ensure_rvc_base_weights(self.models_dir)
        pth, index = find_rvc_files(self.rvc_dir)
        print(f"[rvc] модель: {pth.name}" + (f" + {index.name}" if index else ""))

        rvc_device = "cpu" if self.device == "cpu" else self.device
        self.rvc = RVCInference(
            models_dir=str(self.models_dir),
            device=rvc_device,
            model_path=str(pth),
            index_path=str(index) if index else "",
            version="v2",
        )
        self._apply_rvc_params()

    def _apply_rvc_params(self) -> None:
        self.rvc.set_params(
            f0method=self.f0method,
            f0up_key=self.pitch,
            index_rate=self.index_rate,
            protect=self.protect,
            filter_radius=3,
            rms_mix_rate=0.25,
            resample_sr=0,
        )

    def _lang_settings(self, lang: str) -> dict:
        return dict(self.lang_cfg.get(lang) or {})

    def _pitch_for(self, lang: str) -> int:
        settings = self._lang_settings(lang)
        if "pitch" in settings:
            return int(settings["pitch"])
        return int(LANG_PITCH_DEFAULT.get(lang, self.global_pitch))

    def synthesize_base(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        settings = self._lang_settings(lang)
        engine = settings.get("engine")
        if lang == "ru" or (engine == "silero" and lang == "ru"):
            return self.silero.synthesize(
                text,
                "ru",
                settings.get("speaker", "xenia"),
                int(settings.get("sample_rate", 48000)),
            )
        if lang == "en" or (engine == "silero" and lang == "en"):
            return self.silero.synthesize(
                text,
                "en",
                settings.get("speaker", "en_0"),
                int(settings.get("sample_rate", 48000)),
            )
        if lang == "ja" or engine == "piper":
            model = resolve_path(
                settings.get(
                    "model", "models/piper/ja_JP-hi_fi_captain-medium.onnx"
                )
            )
            speaker_id = int(settings.get("speaker_id", 0))
            return self.piper.synthesize(text, model, speaker_id=speaker_id)
        raise ValueError(f"Язык не поддерживается: {lang}")

    def speak(
        self,
        text: str,
        out_path: Path | None = None,
        lang: str | None = None,
        keep_raw: bool = False,
    ) -> Path:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("Пустой текст")

        resolved = lang or self.force_lang or detect_lang(text)
        if resolved not in {"ru", "ja", "en"}:
            raise ValueError(f"Неизвестный язык: {resolved}")

        if resolved in {"ru", "en"} and (
            self.punct_mode == "on"
            or (self.punct_mode == "auto" and needs_punctuation(text))
        ):
            fixed = self.punctuator.punctuate(text, resolved)
            if fixed != text:
                print(f"[punct] {fixed[:80]}{'…' if len(fixed) > 80 else ''}")
                text = fixed

        pitch = self._pitch_for(resolved)
        if pitch != self.pitch:
            self.pitch = pitch
            self._apply_rvc_params()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_path is None:
            slug = slug_from_text(text)
            out_path = self.output_dir / f"frida_{resolved}_{slug}.wav"
            if out_path.exists():
                out_path = (
                    self.output_dir / f"frida_{resolved}_{slug}_{stamp}.wav"
                )
        else:
            out_path = resolve_path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[lang] {resolved}  pitch={self.pitch}")
        print(f"[tts] «{text[:80]}{'…' if len(text) > 80 else ''}»")
        wav, sr = self.synthesize_base(text, resolved)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self.tmp_dir, delete=False
        ) as raw_f:
            raw_path = Path(raw_f.name)
        sf.write(raw_path, wav, sr)

        print("[rvc] → голос Фриды (Suzuka) …")
        self.rvc.infer_file(str(raw_path), str(out_path))

        if not keep_raw:
            raw_path.unlink(missing_ok=True)
        else:
            kept = self.tmp_dir / f"raw_{resolved}_{stamp}.wav"
            raw_path.rename(kept)
            print(f"[raw] {kept}")

        print(f"[ok] {out_path}")
        return out_path


def interactive(engine: SuzukaVoice) -> None:
    print()
    print("frida-voice — голос Фриды (ru / ja / en)")
    print("Команды: /lang auto|ru|ja|en  /pitch N  /help  /quit")
    print()
    while True:
        try:
            line = input("frida> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/q", "/quit", "/exit"}:
            break
        if line.startswith("/lang"):
            parts = line.split()
            if len(parts) == 2 and parts[1] in {"auto", "ru", "ja", "en"}:
                engine.force_lang = None if parts[1] == "auto" else parts[1]
                print(f"lang = {parts[1]}")
            else:
                print("Использование: /lang auto|ru|ja|en")
            continue
        if line.startswith("/pitch"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                engine.pitch = int(parts[1])
                engine._apply_rvc_params()
                lang = engine.force_lang or "ru"
                engine.lang_cfg.setdefault(lang, {})["pitch"] = engine.pitch
                print(f"pitch = {engine.pitch}")
            else:
                print("Использование: /pitch 4")
            continue
        if line in {"/help", "/?"}:
            print("Язык определяется автоматически (кана/иероглифы→ja, кириллица→ru).")
            print("/lang X   — auto|ru|ja|en")
            print("/pitch N  — сдвиг тона RVC")
            print("/quit")
            continue
        try:
            engine.speak(line)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="frida-voice: local TTS голосом Фриды (ru/ja/en, Silero+Piper+RVC)"
    )
    p.add_argument("-t", "--text", help="Текст для озвучки")
    p.add_argument("-o", "--out", help="Путь к выходному WAV")
    p.add_argument("-l", "--lang", choices=["auto", "ru", "ja", "en"], help="Язык")
    p.add_argument("-c", "--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--pitch", type=int)
    p.add_argument("--device", help="cpu | cuda:0")
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument(
        "--punct",
        choices=["auto", "on", "off"],
        help="Постановка знаков препинания (моделью) перед озвучкой",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Нет конфига: {cfg_path}", file=sys.stderr)
        return 1

    cfg = load_config(cfg_path)
    if args.device:
        cfg["device"] = args.device
    if args.pitch is not None:
        cfg["pitch"] = args.pitch
        for code in ("ru", "en", "ja"):
            cfg.setdefault("languages", {}).setdefault(code, {})["pitch"] = args.pitch
    if args.lang:
        cfg["lang"] = args.lang
    if args.punct:
        cfg["punctuate"] = args.punct

    print(
        f"device={cfg.get('device')} lang={cfg.get('lang', 'auto')} "
        f"pitch={cfg.get('pitch')}"
    )
    engine = SuzukaVoice(cfg)

    if args.text:
        lang = None if (args.lang in (None, "auto")) else args.lang
        engine.speak(
            args.text,
            Path(args.out) if args.out else None,
            lang=lang,
            keep_raw=args.keep_raw,
        )
        return 0

    interactive(engine)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nВыход.")
        raise SystemExit(130)
