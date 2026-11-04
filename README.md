# frida-voice — голос Фриды

**Это проект/модуль семейства frida(ссылка на проект которого нету:)**

Локальный text-to-speech голосом **Фриды**. Без внешних API.

> Голос Фриды построен на RVC-модели Silence Suzuka (Uma Musume Pretty Derby,
> [источник модели](https://huggingface.co/maryishi/IshiMary_RVC_Models)):
> базовый TTS озвучивает текст «нейтральным» голосом, RVC перекрашивает его в тембр Сузуки —
> так Фрида говорит по-русски, по-английски и по-японски одним голосом.

```
текст (ru / ja / en) → базовый TTS → RVC (Silence Suzuka → Фрида) → output/*.wav
```

| Язык | Базовый TTS |
|------|-------------|
| Русский | Silero `v4_ru` |
| Английский | Silero `v3_en` |
| Японский | Piper `ja_JP-hi_fi_captain-medium` (female) |

Язык определяется автоматически (кана/иероглифы → ja, кириллица → ru, иначе en).

## Живая интонация: авто-пунктуация

Текст почти без знаков препинания озвучивается монотонно и «спотыкается». Перед TTS
знаки расставляет отдельная модель (`scripts/punctuate.py`):

| Язык | Модель |
|------|--------|
| Русский | `markusiko/rubert-base-punctuation` |
| Английский | `felflare/bert-restore-punctuation` |
| Японский | без модели (обычно знаки уже есть в тексте) |

Режим задаётся в `config.yaml` → `punctuate: auto | on | off` (auto — только если
знаков мало) или флагом `--punct`:

```bash
docker compose run --rm voice --text "привет как дела рад тебя слышать"
# → [punct] Привет, как дела? Рад тебя слышать.
```

Кэш HF-моделей лежит в `models/hf/` (том с хоста, скачивается один раз).

## Быстрый старт

```bash
docker compose build
docker compose run --rm download          # RVC (Suzuka) + hubert + Piper JA + HF-модели
docker compose run --rm voice             # интерактив
```

Примеры:

```bash
# русский
docker compose run --rm voice --text "Привет, я Фрида." -o output/ru.wav

# японский
docker compose run --rm voice --text "こんにちは、フリーダです。" -o output/ja.wav

# английский
docker compose run --rm voice --text "Hello, I am Frida." -o output/en.wav

# явный язык
docker compose run --rm voice --lang ja --text "がんばります！" -o output/ja2.wav
```

## Интерактив

| Команда | Действие |
|---------|----------|
| текст | озвучить (авто-язык) |
| `/lang auto\|ru\|ja\|en` | зафиксировать язык |
| `/pitch 4` | сдвиг тона RVC |
| `/quit` | выход |

## Настройки

`config.yaml` → секция `languages`: спикер Silero, pitch по языкам, путь к Piper-модели.
