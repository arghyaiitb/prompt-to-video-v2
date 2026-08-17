# LANGUAGES — what it actually takes to ship Spanish and Hindi

Status: **empirical**. Every row below is backed by a command run on this box on 2026-08-17
against the live keys in `.env`, or by a doc URL. Anything not verified is labelled
`UNVERIFIED`. Proof scripts and artefacts:
`/private/tmp/claude-501/-Users-argo-ab-prompt-to-video-v2/17f5789b-d93a-4c4f-af36-254d779b6e1c/scratchpad/lang/`
(`shapetest.py`, `tokenizer_fix.py`, `e2e_anchor2.py`, `roundtrip.py`, `wer.py`,
`typography.py`, `legibility.py`, `budget.py`, `scriptgen.py`, `slide.py`).

---

## 0. Verdicts

| language | script gen | TTS | alignment | rendering | typography | **verdict** |
|---|---|---|---|---|---|---|
| **English** | ✅ ships | ✅ 53 Aura voices | ✅ nova-3 | ✅ ImageMagick | ✅ baseline | **GO** (baseline) |
| **Spanish** | ✅ proven | ✅ 17 Aura + 10 Polly | ⚠️ needs `language=es` | ✅ works today | ✅ unchanged | **GO-WITH-CAVEATS** |
| **Hindi** | ✅ proven | ⚠️ Polly only, **1 usable voice** | ⚠️ needs `language=hi` + Unicode tokenizer | ❌ **renders blank today** | ❌ needs 1.50em line box + new font | **GO-WITH-CAVEATS** |

Neither language is BLOCKED. I proved every stage works end to end for both. The cost is
concentrated in two places: **a shaping-capable text renderer** (Hindi only) and **a
Unicode-correct tokenizer** (both languages).

### The single biggest risk

**Hindi renders as a completely blank text panel today, and nothing in the pipeline
detects it.** `find_font()` returns `Arial Bold.ttf`, which has no Devanagari glyphs at
all; ImageMagick exits 0, writes a valid PNG, and the layout code happily measures a width
from the missing glyphs. A Hindi job would traverse script gen, TTS, alignment, render and
scoring without a single error and produce a video with correctly-timed, correctly-placed,
**empty** text. That is worse than a crash. See §3.1 and §7.

---

## 1. Corrections to the prior measurements

| prior claim | verdict | evidence |
|---|---|---|
| Deepgram has 17 Spanish voices | ✅ **confirmed** | `GET /v1/models` → 102 TTS, 17 with `es` |
| Deepgram has zero Hindi voices | ✅ **confirmed** | 0 of 102 list any `hi*` code |
| `describe-voices --language-code hi-IN` returns nothing | ✅ **confirmed** | empty without `IncludeAdditionalLanguageCodes` |
| Hindi needs an explicit `LanguageCode` on the Polly request | ❌ **WRONG** | Kajal and Aditi return **byte-identical wav** (sha256 match) with and without `LanguageCode=hi-IN`. Polly infers Hindi from the Devanagari script. |
| `pango:` failed on syntax, not capability | ❌ **WRONG** | Capability. `magick -list configure` DELEGATES has no `pango`; `-list format` shows `PANGO ---` (no read/write). `pango:` → `no decode delegate`. Also `rsvg-convert` is **not installed**, so the SVG delegate is dead too. |
| Devanagari rendering is broken; nukta and anusvara dropped | ✅ **confirmed, and worse** | Confirmed for Sangam MN, *and* every font in `FONT_CANDIDATES` lacks Devanagari entirely → blank, not mis-shaped |
| Polly Spanish = Lupe / Penelope / Miguel | ⚠️ **incomplete** | also Pedro (es-US generative), Lucia + Sergio (es-ES generative), Alba + Raul (es-ES long-form), Mia + Andres (es-MX generative) |
| Spanish runs 20–25% longer than English | ⚠️ **measured 12%** | words for the same content: es **1.12×**, hi **1.28×** (§6) |

---

## 2. TTS

### 2.1 Availability (`GET https://api.deepgram.com/v1/models`, live key)

| provider | Spanish | Hindi |
|---|---|---|
| Deepgram Aura-2 | **17** voices: es-ES 6, es-MX 6, es-419 2, es-CO 2, es-AR 1 | **0** |
| Amazon Polly | 10 voices across es-ES / es-US / es-MX | **2**, both `en-IN` primary with `hi-IN` additional |

### 2.2 The Hindi voice situation — the product constraint

```
polly.describe_voices(LanguageCode="hi-IN", IncludeAdditionalLanguageCodes=True)
```

| voice | primary | additional | engines | gender | Hindi verdict |
|---|---|---|---|---|---|
| `Kajal` | en-IN | hi-IN | generative, neural | Female | **use this, on `neural`** |
| `Aditi` | en-IN | hi-IN | standard | Female | fallback only — too slow (§6.3) |

Measured facts:

- `Kajal` + `Engine=standard` → `ValidationException`. Kajal is not a standard voice.
- `Kajal` + `Engine=generative` works, but is **worse for Hindi than neural**: round-trip
  fidelity 75.8% vs **84.8%**, and 147.8 wpm vs 207.6 wpm. The project default
  `video_polly_engine=generative` is the wrong tier here.
- `LanguageCode=hi-IN` is a **no-op** for Devanagari input (byte-identical audio).

**There is exactly one shippable Hindi voice, and it is female with an Indian-English
voice model.** No male Hindi voice exists on either provider. That is a product decision,
not an engineering one: every Hindi video will sound like the same narrator.

### 2.3 Recommended production voices

| language | provider | voice | engine | measured |
|---|---|---|---|---|
| en | deepgram | `aura-2-draco-en` | — | baseline |
| es | deepgram | `aura-2-celeste-es` (es-CO, neutral LatAm) | — | 90.9% round-trip |
| hi | **polly** | `Kajal` | **`neural`** | 84.8% round-trip |

Hindi forces the Polly engine, which means Hindi jobs get SSML support (`supports_ssml =
True`) while Spanish jobs on Deepgram do not. That asymmetry already exists in
`worker/factory.py`; it just becomes language-determined rather than user-chosen.

---

## 3. Rendering — the highest-risk item

### 3.1 Why it is broken: no shaping engine exists on this box

| capability | state | proof |
|---|---|---|
| ffmpeg `drawtext` / `ass` / `subtitles` | **absent** | `ffmpeg -filters` lists 489 filters, none of the three |
| ImageMagick `raqm` (complex-text shaping) | **absent** | `magick -list configure` → `DELEGATES ... fontconfig freetype ...` no raqm |
| ImageMagick `pango:` | **absent** | `-list format` → `PANGO ---`; `pango:file` → `no decode delegate` |
| ImageMagick SVG via `rsvg-convert` | **absent** | binary not installed; falls back to internal MSVG (freetype, no shaping) |
| Homebrew `pango` / `harfbuzz` / `fribidi` | present | `pango-view 1.57.1`, `hb-shape 14.2.0` |
| Pillow with Raqm | present | system py3.13 **and** a fresh `pip install pillow` on py3.14 → `raqm 0.10.5` |

ImageMagick is the pipeline's only text renderer (`text_overlay.py` docstring says so, and
the ffmpeg check above confirms there is no alternative). It cannot shape Devanagari.

### 3.2 The shaping matrix — proven with harfbuzz as ground truth

Method (`shapetest.py`): `hb-shape` gives the correct glyph run and advance sum with the
Deva shaper, and the wrong one with `--shapers=fallback` (cmap-only). Render each
candidate, measure real ink width, see which prediction it matches.

Font: `Devanagari Sangam MN.ttc`, pointsize 64.

| candidate | `सुरक्षित रहें` | `कर्मचारी प्रशिक्षण` | verdict |
|---|---|---|---|
| harfbuzz **shaped** (truth) | 242.2 px | 386.2 px | — |
| harfbuzz **unshaped** (cmap only) | 276.8 px | 478.0 px | — |
| `magick -annotate` (current path) | 278 px | 479 px | ❌ **UNSHAPED** |
| `magick label:` (current path) | 278 px | 479 px | ❌ **UNSHAPED** |
| `magick caption:` | 278 px | 479 px | ❌ **UNSHAPED** |
| `magick pango:` | — | — | ❌ **no delegate** |
| `pango-view` | 244 px | 388 px | ✅ **SHAPED** (±2 px) |
| **Pillow + Raqm** | 243 px | 387 px | ✅ **SHAPED** (±1 px) |

ImageMagick overstates the width of a conjunct-heavy string by **up to 24%** (479 vs 386).
Our whole layout — wrap points, shrink steps, char caps, the fixed 88 px rule, the
`AVG_GLYPH_RATIO` char estimates — is driven by measured width. A 24% measurement error
does not merely look wrong; it wraps and shrinks against fiction.

### 3.3 Per-character survival (visual inspection of the rendered PNGs)

`DEVANAGARI_COMPARISON.png`, `FONT_COVERAGE.png` — rendered and viewed.

| feature | `magick` + Arial (today) | `magick` + Sangam MN | Pillow+Raqm / pango |
|---|---|---|---|
| base consonants | ❌ nothing drawn | ✅ | ✅ |
| nukta `U+093C` (फ़) | ❌ | ❌ **dropped** | ✅ |
| anusvara `U+0902` (ं) | ❌ | ❌ misplaced | ✅ |
| i-matra reorder (ि before its consonant) | ❌ | ❌ **not reordered** | ✅ |
| क्ष conjunct ligature | ❌ | ❌ shown as क् + ष | ✅ |
| प्र rakar ligature | ❌ | ❌ | ✅ |
| र् reph (superscript hook) | ❌ | ❌ explicit halant | ✅ |

`फ़िशिंग हमले को पहचानें` renders as **nothing** today, and as `फशिागि हमले को पहचानें`
with a Devanagari font — a different, meaningless word.

### 3.4 Font coverage — the blank-render root cause

`fontTools` cmap check:

| font | Devanagari `क` | nukta | Spanish `ó ñ ¿ ü` | GSUB |
|---|---|---|---|---|
| `Arial Bold.ttf` — `FONT_CANDIDATES[0]` | ❌ | ❌ | ✅ | ✅ |
| `SFNS.ttf` — `[1]` | ❌ | ❌ | ✅ | ✅ |
| `Arial.ttf` — `[2]` | ❌ | ❌ | ✅ | ✅ |
| `Devanagari Sangam MN.ttc` | ✅ | ✅ | ❌ | ✅ |
| `DevanagariMT.ttc` | ✅ | ✅ | ❌ | ❌ **no GSUB — can never shape** |
| `ITFDevanagari.ttc` | ✅ | ✅ | ❌ | ✅ |
| **`Kohinoor.ttc` index 3 (Bold)** | ✅ | ✅ | **✅** | ✅ |

**`Kohinoor.ttc` index 3 is the only face on this box that covers Devanagari *and* Latin
with accents, so it is one font for all three languages.** It also has the best size parity
(§5.2). Rule out `DevanagariMT` permanently.

### 3.5 The winning command

**Recommended — Pillow + Raqm, in-process.** `pip install pillow` bundles
`raqm 0.10.5 / harfbuzz 11.2.1 / fribidi 1.0.16` in the wheel; verified on the backend's own
Python 3.14.4 with Pillow 12.3.0. No system packages, no subprocess, and it returns exact
metrics from the same code path that draws — which ImageMagick's separate measure/draw
invocations never guaranteed.

```python
from PIL import Image, ImageDraw, ImageFont

FONT, IDX = "/System/Library/Fonts/Kohinoor.ttc", 3   # Kohinoor Devanagari Bold
font = ImageFont.truetype(FONT, 44, index=IDX)

# MEASURE — matches harfbuzz to ±1 px
width = font.getlength(text, language="hi")

# DRAW — anchor on the BASELINE ('ls'), never the ascender (see §5.3)
draw.text((x, baseline_y), text, font=font, fill=colour,
          language="hi", anchor="ls")
```

**Fallback — `pango-view` subprocess**, if a Pillow dependency is unacceptable:

```bash
pango-view -q --font "Kohinoor Devanagari Bold 44px" --width 860 \
  --background '#0d1b2a' --foreground '#f2f5f7' --margin 4 \
  -o out.png text.txt
```

Caveats on the fallback: `--width` is in points, not pixels, so it does not compose with a
px font spec; `--margin 0` clips the anusvara; and it gives no width query, so measurement
would still need a second tool. Prefer Pillow.

### 3.6 Line wrapping and extents — verified

- **Wrapping works.** `WRAP_COMPARISON.png`: a 17-word Devanagari sentence greedy-wrapped
  to 860 px gives 2 correct lines (830.4 px / 699.1 px) in both pango and Pillow. Hindi
  separates words with spaces, so no dictionary-based line breaker is needed. (Devanagari
  never breaks *inside* a word in our copy, so `MAX_LINES` logic is unaffected.)
- **`text_overlay.text_measurer()`'s architecture survives.** It measures each word once
  and sums widths plus space advances. Measured error for Devanagari:
  **0.0 px on every sample** — no shaping crosses a space boundary. Only the *backend*
  must change from ImageMagick to Pillow; the caching and summing design is correct.
- **Extents are correct** in both winning paths (±2 px vs harfbuzz), so layout is safe.

---

## 4. Word-level alignment

### 4.1 The API answer

`POST /v1/listen`, live key, real TTS audio.

| language | model | `language` param | words returned | timings | ref-token match |
|---|---|---|---|---|---|
| en | `nova-3` | omitted | 31 / 31 | ✅ | **100%** |
| es | `nova-3` | **omitted** | **0** | ❌ | 0% — empty transcript |
| es | `nova-3` | `es` | 33 / 33 | ✅ | **90.9%** |
| es | `nova-3` | `multi` | 33 / 33 | ✅ | 90.9% |
| es | `nova-2` | `es` | 33 / 33 | ✅ | 90.9% |
| es | `whisper-medium` | `es` | 33 / 33 | ✅ | 90.9% |
| hi | `nova-3` | **omitted** | 5 (garbage English) | ✅ | 0% |
| hi | `nova-3` | `hi` | 34 | ✅ | 84.8% — but **transliterates loanwords to Latin** ("Phishing", "link", "click") |
| hi | **`nova-2`** | **`hi`** | 34 | ✅ | **84.8% — pure Devanagari** |
| hi | `whisper-medium` | `hi` | 35 | ✅ | lower (mis-hears क्लिक → खलिक) |

**Hindi word timings ARE obtainable.** Hindi bullets can be speech-anchored. This does not
change the product.

Two API-level actions:

1. `DeepgramAligner.transcribe` sends no `language` parameter. **Spanish alignment returns
   zero words today** — every Spanish bullet would silently fall back to proportional
   placement. This is a one-line fix and it is the highest-value one in this document.
2. Use **`nova-2` for Hindi**, not the configured `nova-3`. `nova-3` code-switches Hindi
   loanwords into Latin script, which breaks the difflib pairing for exactly the words a
   security script uses most (`link`, `click`, `phishing`, `password`).

Docs consulted: <https://developers.deepgram.com/docs/models-languages-overview> — confirms
`es` and `hi` on both nova-3 and nova-2, and that nova-3 multilingual covers Hindi. The docs
do **not** state whether word timestamps are returned per language; that is why it was
measured.

### 4.2 The real blocker is our own tokenizer, not the API

`app/providers/deepgram_align.py:43` — `_NON_WORD = re.compile(r"[^a-z0-9]+")`

```
normalize('फ़िशिंग')  -> ''          normalize('सुरक्षित') -> ''
normalize('dirección') -> 'direccin'  normalize('año') -> 'ao'
tokenize('फ़िशिंग हमले को पहचानें।') -> []          # <-- every token dropped
```

`tokenize()` filters out any token whose `normalize()` is empty, so **`align()` returns
`[]` for Hindi no matter what Deepgram sends back.** Proven end to end: with real Hindi
audio and 35 STT words in hand, `align_tokens` produced 0 words and all four bullets fell
to `method="proportional"`.

`app/providers/bullet_timing.py:77` — `_TOKEN = re.compile(r"[A-Za-z0-9']+")`

| input | `_TOKEN.findall` | consequence |
|---|---|---|
| `Check the sender domain` | `['Check','the','sender','domain']` | fine |
| `Verifica la dirección del remitente` | `['Verifica','la','direcci','n','del','remitente']` | **`dirección` shatters**; a spurious content word `n` is injected |
| `फ़िशिंग हमले को पहचानें` | `[]` | total failure |

Measured Spanish damage, `find_anchors` on real audio:

- bullet `dirección del remitente` anchors at index **3 (`del`)** instead of index 2
  (`dirección`) — the accented head word is unmatchable, so the anchor degrades to the two
  least distinctive words in the sentence.
- bullet `Revisa la información` **false-positive fuzzy-matched** `Verifica la dirección`.
  Accent-stripping makes unrelated phrases look similar enough to clear `FUZZY_THRESHOLD`.
  This is a *wrong* anchor, not a missing one, which is the worse failure.
- Of 25 common Spanish function words, **only `no`** is in `_STOPWORDS` (and that is
  coincidence). So `_find_run`'s stopword-skipping does not work in Spanish, and
  `_content_words` never drops `la`/`de`/`del`/`el`, so `_emphasis_index` scores anchors on
  articles.

### 4.3 `\w` is NOT the fix — it drops Devanagari combining marks

This is the trap. Every matra and the virama are Unicode category `Mn`/`Mc`, and Python's
`\w` excludes marks:

| char | name | category | matches `\w` |
|---|---|---|---|
| `फ` U+092B | DEVANAGARI LETTER PHA | `Lo` | yes |
| `र` U+0930 | DEVANAGARI LETTER RA | `Lo` | yes |
| `्` U+094D | DEVANAGARI SIGN VIRAMA | **`Mn`** | **no** |
| `ज` U+091C | DEVANAGARI LETTER JA | `Lo` | yes |
| `ी` U+0940 | DEVANAGARI VOWEL SIGN II | **`Mc`** | **no** |

| tokenizer | en | es | hi |
|---|---|---|---|
| `[A-Za-z0-9']+` (today) | ✅ | ❌ `dirección`→2 tokens | ❌ `[]` |
| `\w+` with `re.UNICODE` | ✅ | ✅ | ❌ `फर्जी`→`['फर','ज']`, 5 words→9 tokens |
| `[^\W_]+` with `re.UNICODE` | ✅ | ✅ | ❌ identical failure |
| **NFC + keep categories `L`/`N`/`M`, split on whitespace** | ✅ | ✅ | ✅ **5 words→5 tokens** |

```python
import unicodedata

def normalize(token: str) -> str:
    """Comparison key: NFC, case-folded, letters/numbers/MARKS only."""
    token = unicodedata.normalize("NFC", token).casefold()
    return "".join(c for c in token
                   if unicodedata.category(c)[0] in "LNM")
```

Keeping category `M` is the load-bearing detail. Dropping marks would make `पता` and
`पते` compare equal — collapsing distinct Hindi words and producing confident wrong
anchors. Note `casefold()` (not `lower()`): Devanagari is caseless so it is a no-op there,
but it is correct for Spanish and for Turkish if that ever comes up.

### 4.4 End-to-end proof that anchoring works after the fix

`e2e_anchor2.py` — real TTS audio → Deepgram with `language=` → `align_tokens` →
`find_anchors`, with the Unicode `normalize()` and the category-based tokenizer:

| scene | aligner | ref tokens | measured (non-interpolated) timings | verbatim n-gram anchors |
|---|---|---|---|---|
| es #1 | nova-3 `es` | 34 | **34 / 34** | **4 / 4** (runs of 2–3) |
| es #2 | nova-3 `es` | 34 | **34 / 34** | **4 / 4** (runs of 3) |
| hi #1 | nova-2 `hi` | 34 | **34 / 34** | **4 / 4** (runs of 4) |
| hi #2 | nova-2 `hi` | 34 | **34 / 34** | **4 / 4** (runs of 3–5) |

**16 / 16 bullets anchored on verbatim n-grams, every word on a measured timing.** Sample:

```
[OK] 'व्यक्तिगत पासवर्ड कभी साझा'  method=ngram run=4 t=6.16s
     matched=['व्यक्तिगत','पासवर्ड','कभी','साझा']
[OK] 'Archivos adjuntos desconocidos' method=ngram run=3 t=10.64s
     matched=['Archivos','adjuntos','desconocidos']
```

`FUZZY_THRESHOLD = 0.55` was calibrated on English inflection drift and is `UNVERIFIED`
for Devanagari. It never fired in these tests because every bullet matched verbatim; if
Hindi copy ever paraphrases, recalibrate before trusting it.

### 4.5 Other ASCII assumptions found

| file:line | code | effect on hi | effect on es |
|---|---|---|---|
| `deepgram_align.py:43` | `[^a-z0-9]+` | **fatal** — `align()` returns `[]` | accent-stripping collisions |
| `bullet_timing.py:77` | `[A-Za-z0-9']+` | **fatal** — no tokens | words shatter |
| `bullet_timing.py:60` | `_STOPWORDS` (English) | no Hindi stopwords → anchors on `का`/`को` | 24 of 25 es function words missing |
| `bullet_timing.py:71` | `_IMPERATIVES` (English) | emphasis pick is random | emphasis pick is random |
| `gemini_script.py:1006,1049,1078` | `[A-Za-z0-9'-]+` | headings/bullets derived from empty word lists | words shatter |
| `gemini_script.py:531` | `_BULLET_GLYPH` `[A-Za-z][.)]` | benign | benign |
| `ssml.py:455` | `matchable=bool(re.search(r"[A-Za-z0-9]", raw))` | **every Devanagari token is "unmatchable"** | fine |
| `ssml.py:402` | `_CORE_RE` `[^\w]*` trailing strip | **corrupts words**: `पहचानें।` → core `पहचान`, trail `ें।` — strips real matras as punctuation | fine |
| `captions.py:86` | ASS style hardcodes `Arial` | tofu captions | fine |
| `api/voices.py:158,41` | Deepgram filtered to `endswith("-en")`; `POLLY_LANGUAGE="en-US"` | **UI cannot offer any Hindi voice** | UI cannot offer any Spanish voice |

`ssml.py` matters because Hindi is forced onto Polly, which is the SSML-consuming engine —
so Hindi is the *only* language that will actually exercise the SSML path, and it is the one
the SSML tokenizer mangles.

---

## 5. Typography

### 5.1 Line height — measured, not guessed

`typography.py`, freetype metrics + real ink bounding boxes via Pillow, at the DIRECTION §2
sizes. `ftLine` = freetype ascent + descent, i.e. the font's own single-line advance.

| font | `ftLine` @44 | `ftLine` @78 | as em | × Arial | worst ink @44 | ink / DIRECTION 1.22em box |
|---|---|---|---|---|---|---|
| Arial Bold (Latin) | 50 px | 88 px | 1.13 | 1.00 | 49 px | 0.91 ✅ |
| Devanagari Sangam MN | 60 px | 107 px | 1.37 | **1.21×** | 56 px | **1.04 ❌** |
| Kohinoor Devanagari | 63 px | 110 px | 1.43 | **1.26×** | 65 px | **1.20 ❌** |
| ITF Devanagari | 63 px | 110 px | 1.43 | **1.25×** | 64 px | **1.19 ❌** |

Worst-case ink strings exercised both extremes: tall matras plus anusvara
(`फ़िशिंग को पहचानें कैसे ौ ैं`) and deep ones plus below-base conjuncts
(`क्षुद्र त्रुटि सूचीबद्ध हुँ`).

**DIRECTION's 1.22 em line height is exceeded by every Devanagari font**, by 4% (Sangam) to
20% (Kohinoor). Shrinking the point size to fit is not available: 54 px / 1.477 = **36.5 px**,
below both the 40 px 1080p floor and the BBC 44 px HD floor. The line box must grow.

**Recommendation: Devanagari line height `1.50 em` (vs Latin `1.22 em`) — a `1.23×`
multiplier.** It clears the worst measured ink (1.477 em) and matches what the fonts
themselves declare in `hhea` (Kohinoor and ITF both 1.50 em).

| token | size | Latin line height | **Devanagari line height** |
|---|---|---|---|
| `kicker` | 32 px | 42 px (1.30) | **48 px** (1.50) |
| `bullet` | 44 px | 54 px (1.22) | **66 px** (1.50) |
| `heading` | 78 px | 95 px (1.22) | **117 px** (1.50) |
| `title` | 105 px | 128 px (1.22) | **158 px** (1.50) |

### 5.2 The grid survives — with a per-script bullet pitch

DIRECTION §3.3 bullet pitch is 84 px = 54 px line + 30 px gap. Keep the 30 px gap:

| | Latin | Devanagari |
|---|---|---|
| bullet pitch | 84 px | **96 px** (66 + 30) |
| bullet cap-tops (§6.2) | 494 / 578 / 662 / 746 | **494 / 590 / 686 / 782** |
| bottom of bullet 4 | 800 px | **848 px** |
| text column bottom | 990 px | 990 px ✅ |

**The fixed first-bullet baseline `y = 494` is preserved, and 4 bullets still fit inside the
904×900 text column.** Rendered proof at full 1920×1080 on the real grid for hi / es / en:
`slide.py` → `SLIDES.png`. All headings fit 904 px, all bullets fit 860 px.

### 5.3 Two things the type scale *does* need per script

**(a) Apparent size.** Latin legibility is set by cap height; Devanagari's analogue is the
base-consonant body between the headstroke and the baseline. Measured on bare consonants
(`कमनपसतथ`, no matras):

| font | body / Latin cap @44 | @78 | px needed for parity |
|---|---|---|---|
| Devanagari Sangam MN | 0.938 | 0.914 | 47 / 85 px |
| **Kohinoor Devanagari** | **1.000** | **0.948** | **44 / 82 px** |

Kohinoor is at parity at 44 px and 5% under at 78 px — so with Kohinoor the 44 px bullet
genuinely clears the BBC floor and **no size change is needed**. With Sangam MN you would
have to go to 47 px. This is the second reason to pick Kohinoor, after its Latin coverage.

**(b) Baseline anchoring — a silent grid break.** DIRECTION §6.2 specifies *cap-top* y
values. Pillow's `anchor="la"` and pango both place the *ascender* top, and the
ascender-to-cap gap is font-specific:

| size | Latin ink below anchor | Devanagari ink below anchor | **drift** |
|---|---|---|---|
| 78 px | 14 px | 30 px | **+16 px** |
| 44 px | 8 px | 17 px | **+9 px** |

Reusing a single `y=330` / `y=494` across scripts moves the Hindi text block 9–16 px down
relative to Latin — visible against a fixed 88 px accent rule, and exactly the class of
defect DIRECTION §2.1 exists to prevent. **Fix: anchor on the baseline (`anchor="ls"`) and
derive the baseline from the spec cap-top per font:**

| font | heading baseline | bullet baseline |
|---|---|---|
| Arial Bold | 330 + 57 = **387** | 494 + 32 = **526** |
| Kohinoor Deva Bold | 330 + 52 = **382** | 494 + 30 = **524** |

### 5.4 Character caps

`HEADING_CHAR_MAX = 22` / `BULLET_CHAR_MAX = 34` are derived from
`AVG_GLYPH_RATIO = 0.52`. Measured advance per character at 44 px (Kohinoor Bold vs Arial
Bold), which confirms 0.52 for Latin and shows Devanagari is **narrower**:

| script | sample | em / char |
|---|---|---|
| Latin | `Check the sender domain` | 0.529 |
| Latin | `Archivos adjuntos desconocidos` | 0.524 |
| Devanagari | `कर्मचारी प्रशिक्षण` | **0.369** |
| Devanagari | `फ़िशिंग हमले को पहचानें` | **0.402** |
| Devanagari | `व्यक्तिगत पासवर्ड कभी साझा` | **0.422** |

Devanagari runs **0.37–0.42 em/char, roughly 72–80% of Latin**, because matras stack
vertically rather than consuming advance. So the existing caps are *conservative* for
Hindi, not tight. On real generated copy Hindi headings came in at 12–18 chars and every
bullet fit 860 px with room to spare. **No change needed** — but a character count is now
the wrong *kind* of limit for Hindi, and a measured-width check is the honest replacement.

---

## 6. Text length and word budgets

### 6.1 Two different quantities, measured separately

`budget.py` — one English narration per role written the way DIRECTION §5 wants it,
translated preserving sentence count and structure, synthesised with the production voices.
Because structure is held fixed, the cross-language comparison is valid.

**(a) Translation expansion — words to say the same thing:**

| role | en | es | hi |
|---|---|---|---|
| TITLE | 18 | 19 (1.06×) | 24 (**1.33×**) |
| CONTENT | 47 | 58 (1.23×) | 63 (**1.34×**) |
| SUMMARY | 30 | 28 (0.93×) | 31 (1.03×) |
| CLOSING | 23 | 27 (1.17×) | 33 (**1.43×**) |
| **total** | **118** | **132 (1.12×)** | **151 (1.28×)** |

**(b) Speaking rate — words per second, production voices:**

| role | en draco | es celeste | hi Kajal |
|---|---|---|---|
| TITLE | 3.24 | 3.03 | 3.40 |
| CONTENT | 2.92 | 3.44 | 3.74 |
| SUMMARY | 2.40 | 2.51 | 2.81 |
| CLOSING | 2.74 | 2.78 | 3.04 |
| **mean** | **2.82** | **2.94** | **3.25** |
| **ratio to en** | 1.000 | **1.040** | **1.151** |

### 6.2 Corrected word budgets

A word budget exists to hit a duration, so it scales with words-per-second: es **×1.04**,
hi **×1.15**.

| role | duration (min, tgt, max) | **en** (unchanged) | **es** ×1.04 | **hi** ×1.15 |
|---|---|---|---|---|
| TITLE | 4.0, 4.5, 6.5 | 9, **10**, 14 | 9, **10**, 15 | 10, **12**, 16 |
| CONTENT | 11.0, 15.0, 19.0 | 25, **34**, 43 | 26, **35**, 45 | 29, **39**, 50 |
| SUMMARY | 9.0, 12.0, 14.0 | 20, **27**, 31 | 21, **28**, 32 | 23, **31**, 36 |
| CLOSING | 6.0, 7.5, 9.0 | 13, **17**, 20 | 14, **18**, 21 | 15, **20**, 23 |

Effective `WORDS_PER_MINUTE`: en **135** (unchanged), es **140**, hi **155**.

### 6.3 The caveats that matter more than the numbers

- **Sentence count dominates language.** The same 34-word budget produced 20.2–23.5 s of
  English (staccato: four 6–8 word sentences) but 12.8–13.0 s of Hindi (three longer ones).
  Measured words-per-second across my experiments ranged **2.40–3.80** — a 58% spread —
  driven almost entirely by how many sentence-final pauses the model wrote. The
  cross-language *ratio* transfers; the absolute constant does not.
  **The prompt must constrain sentences per scene, or the word budget means nothing.**
- **Voice choice rivals language.** Within English, `draco` 2.82 w/s vs `thalia` 3.19 w/s
  (+13%). Within Hindi, `Kajal` 3.25 vs `Aditi` 2.80 (−14%). Pin the voice before trusting
  the budget.
- **`Aditi` busts the duration clamp.** The CONTENT sample took **19.69 s** on Aditi against
  DIRECTION's hard 19.0 s max. Another reason Kajal/neural is the Hindi voice.
- **English is the language currently out of spec.** Real generated 34-word CONTENT scenes:
  en 23.48 s / 20.24 s (**over** the 19.0 s max), es 14.96 s / 15.88 s (on the 15.0 s
  target), hi 12.82 s / 13.01 s (inside, near the floor). Spanish and Hindi fit the existing
  clamps better than English does.
- **Content per slide drops.** Word budgets scale by 1.04× / 1.15× but expansion is
  1.12× / 1.28×. So at budget, a Spanish slide carries ~93% and a Hindi slide ~90% of the
  English information. That is a real teaching-content loss and it cannot be bought back
  without breaking the duration clamps. Accept it, or add a scene.

---

## 7. Script generation

`gemini_script.py` has **no language parameter anywhere** — `_build_prompt` is
English-only, with English worked examples. A `language` argument must be threaded through
`ScriptProvider.generate` → `_build_prompt` → the API job model.

Tested with `gemini-3.7-flash` (the pinned `VIDEO_DEFAULT_LLM_MODEL`) on the **existing
`RESPONSE_SCHEMA`**, adding only a language instruction (`scriptgen.py`):

| | Spanish | Hindi |
|---|---|---|
| structured output valid | ✅ | ✅ |
| correct script/orthography | ✅ | ✅ Devanagari throughout |
| title within 50 chars | ✅ 22 | ✅ 25 |
| headings within 22 chars | ✅ 15–19 | ✅ 12–18 |
| bullets within 34 chars | ✅ 17–30 | ✅ 13–26 |
| narration hit target word count | ✅ exact | ✅ exact |
| sentence case, no terminal punctuation | ✅ | n/a (Devanagari is caseless) |

**The verbatim-anchor invariant holds in both languages.** Every bullet reused ≥2
consecutive content words verbatim, scored with a Unicode tokenizer:

| language | bullets | anchor run length | under today's ASCII rule |
|---|---|---|---|
| en | 10 / 10 | 2–4 | 2–4 ✅ |
| es | 6 / 6 | 2–4 | 2–6 — **inflated**, `atención`→`atenci`+`n` fakes a longer run |
| hi | 10 / 10 | **2–5** | **0 — every bullet reads as unanchored** |

**Is the anchor rule expressible in Hindi given inflection? Yes, and more easily than in
Spanish.** Hindi marks case with *separate postpositions* (`का`, `को`, `से`, `पर`) rather
than by inflecting the noun, so a bullet can lift a noun phrase out of running narration
without touching a single character: `प्रेषक का ईमेल पता` appears verbatim inside
`संदिग्ध संदेश मिलने पर प्रेषक का ईमेल पता जांचें।` Spanish is the harder case, because
gender and number agreement (`el enlace sospechoso` / `los enlaces sospechosos`) forces
re-inflection. Nothing here needed relaxing; the prompt's existing "identical surface
forms, character for character" wording is achievable in both.

One structural bug: `_apply_roles` → `_clean_bullets` calls
`bullet_timing.anchor_position()`, which returns `None` for **every** Hindi bullet under the
ASCII tokenizer. Measured: `anchor_position('फ़िशिंग हमले', <narration containing it>)`
→ `None`. So even a perfectly anchored Hindi script would have its bullets treated as
unanchored, reordered or replaced by the fallback path before rendering.

---

## 8. What will look or sound wrong if we ship this naively

Ordered by how bad it is.

| # | symptom | language | cause | fix |
|---|---|---|---|---|
| 1 | **Text panels are completely empty.** Correct timing, correct layout, no glyphs. No error anywhere. | hi | `find_font()` → Arial Bold, zero Devanagari coverage | §3.4 Kohinoor + a glyph-coverage assertion |
| 2 | Words render as **different words**. `फ़िशिंग`→`फशिागि`, `सुरक्षित`→`सुरक्षति` | hi | ImageMagick has no Raqm; nukta dropped, i-matra unreordered, conjuncts unformed | §3.5 Pillow + Raqm |
| 3 | **Every bullet drifts off the narration** and lands on an even cadence | es | aligner sends no `language=`; nova-3 returns 0 words; all bullets → `proportional` | §4.1 pass `language` |
| 4 | Same, total failure | hi | `tokenize()` drops all Devanagari; `align()` returns `[]` | §4.3 Unicode `normalize()` |
| 5 | Bullets land on `del` / `la` instead of the noun; an unrelated bullet **confidently anchors to the wrong phrase** | es | `dirección` shatters; fuzzy false positive at 0.55 | §4.3 tokenizer + rebuild `_STOPWORDS` |
| 6 | Matras and anusvara **clipped** at the top and bottom of every line; descenders collide with the next bullet | hi | 1.22 em box vs 1.477 em worst ink | §5.1 line height 1.50 em, pitch 96 px |
| 7 | The Hindi text block sits 9–16 px lower than every other language's, breaking the fixed baseline | hi | anchoring on ascender not baseline | §5.3(b) `anchor="ls"` |
| 8 | Wraps and shrink-steps fire at the wrong place; a heading that fits is shrunk anyway | hi | width measured 24% too wide | §3.2 measure with the renderer that draws |
| 9 | Narration says `link`/`click` in English mid-Hindi-sentence and the caption disagrees | hi | `nova-3` transliterates loanwords to Latin | §4.1 use `nova-2` |
| 10 | Slides feel thin — less taught per slide than the English cut | es, hi | expansion 1.12× / 1.28× exceeds budget scaling 1.04× / 1.15× | §6.3 accept, or add a scene |
| 11 | Scenes overrun the 19.0 s CONTENT clamp and the pipeline asserts | hi | `Aditi` runs 19.69 s at budget | §2.3 `Kajal` on `neural` |
| 12 | Hindi narration sounds like the same woman in every video, with an Indian-English accent | hi | exactly one usable voice exists on any provider | none — product decision |
| 13 | Every Hindi video sounds slightly *worse* than it needs to | hi | default `video_polly_engine=generative`; generative scores 75.8% vs neural 84.8% | §2.2 force `neural` for hi |
| 14 | Hindi SSML silently does nothing, or mangles words | hi | `ssml.py` `matchable` is ASCII-only; `_CORE_RE` strips matras as punctuation | §4.5 |
| 15 | Karaoke captions are tofu | hi | `captions.py` hardcodes `Arial` (and this ffmpeg has no `ass` filter at all) | §4.5 |
| 16 | The UI offers no way to pick a Spanish or Hindi voice | es, hi | `api/voices.py` filters to `-en` / `en-US` | §4.5 |

---

## 9. Implementation order

Ordered by risk retired per line of code. **This is a research document; no production code
was written.**

1. **Unicode `normalize()` + category-based tokenizer** in `deepgram_align.py` and
   `bullet_timing.py` (§4.3). Unblocks Hindi alignment entirely and fixes Spanish accent
   anchoring. Smallest change, largest effect. Do not use `\w`.
2. **Pass `language` to `/v1/listen`**, and select `nova-2` for `hi` (§4.1). One parameter;
   without it Spanish alignment is silently dead.
3. **Font resolution per script + a glyph-coverage assertion.** `Kohinoor.ttc` index 3 for
   all three languages; fail the job loudly if the chosen font lacks a codepoint in the
   copy. This is what turns defect #1 from invisible into impossible.
4. **Swap the text raster backend to Pillow + Raqm** (§3.5), measuring and drawing from the
   same font object. `text_measurer`'s per-word caching design is proven correct (0.0 px
   error) and can be kept as-is.
5. **Per-script line height and baseline anchoring** (§5.1, §5.3): 1.50 em, 96 px bullet
   pitch, `anchor="ls"` with a per-font cap-height offset.
6. **Thread a `language` through script generation** (§7), and constrain sentences per scene
   in the prompt so the word budget is meaningful (§6.3).
7. **Per-language word budgets and voice defaults** (§6.2, §2.3). Force `Kajal`/`neural`
   for Hindi regardless of `video_polly_engine`.
8. **Widen `api/voices.py`** to serve es/hi catalogues (§4.5).
9. **Rebuild `_STOPWORDS` and `_IMPERATIVES` per language** (§4.5). Until then, Spanish and
   Hindi emphasis selection is arbitrary — cosmetic, so it is last.
10. `ssml.py` Unicode correctness — only reachable on the Polly path, i.e. Hindi. Needed
    before Hindi SSML is enabled, not before Hindi ships with SSML off.
