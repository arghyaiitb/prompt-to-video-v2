"""Background-image generation.

VERIFIED against the live key on 2026-08-17 with model `gemini-3.1-flash-image`:

    POST /v1beta/models/gemini-3.1-flash-image:generateContent?key=...
    {"contents":[{"parts":[{"text": "..."}]}],
     "generationConfig":{"responseModalities":["IMAGE"],
                         "imageConfig":{"aspectRatio":"16:9","imageSize":"2K"}}}

    -> candidates[0].content.parts == [
         {"inlineData": {"mimeType": "image/jpeg", "data": "<base64>"},
          "thoughtSignature": "<base64>"}
       ]

Two things differ from the assumed shape and both matter:
  * the mime type is image/JPEG, never png, for this model;
  * `inlineData` and `thoughtSignature` sit on the SAME part, so a part filter that
    requires "the part has exactly one key" or that skips parts containing
    thoughtSignature drops the image entirely.

`imageConfig` is honoured: 9:16 returns 768x1376, and imageSize "2K" returns 2752x1536
against 1376x768 at the default. Note that the model's "16:9" is really 1.792:1
(2752/1536), a hair wider than true 16:9 — the render module fits it, we never stretch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.config import get_settings
from app.providers._gemini import generate_content, inline_data_from
from app.providers._media import image_dimensions, run_ffmpeg, transcode

# Aspect ratios the API accepts. Requested width/height snaps to the nearest.
SUPPORTED_RATIOS: tuple[tuple[str, float], ...] = (
    ("21:9", 21 / 9),
    ("16:9", 16 / 9),
    ("4:3", 4 / 3),
    ("3:2", 3 / 2),
    ("1:1", 1.0),
    ("2:3", 2 / 3),
    ("3:4", 3 / 4),
    ("9:16", 9 / 16),
)

# Long edge the API delivers per imageSize tier, measured.
_TIER_LONG_EDGE = {"1K": 1376, "2K": 2752}

NEGATIVE_SUFFIX = (
    "No text, no letters, no words, no numbers, no labels, no signage, no watermarks, "
    "no logos, no captions anywhere in the image."
)

STYLE_SUFFIX = (
    "Photographic, cinematic, high detail, natural lighting, wide landscape composition "
    "with generous empty negative space in the lower third for a caption overlay."
)


class GeminiImageProvider:
    """Satisfies `ImageProvider`.

    Writes whatever the model returns to `out_path` without rescaling. If the requested
    file extension contradicts the returned mime type the bytes are remuxed (never
    resampled) so the file on disk is not a jpeg wearing a .png name.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 240.0,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.video_default_image_model
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.timeout = timeout

    def generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        body = {
            "contents": [{"parts": [{"text": _compose_prompt(prompt)}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": nearest_ratio(width, height),
                    "imageSize": _image_size_tier(width, height),
                },
            },
        }

        response = generate_content(self.model, body, self.api_key, timeout=self.timeout)
        mime, data = inline_data_from(response)
        return _write_image(data, mime, out_path)


class PlaceholderImageProvider:
    """Satisfies `ImageProvider` with a locally generated gradient — no API calls.

    Exists so the render and worker stages can be exercised end to end for free. Both the
    colour and the filter seed are derived from a hash of the prompt, so a given scene
    always gets the same placeholder and visual diffs across runs stay meaningful.

    `gradients` defaults to `seed=-1` (random) and `speed=0.01` (it rotates over time), so
    both are pinned explicitly — without that, two runs of the same prompt produce
    different pixels.
    """

    def __init__(self, *, saturation: float = 0.55, lightness: float = 0.38) -> None:
        self.saturation = saturation
        self.lightness = lightness

    def generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        width = max(2, int(width))
        height = max(2, int(height))
        c0, c1 = self._colors(prompt)
        seed = int.from_bytes(_digest(prompt)[4:8], "big")
        run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"gradients=s={width}x{height}:c0={c0}:c1={c1}:x0=0:y0=0"
                f":x1={width}:y1={height}:nb_colors=2:d=1:seed={seed}:speed=0",
                "-frames:v",
                "1",
                str(out_path),
            ]
        )
        return out_path

    def _colors(self, prompt: str) -> tuple[str, str]:
        hue = (_digest(prompt)[0] / 255.0) * 360.0
        base = _hsl_to_hex(hue, self.saturation, self.lightness)
        # Complementary-ish second stop keeps the gradient readable behind a scrim.
        other = _hsl_to_hex((hue + 40.0) % 360.0, self.saturation, self.lightness * 0.55)
        return base, other


# --------------------------------------------------------------------------- helpers


def _digest(prompt: str) -> bytes:
    return hashlib.sha256(prompt.encode("utf-8")).digest()


def nearest_ratio(width: int, height: int) -> str:
    """Snap an arbitrary pixel size to the closest API-supported aspect ratio string."""
    if width <= 0 or height <= 0:
        return "16:9"
    wanted = width / height
    return min(SUPPORTED_RATIOS, key=lambda pair: abs(pair[1] - wanted))[0]


def _image_size_tier(width: int, height: int) -> str:
    """Ask for 2K whenever 1K would land under the requested frame.

    1K tops out at a 1376px long edge, so a 1920x1080 render fed a 1K image is being
    upscaled before zoompan even starts — visibly soft. 2K (2752px) leaves headroom.
    """
    long_edge = max(int(width), int(height))
    return "1K" if long_edge <= _TIER_LONG_EDGE["1K"] else "2K"


def _compose_prompt(prompt: str) -> str:
    """Append style and no-text guards unless the caller already said them.

    The script provider is asked to include the negative clause, but an image must never
    depend on an upstream model having complied.
    """
    text = prompt.strip().rstrip()
    lowered = text.lower()
    parts = [text if text.endswith((".", "!", "?")) else f"{text}."]
    if "negative space" not in lowered and "open space" not in lowered:
        parts.append(STYLE_SUFFIX)
    if "no text" not in lowered:
        parts.append(NEGATIVE_SUFFIX)
    return " ".join(parts)


_MIME_EXTENSIONS = {
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/webp": {".webp"},
}


def _write_image(data: bytes, mime: str, out_path: Path) -> Path:
    """Persist bytes at out_path, remuxing only if the extension disagrees with the mime."""
    expected = _MIME_EXTENSIONS.get(mime.lower(), set())
    if not expected or out_path.suffix.lower() in expected:
        out_path.write_bytes(data)
        return out_path

    native_suffix = sorted(expected)[0]
    staged = out_path.with_name(f"{out_path.stem}.raw{native_suffix}")
    staged.write_bytes(data)
    try:
        transcode(staged, out_path)
    finally:
        staged.unlink(missing_ok=True)
    return out_path


def _hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    """ffmpeg's gradients filter wants 0xRRGGBB."""
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    hue_prime = (hue % 360.0) / 60.0
    secondary = chroma * (1 - abs(hue_prime % 2 - 1))
    sector = int(hue_prime) % 6
    rgb = [
        (chroma, secondary, 0.0),
        (secondary, chroma, 0.0),
        (0.0, chroma, secondary),
        (0.0, secondary, chroma),
        (secondary, 0.0, chroma),
        (chroma, 0.0, secondary),
    ][sector]
    offset = lightness - chroma / 2
    r, g, b = (max(0, min(255, round((c + offset) * 255))) for c in rgb)
    return f"0x{r:02X}{g:02X}{b:02X}"


def actual_dimensions(path: Path) -> tuple[int, int]:
    """Convenience re-export: what the model really gave us, for logging and tests."""
    return image_dimensions(path)
