"""Brand-logo upload store. `POST /api/logos` replaces the mark burned into every video.

This is the one endpoint in the app that takes an untrusted *file* and hands it to image
tooling, so the validation here is deliberately paranoid and every rule states its threat:

*Streamed size cap.* The ceiling (`settings.video_logo_max_bytes`) is enforced chunk by
chunk while the body arrives, not after buffering it — otherwise the cap only limits what
we *keep*, and a 2 GiB upload has already cost the memory by the time it is measured.

*Content, never the client's word for it.* The format is decided from PNG magic bytes and
from whether the document actually parses as XML with an `<svg>` root. The filename and
`Content-Type` are used for exactly one thing: choosing which diagnostic to return, so a
file called `logo.png` that is not a PNG says so instead of "unsupported format".

*Dimensions before decode.* Sizes come from the PNG IHDR header and from the SVG's own
`width`/`height`/`viewBox` — parsed, not decoded. A 40 KB file that expands to
99999x99999 is rejected having never allocated a bitmap.

*Generated ids.* The stored name is a content hash, never the uploaded filename, so
`../../etc/x.png` is not a path we could be tricked into writing: the only thing that
reaches the filesystem is 32 hex characters. Lookups re-validate against
:data:`LOGO_ID_RE` before touching the disk, so a traversal id 404s rather than resolving.

*SVG is rasterised at upload time.* ImageMagick on this box has no `rsvg-convert`
delegate, so its built-in MSVG renderer implements neither `<mask>` nor `<filter>` and
turns such groups into black blobs — the app's own favicon is exactly that shape. Rather
than discover it in a finished video, every SVG is rasterised here, the PNG is stored
alongside as the render asset (`{id}.render.png`), and any construct the renderer cannot
be faithful to comes back as a `warnings` entry on the upload response. PNG with alpha is
the preferred input and needs none of this.

Metadata lives in a JSON sidecar next to the file rather than a DB table: the store is
then self-describing and portable, `cache/logos` can be copied or wiped as a unit, and a
logo is not a row anybody joins against — the only relational fact is `Job.logo_id`,
which is what makes DELETE a 409 while a job still names it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET  # noqa: S405 - never fed a DTD; see _check_svg_text
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final, Literal, NoReturn

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import NO_LOGO_ID, Job
from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["logos"])

SessionDep = Annotated[Session, Depends(get_session)]

LogoFormat = Literal["png", "svg"]

NO_LOGO: Final = NO_LOGO_ID
"""Re-exported for the HTTP layer: the ``logo_id`` a client sends for "no mark at all".

A distinct answer from omitting ``logo_id``, which means "the bundled default". Defined in
``app.db.models`` because it is a *stored* value; see the note there.
"""

ID_LENGTH: Final = 32
"""Hex characters of SHA-256 kept as the logo id.

128 bits of a content hash: collision-proof for the purpose, short enough to sit in a URL,
and content-addressed so re-uploading the same file is idempotent rather than a leak.
"""

LOGO_ID_RE: Final = re.compile(rf"^[0-9a-f]{{{ID_LENGTH}}}$")
"""The only shape a stored id can have. Every path is built from a string that matched
this, which is what makes filesystem traversal impossible rather than merely unlikely."""

CHUNK_BYTES: Final = 64 * 1024

PNG_MAGIC: Final = b"\x89PNG\r\n\x1a\n"

MAX_PNG_CHUNKS: Final = 8192
"""Chunk-table entries walked before we call a PNG malformed. Bounds the header scan on a
file crafted to be one enormous chunk list."""

RASTER_HEIGHT: Final = 512
"""Height an SVG is rasterised to at upload, in pixels.

The watermark is ~49px tall at 1080p and the renderer re-rasterises to exactly that, so
512 is roughly 10x oversampled: downscaling to the final height stays sharp, and the file
is a few tens of KB.
"""

RASTER_TIMEOUT_S: Final = 30

_BLANK_ALPHA_FLOOR: Final = 0.02
"""Mean alpha below which the raster is treated as having produced nothing visible.
Mirrors ``text_overlay.LOGO_MIN_ALPHA_COVERAGE`` — the renderer silently skips a mark this
empty, so the upload has to say so."""

SVG_NS: Final = "http://www.w3.org/2000/svg"

_HOSTILE_MARKERS: Final = (
    ("<!doctype", "an XML DOCTYPE"),
    ("<!entity", "an entity declaration"),
    ("<?xml-stylesheet", "an external stylesheet instruction"),
)
"""Rejected on the raw text, *before* the document reaches a parser.

A DOCTYPE is refused outright rather than only when it declares entities: a brand mark has
no legitimate use for a DTD, and this way billion-laughs expansion and external-entity
retrieval are both structurally impossible instead of being someone's parser setting.
CPython's ElementTree does not resolve external entities, but "the stdlib currently does
not" is not a control.
"""

_UNSAFE_ELEMENTS: Final = frozenset({"script", "foreignobject", "handler", "set", "animate"})
"""Elements that carry behaviour rather than shape. ``<set>``/``<animate>`` can retarget
another element's attributes, which is scripting with extra steps."""

_UNRENDERABLE_ELEMENTS: Final = {
    "mask": "<mask>",
    "filter": "<filter>",
    "clippath": "<clipPath>",
    "pattern": "<pattern>",
    "use": "<use>",
    "style": "<style>",
    "text": "<text>",
    "tspan": "<tspan>",
    "image": "<image>",
}
"""Constructs ImageMagick's built-in SVG renderer gets wrong or drops. Reported as
warnings, not errors — the mark still rasterises, it just may not be what was drawn."""

_LENGTH_RE: Final = re.compile(r"^\s*([+-]?[0-9]*\.?[0-9]+)\s*([a-z%]*)\s*$", re.IGNORECASE)

_UNIT_PX: Final = {
    "": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "mm": 96 / 25.4,
    "cm": 96 / 2.54, "in": 96.0, "q": 96 / 101.6,
}
"""User units per CSS unit. ``em``/``ex``/``%`` are deliberately absent: they depend on a
context the file does not carry, so they read as "unknown" and the raster's own size is
used for the dimension check instead."""

_FILE_HEADERS: Final = {
    "X-Content-Type-Options": "nosniff",
    # Served from the API origin, so an SVG is a document the browser will execute in
    # principle. Uploads containing script/hrefs are rejected already; this is the second
    # layer, and it costs nothing.
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
    # Content-addressed: the bytes behind an id can never change.
    "Cache-Control": "public, max-age=31536000, immutable",
}


# ------------------------------------------------------------------------------ records


class StoredLogo(BaseModel):
    """The JSON sidecar — everything about an upload that outlives the request."""

    id: str
    format: LogoFormat
    width: int
    height: int
    has_alpha: bool
    size_bytes: int
    uploaded_at: datetime
    filename: str = ""
    """The uploaded name, sanitised down to a basename, kept only for display. Nothing on
    disk is ever named from it."""

    warnings: list[str] = Field(default_factory=list)


class LogoOut(BaseModel):
    """What the API returns for a logo, on upload and on every read."""

    id: str
    url: str
    """Serves the original upload — PNG as PNG, SVG as SVG. What a picker should preview:
    a browser has a complete SVG renderer, so this is the truthful view of the file."""

    render_url: str
    """Serves the PNG the video is actually branded with. Identical to ``url`` for a PNG
    upload; for an SVG this is the rasterisation, which is what the renderer consumes."""

    format: LogoFormat
    width: int
    height: int
    """Pixel dimensions of the *source*. For an SVG these are its nominal width/height (or
    viewBox), which is what "how big is this mark" means for a vector."""

    has_alpha: bool
    size_bytes: int
    uploaded_at: datetime
    filename: str = ""
    warnings: list[str] = Field(default_factory=list)
    """Non-fatal findings, e.g. an SVG using constructs this box cannot rasterise
    faithfully. An empty list means the stored render is a faithful copy of the upload."""


def _to_out(record: StoredLogo) -> LogoOut:
    return LogoOut(
        **record.model_dump(),
        url=f"/api/logos/{record.id}",
        render_url=f"/api/logos/{record.id}/render",
    )


# ------------------------------------------------------------------------------- store


def logo_dir() -> Path:
    """`cache/logos`, created on demand. Outside `out/`, so job cleanup cannot wipe it."""
    return get_settings().logo_dir


def _valid_id(logo_id: str) -> str | None:
    """The id, or None if it is not one we could have generated."""
    candidate = (logo_id or "").strip().lower()
    return candidate if LOGO_ID_RE.match(candidate) else None


def _meta_path(logo_id: str) -> Path:
    return logo_dir() / f"{logo_id}.json"


def source_path(logo_id: str, fmt: LogoFormat) -> Path:
    return logo_dir() / f"{logo_id}.{fmt}"


def _render_file(logo_id: str, fmt: LogoFormat) -> Path:
    """Where the render-ready PNG lives. For a PNG upload that is the upload itself."""
    return source_path(logo_id, "png") if fmt == "png" else logo_dir() / f"{logo_id}.render.png"


def load_logo(logo_id: str) -> StoredLogo | None:
    """The record for `logo_id`, or None — for an unknown id, a bad id, or a bad sidecar.

    Never raises: a corrupt sidecar means the store has a file nobody can describe, which
    is a missing logo, not a 500.
    """
    valid = _valid_id(logo_id)
    if valid is None:
        return None
    path = _meta_path(valid)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        record = StoredLogo.model_validate(raw)
    except ValueError:
        logger.warning("logo %s has an unreadable sidecar; treating it as missing", valid)
        return None
    if record.id != valid or not _render_file(valid, record.format).is_file():
        return None
    return record


def logo_exists(logo_id: str) -> bool:
    """True when `logo_id` names a usable stored logo. What POST /api/jobs validates on."""
    return load_logo(logo_id) is not None


def logo_render_path(logo_id: str) -> Path | None:
    """The PNG a render should composite for `logo_id`, or None if it is not usable.

    The pipeline's entry point into this module: it stamps the returned path onto
    ``Timeline.logo_path`` so a re-render reproduces the same branding.
    """
    record = load_logo(logo_id)
    if record is None:
        return None
    return _render_file(record.id, record.format)


def list_stored() -> list[StoredLogo]:
    """Every readable logo, newest first."""
    records = [
        record
        for path in logo_dir().glob("*.json")
        if (record := load_logo(path.stem)) is not None
    ]
    return sorted(records, key=lambda r: r.uploaded_at, reverse=True)


# -------------------------------------------------------------------------- validation


def _reject(code: int, error: str, message: str, **extra: Any) -> NoReturn:
    """Every rejection in one shape: `{error, message, ...}`, so a client can branch."""
    raise HTTPException(status_code=code, detail={"error": error, "message": message, **extra})


@dataclass(frozen=True)
class PngInfo:
    width: int
    height: int
    has_alpha: bool


def read_png_info(path: Path) -> PngInfo:
    """Dimensions and alpha from the PNG header, without decoding a single pixel.

    Also a structural check: magic, an IHDR first chunk, a walkable chunk table and a
    terminating IEND. That is what makes "PNG magic plus 4 MB of noise" a 422 rather than
    something ImageMagick discovers six minutes into a render.
    """
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        if fh.read(len(PNG_MAGIC)) != PNG_MAGIC:
            _reject(422, "png_magic_mismatch", "the file does not start with the PNG signature")
        head = fh.read(25)
        if len(head) < 25 or head[4:8] != b"IHDR" or int.from_bytes(head[:4], "big") != 13:
            _reject(422, "png_malformed", "the PNG has no IHDR header chunk")

        width = int.from_bytes(head[8:12], "big")
        height = int.from_bytes(head[12:16], "big")
        colour_type = head[17]
        _check_dimensions(width, height)
        has_alpha = colour_type in (4, 6)

        offset = 8 + 12 + 13
        terminated = False
        for _ in range(MAX_PNG_CHUNKS):
            fh.seek(offset)
            header = fh.read(8)
            if len(header) < 8:
                break
            length = int.from_bytes(header[:4], "big")
            kind = header[4:8]
            if not kind.isalpha() or length > file_size:
                _reject(422, "png_malformed", "the PNG chunk table is not readable")
            if kind == b"tRNS":
                # Palette/truecolour transparency: alpha without an alpha channel.
                has_alpha = True
            if kind == b"IEND":
                terminated = True
                break
            offset += 12 + length
            if offset > file_size:
                _reject(422, "png_malformed", "a PNG chunk runs past the end of the file")
        if not terminated:
            _reject(422, "png_malformed", "the PNG is truncated: no IEND chunk")
    return PngInfo(width, height, has_alpha)


def _check_dimensions(width: int, height: int) -> None:
    """Bound the bitmap before anything allocates one."""
    limit = get_settings().video_logo_max_dimension
    if width <= 0 or height <= 0:
        _reject(422, "logo_dimensions", f"the logo reports a {width}x{height} size")
    if width > limit or height > limit:
        _reject(
            422,
            "logo_dimensions",
            f"the logo decodes to {width}x{height}, over the {limit}px limit per side",
            width=width,
            height=height,
            max_dimension=limit,
        )


@dataclass(frozen=True)
class SvgInfo:
    width: int
    height: int
    warnings: list[str]


def _svg_length(value: str | None) -> float | None:
    """A CSS length in user units, or None when it is missing or context-dependent."""
    if not value:
        return None
    match = _LENGTH_RE.match(value)
    if match is None:
        return None
    factor = _UNIT_PX.get(match.group(2).lower())
    if factor is None:
        return None
    return float(match.group(1)) * factor


def _local(tag: object) -> str:
    """Namespace-stripped, lower-cased tag or attribute name."""
    name = tag if isinstance(tag, str) else ""
    return name.rsplit("}", 1)[-1].lower()


def _check_svg_text(text: str) -> None:
    """Refuse a document with a DTD or an external stylesheet, before parsing it."""
    lowered = text.lower()
    for marker, description in _HOSTILE_MARKERS:
        if marker in lowered:
            _reject(
                422,
                "svg_unsafe",
                f"the SVG contains {description} ({marker}); "
                "a brand mark must be a plain, self-contained SVG document",
                construct=marker,
            )


def inspect_svg(text: str) -> SvgInfo:
    """Parse and vet an SVG. Returns its nominal size plus any fidelity warnings.

    Rejects, in order: a DTD or external stylesheet (see :data:`_HOSTILE_MARKERS`), a
    document that is not XML, a root that is not `<svg>`, behavioural elements, `on*`
    event handlers, `javascript:` anywhere, and any `href`/`xlink:href` that is not a
    same-document `#fragment` — which covers remote fetches, `file://` reads and
    `data:` payloads in one rule.
    """
    _check_svg_text(text)
    try:
        root = ET.fromstring(text)  # noqa: S314 - DTD-free by the check above
    except ET.ParseError as exc:
        _reject(422, "svg_malformed", f"the file is not well-formed XML: {exc}")
    if _local(root.tag) != "svg":
        _reject(
            422,
            "svg_malformed",
            f"the XML root element is <{_local(root.tag)}>, not <svg>",
        )

    found: set[str] = set()
    for element in root.iter():
        name = _local(element.tag)
        if name in _UNSAFE_ELEMENTS:
            _reject(
                422,
                "svg_unsafe",
                f"the SVG contains <{name}>, which carries behaviour rather than shape",
                construct=name,
            )
        if name in _UNRENDERABLE_ELEMENTS:
            found.add(name)
        for attribute, value in element.attrib.items():
            _check_svg_attribute(_local(attribute), value)

    warnings: list[str] = []
    if found:
        listed = ", ".join(_UNRENDERABLE_ELEMENTS[name] for name in sorted(found))
        warnings.append(
            f"this SVG uses {listed}, which the SVG renderer available on this host "
            f"({_renderer_name()}) does not implement; the stored PNG may not match the "
            "original. Upload a PNG with alpha for an exact mark."
        )
    return SvgInfo(*_svg_dimensions(root), warnings)


def _check_svg_attribute(name: str, value: str) -> None:
    if name.startswith("on"):
        _reject(
            422, "svg_unsafe", f"the SVG sets the event handler {name!r}", construct=name
        )
    if "javascript:" in value.lower():
        _reject(422, "svg_unsafe", "the SVG contains a javascript: URL", construct=name)
    if name == "href" and not value.strip().startswith("#"):
        _reject(
            422,
            "svg_unsafe",
            f"the SVG references {value.strip()[:80]!r}; only same-document #fragment "
            "references are allowed, so the mark cannot pull in anything at render time",
            construct=name,
        )


def _svg_dimensions(root: ET.Element) -> tuple[int, int]:
    """Nominal pixel size from width/height, falling back to the viewBox, else 0x0.

    0x0 means "the document does not say", and the caller measures the raster instead —
    unknown is not the same as invalid, and a viewBox-less SVG is perfectly renderable.
    """
    width = _svg_length(root.get("width"))
    height = _svg_length(root.get("height"))
    if width is None or height is None:
        box = (root.get("viewBox") or "").replace(",", " ").split()
        if len(box) == 4:
            width = width if width is not None else _svg_length(box[2])
            height = height if height is not None else _svg_length(box[3])
    if width is None or height is None:
        return 0, 0
    limit = get_settings().video_logo_max_dimension
    if width > limit or height > limit:
        # Checked here rather than after rasterising: this is the decode bomb, and the
        # point is that nothing ever tries to allocate it.
        _check_dimensions(int(width), int(height))
    return max(0, int(round(width))), max(0, int(round(height)))


# ----------------------------------------------------------------------- rasterisation


def _imagemagick_bin() -> str | None:
    """`magick` (IM7) or `convert` (IM6), or None.

    Duplicated from `text_overlay.imagemagick_bin` on purpose: the HTTP surface must not
    import the renderer, which pulls in fonts, ffmpeg probing and a much larger blast
    radius than "which binary rasterises this file".
    """
    return (
        os.environ.get("IMAGEMAGICK_BIN") or shutil.which("magick") or shutil.which("convert")
    )


def _renderer_name() -> str:
    """Which SVG renderer ImageMagick will actually use — the whole reason for warnings."""
    return "rsvg-convert" if shutil.which("rsvg-convert") else "ImageMagick's built-in MSVG"


def rasterise_svg(source: Path, out_path: Path, nominal_height: int) -> PngInfo:
    """Render `source` to an RGBA PNG of at most :data:`RASTER_HEIGHT` pixels tall.

    Density is derived from the SVG's own height so a 48px favicon is rasterised *at* 512
    rather than rasterised at 48 and upscaled. Resource limits and a hard timeout are set
    because this is untrusted input: a pathological SVG must cost seconds, not the box.
    """
    binary = _imagemagick_bin()
    if binary is None:
        _reject(
            503,
            "no_rasteriser",
            "SVG uploads need ImageMagick to rasterise, and it is not installed; "
            "upload a PNG with alpha instead",
        )
    density = 96.0
    if nominal_height > 0:
        density = min(2400.0, max(96.0, 96.0 * RASTER_HEIGHT / nominal_height))
    limit = get_settings().video_logo_max_dimension
    argv = [
        binary,
        "-limit", "memory", "256MiB",
        "-limit", "map", "512MiB",
        "-limit", "time", "20",
        "-background", "none",
        "-density", f"{density:.0f}",
        f"svg:{source}",
        # A geometry box, not `x512`: a very wide mark must not rasterise to a canvas
        # wider than the dimension cap just because it is short.
        "-resize", f"{limit}x{RASTER_HEIGHT}",
        "-strip",
        f"png32:{out_path}",
    ]
    try:
        subprocess.run(  # noqa: S603 - fixed argv, no shell, paths are ours
            argv, capture_output=True, timeout=RASTER_TIMEOUT_S, check=True
        )
    except subprocess.TimeoutExpired:
        _reject(422, "svg_rasterise_failed", "rasterising the SVG timed out")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()[:300]
        _reject(422, "svg_rasterise_failed", f"the SVG could not be rasterised: {detail}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        _reject(422, "svg_rasterise_failed", "rasterising the SVG produced no output")
    return read_png_info(out_path)


def _mean_alpha(path: Path) -> float | None:
    """Mean alpha of a PNG in 0..1, or None when it cannot be measured.

    Only used to warn: a mark that rasterises to nothing is composited as nothing, and the
    renderer would skip it silently — see ``text_overlay.LOGO_MIN_ALPHA_COVERAGE``.
    """
    binary = _imagemagick_bin()
    if binary is None:
        return None
    argv = (
        [binary, "identify"] if Path(binary).name.startswith("magick") else [
            shutil.which("identify") or binary
        ]
    )
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [*argv, "-format", "%[fx:mean.a]", str(path)],
            capture_output=True, timeout=RASTER_TIMEOUT_S, check=True,
        )
        return float(done.stdout.decode().strip().split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


# ------------------------------------------------------------------------------ upload


def _sanitise_filename(name: str | None) -> str:
    """A basename fit to echo back in JSON. Never used to build a path.

    Both separators are stripped, not just the platform's: an upload is not required to
    have come from this OS, and `..\\..\\x.png` is the same attempt as `../../x.png`.
    """
    raw = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(ch for ch in raw if ch.isprintable() and ch not in '<>:"|?*')
    cleaned = cleaned.strip().strip(".")
    return cleaned[:120]


async def _stream_to_temp(upload: UploadFile, limit: int) -> tuple[Path, int, str]:
    """Spool the body to a temp file beside the store, enforcing `limit` as it arrives.

    Returns `(path, size, sha256-prefix)`. The temp file lives in the store's own
    directory so the commit is an `os.replace` on the same filesystem — atomic, so a
    reader can never see a half-written logo.
    """
    directory = logo_dir()
    handle, raw_path = tempfile.mkstemp(prefix=".incoming-", dir=directory)
    path = Path(raw_path)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(handle, "wb") as sink:
            while chunk := await upload.read(CHUNK_BYTES):
                size += len(chunk)
                if size > limit:
                    _reject(
                        413,
                        "logo_too_large",
                        f"the logo exceeds the {limit} byte limit; "
                        "the upload was stopped without being buffered",
                        max_bytes=limit,
                    )
                digest.update(chunk)
                sink.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if size == 0:
        path.unlink(missing_ok=True)
        _reject(422, "logo_empty", "the uploaded file is empty")
    return path, size, digest.hexdigest()[:ID_LENGTH]


def _claims_png(upload: UploadFile) -> bool:
    """Whether the *client* called this a PNG.

    Used only to pick the error message. The format decision itself is made from the
    bytes, so a lying client gets a precise diagnostic and no special handling.
    """
    name = (upload.filename or "").lower()
    return name.endswith(".png") or (upload.content_type or "").lower() == "image/png"


def _detect_format(path: Path, upload: UploadFile) -> LogoFormat:
    """PNG or SVG, from the content. Anything else is a 415."""
    with path.open("rb") as fh:
        prefix = fh.read(len(PNG_MAGIC))
    if prefix == PNG_MAGIC:
        return "png"
    if _claims_png(upload):
        _reject(
            422,
            "png_magic_mismatch",
            "the upload is declared as PNG but does not start with the PNG signature; "
            "the file content is what decides, and it is not a PNG",
        )
    return "svg"


def _decode_svg(path: Path) -> str:
    text = path.read_bytes()
    try:
        return text.decode("utf-8-sig")
    except UnicodeDecodeError:
        _reject(
            415,
            "unsupported_logo_format",
            "only PNG and SVG are accepted; this file is neither a PNG nor UTF-8 text",
        )


def _write_sidecar(record: StoredLogo) -> None:
    """Atomic metadata write: temp file then replace, so a reader sees all or nothing."""
    target = _meta_path(record.id)
    scratch = target.with_suffix(".json.tmp")
    scratch.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    os.replace(scratch, target)


@router.post("/logos", response_model=LogoOut, status_code=status.HTTP_201_CREATED)
async def upload_logo(file: Annotated[UploadFile, File()]) -> LogoOut:
    """Store a brand mark. PNG (preferred) or SVG; see the module docstring for the rules.

    Content-addressed and therefore idempotent: uploading the same bytes twice returns the
    same id and overwrites nothing meaningful.
    """
    settings = get_settings()
    temp_path, size, logo_id = await _stream_to_temp(file, settings.video_logo_max_bytes)
    filename = _sanitise_filename(file.filename)
    try:
        fmt = _detect_format(temp_path, file)
        warnings: list[str] = []
        if fmt == "png":
            info = read_png_info(temp_path)
            width, height, has_alpha = info.width, info.height, info.has_alpha
            if not has_alpha:
                warnings.append(
                    "this PNG has no alpha channel, so the mark will be composited as a "
                    "rectangle including its background. Export it with transparency."
                )
        else:
            svg = inspect_svg(_decode_svg(temp_path))
            width, height, warnings = svg.width, svg.height, list(svg.warnings)
            has_alpha = True

        stored_source = source_path(logo_id, fmt)
        os.replace(temp_path, stored_source)
    finally:
        temp_path.unlink(missing_ok=True)

    try:
        if fmt == "svg":
            raster = rasterise_svg(stored_source, _render_file(logo_id, fmt), height)
            if width == 0 or height == 0:
                # The document never said how big it is; the raster is the measurement.
                width, height = raster.width, raster.height
            alpha = _mean_alpha(_render_file(logo_id, fmt))
            if alpha is not None and alpha < _BLANK_ALPHA_FLOOR:
                warnings.append(
                    f"the SVG rasterised to an almost entirely transparent image "
                    f"(mean alpha {alpha:.3f}); the renderer will skip a mark this empty"
                )
    except BaseException:
        # Never leave a source file with no render asset behind it: load_logo would keep
        # returning None for it and nothing would ever clean it up.
        stored_source.unlink(missing_ok=True)
        _render_file(logo_id, fmt).unlink(missing_ok=True)
        raise

    record = StoredLogo(
        id=logo_id,
        format=fmt,
        width=width,
        height=height,
        has_alpha=has_alpha,
        size_bytes=size,
        uploaded_at=datetime.now(UTC),
        filename=filename,
        warnings=warnings,
    )
    _write_sidecar(record)
    logger.info(
        "stored brand logo %s (%s %dx%d, %d bytes, %d warning(s)) from upload %r",
        logo_id, fmt, width, height, size, len(warnings), filename,
    )
    return _to_out(record)


# ------------------------------------------------------------------------------- reads


def _get_or_404(logo_id: str) -> StoredLogo:
    record = load_logo(logo_id)
    if record is None:
        # Also the answer for a malformed or traversal id: it cannot name a stored logo,
        # and saying "invalid" would confirm the shape of ids to someone probing.
        raise HTTPException(status_code=404, detail=f"logo {logo_id} not found")
    return record


@router.get("/logos", response_model=list[LogoOut])
def list_logos() -> list[LogoOut]:
    """Every stored logo, newest first — the picker's catalogue."""
    return [_to_out(record) for record in list_stored()]


@router.get("/logos/{logo_id}")
def get_logo_file(logo_id: str) -> FileResponse:
    """The original upload, with its real media type."""
    record = _get_or_404(logo_id)
    path = source_path(record.id, record.format)
    if not path.is_file():  # pragma: no cover - load_logo checks the render asset
        raise HTTPException(status_code=404, detail="logo file is missing on disk")
    media = "image/png" if record.format == "png" else "image/svg+xml"
    return FileResponse(path, media_type=media, headers=_FILE_HEADERS)


@router.get("/logos/{logo_id}/render")
def get_logo_render(logo_id: str) -> FileResponse:
    """The PNG the video is branded with. For a PNG upload, the upload itself."""
    record = _get_or_404(logo_id)
    return FileResponse(
        _render_file(record.id, record.format), media_type="image/png", headers=_FILE_HEADERS
    )


@router.get("/logos/{logo_id}/meta", response_model=LogoOut)
def get_logo(logo_id: str) -> LogoOut:
    """Metadata only — dimensions, warnings, upload time."""
    return _to_out(_get_or_404(logo_id))


@router.delete("/logos/{logo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_logo(logo_id: str, session: SessionDep) -> None:
    """Delete a logo, unless a job still names it.

    409 rather than a soft delete or a cascade: `Timeline.logo_path` points into this
    store, so deleting a mark a job was rendered with would silently change what a
    re-render produces. The 409 names the jobs, so a client can offer to delete them.
    """
    record = _get_or_404(logo_id)
    users = session.exec(select(Job.id).where(Job.logo_id == record.id)).all()
    if users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "logo_in_use",
                "message": (
                    f"{len(users)} job(s) were rendered with this logo; deleting it would "
                    "change what a re-render produces"
                ),
                "logo_id": record.id,
                "jobs": list(users)[:20],
            },
        )
    for path in (
        source_path(record.id, record.format),
        _render_file(record.id, record.format),
        _meta_path(record.id),
    ):
        path.unlink(missing_ok=True)
    logger.info("deleted brand logo %s", record.id)
