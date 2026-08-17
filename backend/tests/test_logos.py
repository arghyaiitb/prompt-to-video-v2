"""Brand-logo uploads: the store, the validation, and the wiring into a job.

Every hostile input here is a real one — a PNG that is not a PNG, an SVG carrying script,
an external entity, a 40-byte file claiming to be 99999px square, a filename trying to
climb out of the store. The fixtures generate their own PNG with ImageMagick rather than
committing a binary: a checked-in blob nobody can read is not a test input, it is a
mystery, and the SVG case uses the repo's own favicon, which is the exact file this
feature exists to replace.

No network, no ffmpeg, no render.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import logos
from app.core.config import REPO_ROOT, get_settings
from app.db.models import NO_LOGO_ID, Job
from app.db.session import init_db, make_engine, set_engine
from app.main import app
from app.worker import factory, pipeline

FAVICON = REPO_ROOT / "frontend" / "public" / "favicon.svg"

SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="46" viewBox="0 0 48 46">'
SQUARE = '<path fill="#863bff" d="M4 4h40v38H4z"/>'


def _imagemagick() -> str | None:
    return shutil.which("magick") or shutil.which("convert")


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    """A real 240x120 RGBA PNG, generated here rather than committed."""
    binary = _imagemagick()
    if binary is None:  # pragma: no cover - environment-dependent
        pytest.skip("ImageMagick is not installed")
    with tempfile.TemporaryDirectory() as raw_dir:
        out = Path(raw_dir) / "logo.png"
        subprocess.run(
            [
                binary, "-size", "240x120", "xc:none",
                "-fill", "#863bff", "-draw", "roundrectangle 12,12 228,108 20,20",
                f"png32:{out}",
            ],
            check=True,
            capture_output=True,
        )
        return out.read_bytes()


@pytest.fixture(scope="session")
def opaque_png_bytes() -> bytes:
    """A PNG with no alpha channel — valid, but a watermark with a visible box."""
    binary = _imagemagick()
    if binary is None:  # pragma: no cover - environment-dependent
        pytest.skip("ImageMagick is not installed")
    with tempfile.TemporaryDirectory() as raw_dir:
        out = Path(raw_dir) / "flat.png"
        subprocess.run(
            [binary, "-size", "64x64", "xc:#863bff", f"png24:{out}"],
            check=True, capture_output=True,
        )
        return out.read_bytes()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """App on a temp DB and a temp cache dir, so the logo store is per-test."""
    set_engine(make_engine(tmp_path / "test.db"))
    init_db()

    settings = get_settings()
    monkeypatch.setattr(settings, "video_output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "video_cache_dir", tmp_path / "cache")

    async def _fake_run_job(job_id: str) -> None:
        return None

    monkeypatch.setattr(pipeline, "run_job", _fake_run_job)

    with TestClient(app) as test_client:
        yield test_client

    set_engine(None)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The logo store on its own, for tests that need no HTTP."""
    monkeypatch.setattr(get_settings(), "video_cache_dir", tmp_path / "cache")
    return logos.logo_dir()


def _upload(client: TestClient, name: str, data: bytes, content_type: str = "image/png"):
    return client.post("/api/logos", files={"file": (name, data, content_type)})


def _error(response) -> str:  # noqa: ANN001 - httpx.Response
    detail = response.json()["detail"]
    return detail["error"] if isinstance(detail, dict) else str(detail)


# ------------------------------------------------------------------- happy paths


def test_png_upload_returns_the_full_contract(client: TestClient, png_bytes: bytes) -> None:
    response = _upload(client, "brand.png", png_bytes)
    assert response.status_code == 201, response.text
    body = response.json()

    assert logos.LOGO_ID_RE.match(body["id"])
    assert body["url"] == f"/api/logos/{body['id']}"
    assert body["render_url"] == f"/api/logos/{body['id']}/render"
    assert body["format"] == "png"
    assert (body["width"], body["height"]) == (240, 120)
    assert body["has_alpha"] is True
    assert body["size_bytes"] == len(png_bytes)
    assert body["filename"] == "brand.png"
    assert body["warnings"] == []
    assert body["uploaded_at"]

    # the served bytes are the bytes that were uploaded, and the render asset is the same
    # file for a PNG — nothing is re-encoded behind the caller's back
    fetched = client.get(body["url"])
    assert fetched.status_code == 200
    assert fetched.content == png_bytes
    assert fetched.headers["content-type"] == "image/png"
    assert fetched.headers["x-content-type-options"] == "nosniff"
    assert client.get(body["render_url"]).content == png_bytes


def test_ids_are_content_addressed(client: TestClient, png_bytes: bytes) -> None:
    """The same bytes twice is one logo, not two — and never a second copy on disk."""
    first = _upload(client, "a.png", png_bytes).json()
    second = _upload(client, "b-renamed.png", png_bytes).json()
    assert first["id"] == second["id"]
    assert len(client.get("/api/logos").json()) == 1


def test_a_png_without_alpha_is_accepted_with_a_warning(
    client: TestClient, opaque_png_bytes: bytes
) -> None:
    body = _upload(client, "flat.png", opaque_png_bytes).json()
    assert body["has_alpha"] is False
    assert any("alpha" in w for w in body["warnings"])


def test_svg_upload_rasterises_and_warns_about_what_it_cannot_render(
    client: TestClient,
) -> None:
    """The repo's own favicon: a masked group of blurred ellipses over a flat path.

    ImageMagick here has no `rsvg-convert` delegate, so `<mask>`/`<filter>` are ignored and
    the mark rasterises wrong. That is a warning at upload rather than a black blob
    discovered in a finished video.
    """
    if not FAVICON.is_file():  # pragma: no cover - repo layout
        pytest.skip("frontend/public/favicon.svg is missing")
    response = _upload(client, "favicon.svg", FAVICON.read_bytes(), "image/svg+xml")
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["format"] == "svg"
    assert (body["width"], body["height"]) == (48, 46)
    assert body["has_alpha"] is True
    assert any("<mask>" in w for w in body["warnings"]), body["warnings"]

    # the original is served as SVG; the render asset is a real PNG of the raster height
    assert client.get(body["url"]).headers["content-type"] == "image/svg+xml"
    render = client.get(body["render_url"])
    assert render.status_code == 200
    assert render.content.startswith(logos.PNG_MAGIC)

    stored = logos.logo_render_path(body["id"])
    assert stored is not None and stored.suffix == ".png"
    info = logos.read_png_info(stored)
    assert info.height == logos.RASTER_HEIGHT
    assert info.has_alpha is True


def test_a_clean_svg_uploads_without_warnings(client: TestClient) -> None:
    svg = f"{SVG_HEAD}{SQUARE}</svg>".encode()
    body = _upload(client, "clean.svg", svg, "image/svg+xml").json()
    assert body["warnings"] == []
    assert body["format"] == "svg"


# --------------------------------------------------------------------- rejections


def test_an_oversize_upload_is_rejected_and_leaves_nothing_behind(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, png_bytes: bytes
) -> None:
    monkeypatch.setattr(get_settings(), "video_logo_max_bytes", 512)
    response = _upload(client, "huge.png", png_bytes + b"\x00" * 200_000)
    assert response.status_code == 413
    assert _error(response) == "logo_too_large"
    assert response.json()["detail"]["max_bytes"] == 512
    # no spool file, no source, no sidecar
    assert sorted(p.name for p in logos.logo_dir().iterdir()) == []


async def test_the_size_cap_stops_reading_instead_of_buffering_the_body(store: Path) -> None:
    """The ceiling has to bind *while* the body arrives.

    Enforced after buffering, the cap only limits what is kept — the memory and the disk
    write have already happened. This asserts the reader stopped near the limit rather than
    consuming the whole 4 MiB payload.
    """

    class _CountingUpload:
        filename = "big.png"
        content_type = "image/png"

        def __init__(self, data: bytes) -> None:
            self.data = data
            self.position = 0
            self.served = 0

        async def read(self, size: int = -1) -> bytes:
            chunk = self.data[self.position : self.position + size]
            self.position += len(chunk)
            self.served += len(chunk)
            return chunk

    upload = _CountingUpload(b"\x89PNG\r\n\x1a\n" + b"\x00" * (4 * 1024 * 1024))
    with pytest.raises(logos.HTTPException) as raised:
        await logos._stream_to_temp(upload, 100_000)  # type: ignore[arg-type]

    assert raised.value.status_code == 413
    assert upload.served <= 100_000 + logos.CHUNK_BYTES
    assert upload.served < len(upload.data) / 4
    assert list(store.iterdir()) == []


def test_a_file_declared_png_that_is_not_a_png_is_rejected(client: TestClient) -> None:
    """The client's word for the content is worth nothing; the magic bytes decide."""
    response = _upload(client, "logo.png", b"GIF89a" + b"\x00" * 64, "image/png")
    assert response.status_code == 422
    assert _error(response) == "png_magic_mismatch"
    assert list(logos.logo_dir().iterdir()) == []


def test_png_magic_with_rubbish_behind_it_is_rejected(client: TestClient) -> None:
    response = _upload(client, "logo.png", logos.PNG_MAGIC + b"\xde\xad\xbe\xef" * 32)
    assert response.status_code == 422
    assert _error(response) == "png_malformed"


def test_a_png_that_decodes_to_an_enormous_bitmap_is_rejected(
    client: TestClient, png_bytes: bytes
) -> None:
    """A small file claiming 99999x99999. Caught from the header, never allocated."""
    bomb = bytearray(png_bytes)
    bomb[16:24] = (99999).to_bytes(4, "big") + (99999).to_bytes(4, "big")
    response = _upload(client, "bomb.png", bytes(bomb))
    assert response.status_code == 422
    assert _error(response) == "logo_dimensions"
    assert response.json()["detail"]["max_dimension"] == get_settings().video_logo_max_dimension


def test_an_svg_claiming_enormous_dimensions_is_rejected(client: TestClient) -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100000" height="100000">'
        f"{SQUARE}</svg>"
    ).encode()
    response = _upload(client, "bomb.svg", svg, "image/svg+xml")
    assert response.status_code == 422
    assert _error(response) == "logo_dimensions"


def test_an_svg_containing_script_is_rejected(client: TestClient) -> None:
    svg = f"{SVG_HEAD}<script>alert(1)</script>{SQUARE}</svg>".encode()
    response = _upload(client, "xss.svg", svg, "image/svg+xml")
    assert response.status_code == 422
    assert _error(response) == "svg_unsafe"
    assert response.json()["detail"]["construct"] == "script"


def test_an_svg_with_an_external_entity_is_rejected(client: TestClient) -> None:
    """XXE, refused on the raw bytes: no DTD ever reaches the XML parser."""
    svg = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
        f"{SVG_HEAD}<desc>&xxe;</desc>{SQUARE}</svg>"
    ).encode()
    response = _upload(client, "xxe.svg", svg, "image/svg+xml")
    assert response.status_code == 422
    assert _error(response) == "svg_unsafe"
    assert response.json()["detail"]["construct"] == "<!doctype"


def test_an_svg_with_a_billion_laughs_entity_is_rejected(client: TestClient) -> None:
    svg = (
        '<!DOCTYPE svg [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;">]>'
        f"{SVG_HEAD}<desc>&b;</desc>{SQUARE}</svg>"
    ).encode()
    assert _error(_upload(client, "lol.svg", svg, "image/svg+xml")) == "svg_unsafe"


def test_an_svg_referencing_something_off_origin_is_rejected(client: TestClient) -> None:
    svg = (
        f'{SVG_HEAD}<image href="https://evil.example/x.png" width="48" height="46"/></svg>'
    ).encode()
    response = _upload(client, "remote.svg", svg, "image/svg+xml")
    assert response.status_code == 422
    assert _error(response) == "svg_unsafe"


def test_an_svg_with_an_event_handler_is_rejected(client: TestClient) -> None:
    svg = f'{SVG_HEAD}<path onload="fetch(1)" d="M0 0h4v4H0z"/></svg>'.encode()
    assert _error(_upload(client, "onload.svg", svg, "image/svg+xml")) == "svg_unsafe"


def test_a_fragment_reference_inside_the_document_is_allowed(client: TestClient) -> None:
    """`#id` is same-document and cannot fetch anything, so it is not a threat."""
    svg = (
        f'{SVG_HEAD}<defs><path id="p" d="M4 4h40v38H4z"/></defs>'
        '<use href="#p" fill="#863bff"/></svg>'
    ).encode()
    response = _upload(client, "use.svg", svg, "image/svg+xml")
    assert response.status_code == 201, response.text
    # ...and it is still warned about, because MSVG does not implement <use>
    assert any("<use>" in w for w in response.json()["warnings"])


def test_an_svg_this_box_cannot_rasterise_at_all_is_rejected(client: TestClient) -> None:
    """Better a 422 now than a stored mark that renders as an error later.

    ImageMagick's built-in renderer rejects this construct outright rather than dropping it,
    so there is no PNG to store and nothing to warn about.
    """
    svg = (
        f'{SVG_HEAD}<defs><linearGradient id="g"><stop offset="0" stop-color="#863bff"/>'
        '</linearGradient></defs><use href="#g"/></svg>'
    ).encode()
    response = _upload(client, "broken.svg", svg, "image/svg+xml")
    assert response.status_code == 422
    assert _error(response) == "svg_rasterise_failed"
    # nothing half-stored: no source file left without a render asset behind it
    assert list(logos.logo_dir().iterdir()) == []


def test_neither_png_nor_xml_is_a_415(client: TestClient) -> None:
    response = _upload(client, "logo.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp")
    assert response.status_code == 415
    assert _error(response) == "unsupported_logo_format"


def test_an_empty_upload_is_rejected(client: TestClient) -> None:
    assert _error(_upload(client, "empty.png", b"")) == "logo_empty"


# ------------------------------------------------------------------ untrusted names


def test_a_traversal_filename_cannot_escape_the_store(
    client: TestClient, tmp_path: Path, png_bytes: bytes
) -> None:
    """The uploaded name never reaches the filesystem — the id is a content hash."""
    body = _upload(client, "../../../../etc/x.png", png_bytes).json()

    assert logos.LOGO_ID_RE.match(body["id"])
    assert body["filename"] == "x.png"  # sanitised for display only
    written = sorted(p.name for p in logos.logo_dir().iterdir())
    assert written == [f"{body['id']}.json", f"{body['id']}.png"]
    assert not (tmp_path / "etc").exists()
    assert logos.source_path(body["id"], "png").parent == logos.logo_dir()


def test_a_windows_separator_filename_is_sanitised(client: TestClient, png_bytes: bytes) -> None:
    body = _upload(client, r"..\..\windows\system32\evil.png", png_bytes).json()
    assert body["filename"] == "evil.png"


@pytest.mark.parametrize(
    "logo_id",
    [
        "../../../etc/passwd",
        "..%2F..%2Fvideos.db",
        "not-a-hash",
        "",
        "0" * 31,
        "0" * 33,
        "0" * 31 + "Z",
    ],
)
def test_a_bad_id_never_resolves_to_a_file(store: Path, logo_id: str) -> None:
    """The loader is the gate: no id that could not have been generated reaches the disk."""
    assert logos.load_logo(logo_id) is None
    assert logos.logo_exists(logo_id) is False
    assert logos.logo_render_path(logo_id) is None


@pytest.mark.parametrize(
    "logo_id", ["../../../etc/passwd", "not-a-hash", "0" * 31, "0" * 33]
)
def test_fetching_a_bad_id_is_a_404(client: TestClient, logo_id: str) -> None:
    assert client.get(f"/api/logos/{logo_id}").status_code == 404
    assert client.get(f"/api/logos/{logo_id}/render").status_code == 404
    assert client.delete(f"/api/logos/{logo_id}").status_code == 404


# -------------------------------------------------------------------- list/get/delete


def test_list_returns_newest_first_with_dimensions_and_upload_time(
    client: TestClient, png_bytes: bytes, opaque_png_bytes: bytes
) -> None:
    first = _upload(client, "one.png", png_bytes).json()
    second = _upload(client, "two.png", opaque_png_bytes).json()

    listed = client.get("/api/logos").json()
    assert [row["id"] for row in listed] == [second["id"], first["id"]]
    for row in listed:
        assert row["width"] > 0 and row["height"] > 0
        assert row["uploaded_at"]
        assert row["url"].endswith(row["id"])

    meta = client.get(f"/api/logos/{first['id']}/meta").json()
    assert meta == first


def test_delete_refuses_while_a_job_still_names_the_logo(
    client: TestClient, png_bytes: bytes
) -> None:
    """409, not a cascade: `Timeline.logo_path` points into the store, so deleting a mark a
    job was rendered with would silently change what a re-render produces."""
    logo = _upload(client, "brand.png", png_bytes).json()
    created = client.post(
        "/api/jobs",
        json={"topic": "Phishing", "slide_count": 2, "logo_id": logo["id"]},
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]

    refused = client.delete(f"/api/logos/{logo['id']}")
    assert refused.status_code == 409
    assert _error(refused) == "logo_in_use"
    assert refused.json()["detail"]["jobs"] == [job_id]
    assert client.get(logo["url"]).status_code == 200

    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.delete(f"/api/logos/{logo['id']}").status_code == 204
    assert client.get(logo["url"]).status_code == 404
    assert client.get("/api/logos").json() == []
    assert list(logos.logo_dir().iterdir()) == []


def test_deleting_an_unknown_logo_is_a_404(client: TestClient) -> None:
    assert client.delete(f"/api/logos/{'a' * 32}").status_code == 404


def test_a_sidecar_without_its_file_reads_as_missing(store: Path, png_bytes: bytes) -> None:
    """A half-deleted store must answer "no logo", not hand out a path that is not there."""
    logo_id = "b" * 32
    (store / f"{logo_id}.json").write_text(
        json.dumps(
            {
                "id": logo_id,
                "format": "png",
                "width": 10,
                "height": 10,
                "has_alpha": True,
                "size_bytes": 10,
                "uploaded_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    assert logos.load_logo(logo_id) is None
    assert logos.logo_exists(logo_id) is False
    assert logos.list_stored() == []


# ----------------------------------------------------------------------- job wiring


def test_a_job_can_name_an_uploaded_logo(client: TestClient, png_bytes: bytes) -> None:
    logo = _upload(client, "brand.png", png_bytes).json()
    created = client.post(
        "/api/jobs", json={"topic": "Badges", "slide_count": 2, "logo_id": logo["id"]}
    )
    assert created.status_code == 202
    body = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert body["logo_id"] == logo["id"]
    assert body["logo_url"] == f"/api/logos/{logo['id']}"


def test_an_unknown_logo_id_is_a_422_naming_it(client: TestClient) -> None:
    response = client.post(
        "/api/jobs", json={"topic": "Badges", "slide_count": 2, "logo_id": "f" * 32}
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "unknown_logo"
    assert detail["logo_id"] == "f" * 32
    assert "f" * 32 in detail["message"]


def test_no_logo_id_leaves_the_default_untouched(client: TestClient) -> None:
    """The whole compatibility contract: say nothing, get exactly today's behaviour."""
    created = client.post("/api/jobs", json={"topic": "Badges", "slide_count": 2})
    body = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert body["logo_id"] is None
    assert body["logo_url"] is None
    assert pipeline.resolve_job_logo(Job(topic="t", slide_count=2, voice="v")) is None


def test_the_none_sentinel_means_no_branding(client: TestClient) -> None:
    created = client.post(
        "/api/jobs", json={"topic": "Badges", "slide_count": 2, "logo_id": "NONE"}
    )
    assert created.status_code == 202
    body = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert body["logo_id"] == NO_LOGO_ID
    # Nothing to fetch, so a picker never has to special-case the sentinel as a URL.
    assert body["logo_url"] is None


# --------------------------------------------------------------- timeline stamping


class _OneSceneProvider:
    def generate(self, topic: str, slide_count: int, **_: object):  # noqa: ANN201
        from app.core.models import SceneScript, Script

        return Script(
            topic=topic,
            title="T",
            scenes=[
                SceneScript(id=1, narration="n", heading="H", bullets=["A"], image_prompt="p")
            ],
        )


async def test_stage_script_stamps_the_jobs_logo_onto_the_timeline(
    client: TestClient, tmp_path: Path, png_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Timeline has to be self-describing: a re-render reproduces the branding."""
    monkeypatch.setattr(factory, "script_provider", lambda *a, **k: _OneSceneProvider())
    logo = _upload(client, "brand.png", png_bytes).json()

    timeline = await pipeline._stage_script(
        Job(topic="t", slide_count=1, voice="v", logo_id=logo["id"]), tmp_path
    )
    assert timeline.logo_path == str(logos.logo_render_path(logo["id"]))
    assert Path(timeline.logo_path).is_file()


@pytest.mark.parametrize(
    ("logo_id", "expected"),
    [(None, None), ("", None), (NO_LOGO_ID, NO_LOGO_ID), ("c" * 32, None)],
)
async def test_stage_script_logo_states(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logo_id: str | None,
    expected: str | None,
) -> None:
    """None -> default, "none" -> no branding, a vanished id -> default with a warning."""
    monkeypatch.setattr(factory, "script_provider", lambda *a, **k: _OneSceneProvider())
    timeline = await pipeline._stage_script(
        Job(topic="t", slide_count=1, voice="v", logo_id=logo_id), tmp_path
    )
    assert timeline.logo_path == expected


def test_the_render_backend_is_pointed_at_the_jobs_mark(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, png_bytes: bytes
) -> None:
    """What `ffmpeg_backend` needs from us: a resolved `logo_source` before it renders.

    `resolve_logo_source` is the renderer's own three-state resolver, so this asserts
    against the real thing: an upload path survives, `"none"` becomes no branding, and an
    unset `logo_path` still lands on the bundled favicon.
    """
    from app.core.models import Timeline
    from app.render.ffmpeg_backend import AUTO_LOGO, resolve_logo_source

    class _FakeBackend:
        def __init__(self, *, theme=None, logo_path=AUTO_LOGO) -> None:  # noqa: ANN001
            self.theme = theme
            self.logo_source = resolve_logo_source(logo_path)

    monkeypatch.setattr(factory, "_load", lambda *a, **k: _FakeBackend)
    logo = _upload(client, "brand.png", png_bytes).json()
    render_asset = logos.logo_render_path(logo["id"])

    def _source(logo_path: str | None):  # noqa: ANN202
        timeline = Timeline(
            job_id="j", topic="t", title="t", scenes=[], voice="v", logo_path=logo_path
        )
        return pipeline._video_backend(timeline, None).logo_source

    assert _source(str(render_asset)) == render_asset
    assert _source(NO_LOGO_ID) is None
    assert _source(None) == resolve_logo_source(AUTO_LOGO)
