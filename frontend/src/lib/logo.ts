/**
 * Brand-mark geometry, pre-flight validation and the contrast maths for a logo.
 *
 * Every number here is the renderer's own, read out of the backend rather than
 * guessed — see the citations on each constant. That matters because the whole
 * point of the uploader is telling someone the truth about a 49-pixel mark
 * *before* they spend three minutes rendering it.
 *
 * The WCAG arithmetic is imported from `lib/contrast.ts`, never re-implemented:
 * that module is the one covered by `pnpm run test:contrast`.
 */

import {
  contrastRatio,
  parseHex,
  WCAG_LARGE_OBJECT,
  type Palette,
  type Rgb,
} from '@/lib/contrast'

/* ------------------------------------------------------------------ *
 * Geometry — `Theme` in `backend/app/core/models.py`, and
 * `logo_height` / `logo_rect` in `backend/app/render/text_overlay.py`.
 * ------------------------------------------------------------------ */

/** `Theme.logo_height_fraction`. A fraction of the frame **height**. */
export const LOGO_HEIGHT_FRACTION = 0.045

/**
 * `Theme.logo_margin_fraction`. A fraction of the frame **width** on *both*
 * axes, so the corner reads as square rather than as a wider gap on one side.
 */
export const LOGO_MARGIN_FRACTION = 0.028

/** `Theme.logo_opacity` — the default, and what every dark preset uses. */
export const LOGO_OPACITY_DARK = 0.85

/**
 * What the four light presets override `logo_opacity` to (`app/core/themes.py`).
 *
 * `list_themes()` does not currently serialise it, so it is derived from
 * `is_light` and overridden by an explicit `logo_opacity` if the catalogue ever
 * starts sending one.
 */
export const LOGO_OPACITY_LIGHT = 0.92

/** The reference frame the renderer's 1080p profile produces. */
export const REFERENCE_FRAME_HEIGHT = 1080

/**
 * `logo_height(profile, theme)` at 1080p: `round(1080 * 0.045)` = 49px.
 *
 * This is the number the UI has to make visceral. A mark that looks sharp in a
 * 400px preview is 49 pixels tall in the delivered video.
 */
export const LOGO_RENDER_HEIGHT = Math.max(
  1,
  Math.round(REFERENCE_FRAME_HEIGHT * LOGO_HEIGHT_FRACTION),
)

export function logoHeightPx(frameHeight: number = REFERENCE_FRAME_HEIGHT): number {
  return Math.max(1, Math.round(frameHeight * LOGO_HEIGHT_FRACTION))
}

/**
 * The opacity this theme composites the mark at.
 *
 * @param theme anything carrying `is_light`, plus an optional `logo_opacity`
 *   from the server. The server's value wins when present — it is what the
 *   render actually uses.
 */
export function logoOpacityFor(
  theme: { is_light: boolean; logo_opacity?: number | null } | null,
): number {
  const reported = theme?.logo_opacity
  if (typeof reported === 'number' && Number.isFinite(reported) && reported > 0) {
    return Math.min(1, reported)
  }
  return theme?.is_light === true ? LOGO_OPACITY_LIGHT : LOGO_OPACITY_DARK
}

/* ------------------------------------------------------------------ *
 * Upload limits — `Settings` in `backend/app/core/config.py`.
 * ------------------------------------------------------------------ */

/** `video_logo_max_bytes`. */
export const MAX_LOGO_BYTES = 4 * 1024 * 1024

/** `video_logo_max_dimension`. */
export const MAX_LOGO_DIMENSION = 4096

/**
 * `LOGO_MIN_ALPHA_COVERAGE` in `app/render/text_overlay.py`: the **mean** alpha
 * under which `rasterise_logo` decides the rasteriser gave up and skips
 * branding entirely. Checked client-side so a near-empty PNG is caught here
 * rather than producing a video with no mark and no explanation.
 */
export const MIN_MEAN_ALPHA = 0.02

/**
 * Formats we offer. PNG is first because it is the format that renders
 * faithfully: `rasterise_logo` hands an SVG to ImageMagick, and with no
 * `rsvg-convert` on the box it falls back to `flatten_svg_paths` — top-level
 * `<path>` elements only, masks and filters dropped.
 */
export const ACCEPTED_LOGO_MIME = ['image/png', 'image/svg+xml'] as const

/** Some browsers report `''` for an SVG dragged from the desktop. */
export const ACCEPTED_LOGO_EXTENSIONS = ['.png', '.svg'] as const

/** `accept` for the file input — MIME types plus extensions, for Safari. */
export const LOGO_ACCEPT_ATTRIBUTE = [
  ...ACCEPTED_LOGO_MIME,
  ...ACCEPTED_LOGO_EXTENSIONS,
].join(',')

export function isSvgFile(file: File): boolean {
  return file.type === 'image/svg+xml' || file.name.toLowerCase().endsWith('.svg')
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot).toLowerCase()
}

/** Bytes as a short human string. Kept local: `lib/format.ts` is time-only. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${String(bytes)} B`
  const kib = bytes / 1024
  if (kib < 1024) return `${kib < 10 ? kib.toFixed(1) : String(Math.round(kib))} KB`
  const mib = kib / 1024
  return `${mib < 10 ? mib.toFixed(1) : String(Math.round(mib))} MB`
}

/* ------------------------------------------------------------------ *
 * Contrast of a mark against a theme
 * ------------------------------------------------------------------ */

function toHex(rgb: Rgb): string {
  const pair = (value: number): string =>
    Math.max(0, Math.min(255, Math.round(value)))
      .toString(16)
      .padStart(2, '0')
      .toUpperCase()
  return `#${pair(rgb.r)}${pair(rgb.g)}${pair(rgb.b)}`
}

/**
 * `fg` composited over `bg` at `alpha`, as a hex colour.
 *
 * This step is not cosmetic. The mark is drawn at 85% (92% on light presets),
 * so the pixels a viewer actually sees are the blend, and *that* is what has to
 * be measured. The product's own violet `#863BFF` is 4.84:1 against Daylight's
 * background as a raw colour but 3.87:1 once the 85% composite is applied —
 * measuring the raw colour would overstate the mark by a full point.
 */
export function blendOver(fg: string, bg: string, alpha: number): string | null {
  const f = parseHex(fg)
  const b = parseHex(bg)
  if (f === null || b === null) return null
  const a = Math.max(0, Math.min(1, alpha))
  return toHex({
    r: a * f.r + (1 - a) * b.r,
    g: a * f.g + (1 - a) * b.g,
    b: a * f.b + (1 - a) * b.b,
  })
}

export interface LogoContrast {
  /** The mark's dominant colour as uploaded. */
  colour: string
  /** That colour composited over the background at the render opacity. */
  composite: string
  /** WCAG ratio of the composite against the background, or `null`. */
  ratio: number | null
  /** Clears SC 1.4.11 non-text contrast, 3:1. A logo is a graphic, not text. */
  passes: boolean
  opacity: number
}

/**
 * How a mark of colour `colour` will read on `palette`'s background.
 *
 * Gated at `WCAG_LARGE_OBJECT` (3:1), not 4.5: a logo is a non-text graphic.
 * Nothing here blocks anything — a brand colour is not ours to refuse — but a
 * mark that vanishes into the ground is worth one sentence before the render.
 */
export function evaluateLogoContrast(
  colour: string,
  palette: Palette,
  opacity: number,
): LogoContrast | null {
  const composite = blendOver(colour, palette.bg, opacity)
  if (composite === null) return null
  const ratio = contrastRatio(composite, palette.bg)
  return {
    colour,
    composite,
    ratio,
    passes: ratio !== null && ratio >= WCAG_LARGE_OBJECT,
    opacity,
  }
}

/**
 * The built-in mark's ink colour — the single `fill` in
 * `frontend/public/favicon.svg`, which is `DEFAULT_LOGO_RELATIVE` in
 * `app/render/ffmpeg_backend.py` and therefore the actual default source.
 */
export const BUILT_IN_MARK_COLOUR = '#863BFF'

/* ------------------------------------------------------------------ *
 * Pre-flight inspection
 *
 * Everything below runs in the browser before a byte is uploaded, because a
 * 4MiB round trip to learn "that is 6000px wide" is a worse experience than
 * decoding it locally and saying so instantly.
 * ------------------------------------------------------------------ */

/** A problem that stops the upload. */
export interface LogoProblem {
  code: string
  message: string
}

export interface LogoInspection {
  file: File
  /** Object URL for previewing. The caller owns revoking it. */
  objectUrl: string
  isSvg: boolean
  width: number | null
  height: number | null
  /** `null` when it could not be measured (an SVG we could not sample). */
  hasAlpha: boolean | null
  /** Mean alpha 0..1, or `null`. Compared against `MIN_MEAN_ALPHA`. */
  meanAlpha: number | null
  /** Most common opaque colour, `#RRGGBB`, or `null`. */
  dominant: string | null
  /** Blocking: the upload must not be attempted. */
  errors: LogoProblem[]
  /** Non-blocking, but said out loud before the render. */
  warnings: LogoProblem[]
}

export function isUploadable(inspection: LogoInspection): boolean {
  return inspection.errors.length === 0
}

/**
 * SVG constructs ImageMagick's built-in renderer does not implement.
 *
 * `_logo_source_argv` only reaches `flatten_svg_paths` — which keeps top-level
 * `<path>` elements and drops everything else — when there is no
 * `rsvg-convert` on PATH, which is the case on the render box. So a mark whose
 * ink lives inside a `<mask>` or `<filter>` either loses that ink or comes out
 * as a black blob. Detected by reading the markup, which is cheap and exact
 * enough: these are element names, not styling guesses.
 */
const SVG_RISKY_CONSTRUCTS: readonly { pattern: RegExp; what: string }[] = [
  { pattern: /<mask[\s>]/i, what: '<mask>' },
  { pattern: /<filter[\s>]/i, what: '<filter>' },
  { pattern: /<clipPath[\s>]/i, what: '<clipPath>' },
  { pattern: /<image[\s>]/i, what: 'embedded <image>' },
  { pattern: /<text[\s>]/i, what: '<text>' },
]

/** Whether the markup has a `<path>` outside any group — what survives flattening. */
const SVG_HAS_PATH = /<path[\s>]/i

export function scanSvgMarkup(source: string): LogoProblem[] {
  const found = SVG_RISKY_CONSTRUCTS.filter((entry) => entry.pattern.test(source)).map(
    (entry) => entry.what,
  )
  const problems: LogoProblem[] = []
  if (found.length > 0) {
    problems.push({
      code: 'svg_unsupported_construct',
      message: `This SVG uses ${found.join(', ')}. The render box has no rsvg-convert, so ImageMagick keeps only the top-level <path> shapes — anything drawn through those constructs is dropped or comes out as a black blob. Export a PNG with transparency instead.`,
    })
  }
  if (!SVG_HAS_PATH.test(source)) {
    problems.push({
      code: 'svg_no_path',
      message:
        'This SVG has no <path> elements, so there is nothing for the fallback rasteriser to keep. Export a PNG with transparency instead.',
    })
  }
  return problems
}

/** Decodes to an `HTMLImageElement`, or `null` if the browser refuses it. */
async function decodeImage(url: string): Promise<HTMLImageElement | null> {
  const image = new Image()
  image.decoding = 'sync'
  image.src = url
  try {
    await image.decode()
  } catch {
    return null
  }
  return image
}

interface PixelStats {
  hasAlpha: boolean
  meanAlpha: number
  dominant: string | null
}

/**
 * Samples the decoded bitmap for alpha and a dominant colour.
 *
 * Colours are bucketed 4 bits per channel and the winning bucket is averaged,
 * so antialiasing and gradients collapse onto one representative ink colour
 * rather than fragmenting the histogram. Pixels under 50% alpha are ignored:
 * a soft edge is not the brand colour.
 *
 * Returns `null` when the canvas cannot be read — an SVG referencing something
 * external taints it, and a tainted canvas is a security error, not a logo
 * problem, so it degrades to "unknown" rather than to a warning.
 */
function samplePixels(image: HTMLImageElement, width: number, height: number): PixelStats | null {
  const SAMPLE = 96
  const scale = Math.min(1, SAMPLE / Math.max(width, height))
  const w = Math.max(1, Math.round(width * scale))
  const h = Math.max(1, Math.round(height * scale))

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (context === null) return null
  context.clearRect(0, 0, w, h)
  try {
    context.drawImage(image, 0, 0, w, h)
  } catch {
    return null
  }

  let data: Uint8ClampedArray
  try {
    data = context.getImageData(0, 0, w, h).data
  } catch {
    return null
  }

  const buckets = new Map<number, { count: number; r: number; g: number; b: number }>()
  let alphaSum = 0
  let transparent = 0

  for (let index = 0; index + 3 < data.length; index += 4) {
    const r = data[index] ?? 0
    const g = data[index + 1] ?? 0
    const b = data[index + 2] ?? 0
    const a = data[index + 3] ?? 0
    alphaSum += a
    if (a < 250) transparent += 1
    if (a < 128) continue
    const key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
    const bucket = buckets.get(key)
    if (bucket === undefined) buckets.set(key, { count: 1, r, g, b })
    else {
      bucket.count += 1
      bucket.r += r
      bucket.g += g
      bucket.b += b
    }
  }

  const pixels = (w * h) || 1
  let best: { count: number; r: number; g: number; b: number } | null = null
  for (const bucket of buckets.values()) {
    if (best === null || bucket.count > best.count) best = bucket
  }

  return {
    hasAlpha: transparent > 0,
    meanAlpha: alphaSum / (pixels * 255),
    dominant:
      best === null
        ? null
        : toHex({ r: best.r / best.count, g: best.g / best.count, b: best.b / best.count }),
  }
}

/**
 * Validates and measures a candidate logo without uploading it.
 *
 * Order matters: type and size are checked first because they are free, and a
 * 40MB file should not be handed to the decoder at all.
 */
export async function inspectLogoFile(file: File): Promise<LogoInspection> {
  const svg = isSvgFile(file)
  const errors: LogoProblem[] = []
  const warnings: LogoProblem[] = []

  const mimeOk = (ACCEPTED_LOGO_MIME as readonly string[]).includes(file.type)
  const extOk = (ACCEPTED_LOGO_EXTENSIONS as readonly string[]).includes(extensionOf(file.name))
  if (!mimeOk && !extOk) {
    errors.push({
      code: 'type',
      message: `${file.type === '' ? extensionOf(file.name) || 'That file' : file.type} is not a logo format. Upload a PNG with transparency (preferred) or an SVG.`,
    })
  }
  if (file.size === 0) {
    errors.push({ code: 'empty', message: 'That file is empty.' })
  }
  if (file.size > MAX_LOGO_BYTES) {
    errors.push({
      code: 'size',
      message: `That file is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_LOGO_BYTES)} — the mark is only ${String(LOGO_RENDER_HEIGHT)}px tall in the video, so a large source buys nothing.`,
    })
  }

  const objectUrl = URL.createObjectURL(file)
  const inspection: LogoInspection = {
    file,
    objectUrl,
    isSvg: svg,
    width: null,
    height: null,
    hasAlpha: null,
    meanAlpha: null,
    dominant: null,
    errors,
    warnings,
  }

  // A file we have already rejected is not decoded: the errors above are enough
  // to act on, and decoding a 40MB bitmap to add nothing is wasteful.
  if (errors.length > 0) return inspection

  if (svg) {
    try {
      warnings.push(...scanSvgMarkup(await file.text()))
    } catch {
      // Unreadable text on a file the browser just handed us is not worth a
      // message of its own; the generic SVG caveat still applies.
    }
    warnings.push({
      code: 'svg_rasterised',
      message:
        'SVGs are rasterised server-side at upload. A PNG with transparency is the safer choice: it is exactly the pixels you exported.',
    })
  }

  const image = await decodeImage(objectUrl)
  if (image === null) {
    errors.push({
      code: 'decode',
      message: 'That file could not be decoded as an image. It may be corrupt or mislabelled.',
    })
    return inspection
  }

  // An SVG with no intrinsic size reports 0; `naturalWidth` is the rasterised
  // size for a PNG and the viewBox size for an SVG that declares one.
  const width = image.naturalWidth > 0 ? image.naturalWidth : null
  const height = image.naturalHeight > 0 ? image.naturalHeight : null
  inspection.width = width
  inspection.height = height

  if (width !== null && height !== null) {
    if (width > MAX_LOGO_DIMENSION || height > MAX_LOGO_DIMENSION) {
      errors.push({
        code: 'dimension',
        message: `That image is ${String(width)}x${String(height)}. The limit is ${String(MAX_LOGO_DIMENSION)}px on either side.`,
      })
    }
    if (height < LOGO_RENDER_HEIGHT) {
      warnings.push({
        code: 'small',
        message: `Only ${String(height)}px tall. The mark is scaled to ${String(LOGO_RENDER_HEIGHT)}px at 1080p, so this one is upscaled and will look soft.`,
      })
    }
    // The renderer reserves a box of aspect 1.6 for its collision check
    // (`LOGO_RESERVED_ASPECT`); a much wider mark reaches further across the
    // frame than that check assumes.
    const aspect = width / height
    if (aspect > 6) {
      warnings.push({
        code: 'wide',
        message: `This mark is ${aspect.toFixed(1)}x wider than it is tall, so at ${String(LOGO_RENDER_HEIGHT)}px tall it stretches ${String(Math.round(LOGO_RENDER_HEIGHT * aspect))}px across the bottom of the frame. A square or stacked lockup survives the size better than a wide wordmark.`,
      })
    }
  }

  const stats = samplePixels(image, width ?? 512, height ?? 512)
  if (stats !== null) {
    inspection.hasAlpha = stats.hasAlpha
    inspection.meanAlpha = stats.meanAlpha
    inspection.dominant = stats.dominant

    if (!svg && !stats.hasAlpha) {
      warnings.push({
        code: 'opaque',
        message:
          'No transparency detected. This will composite as a solid rectangle over the bottom-left of every frame. Export a PNG with an alpha channel.',
      })
    }
    if (stats.meanAlpha !== null && stats.meanAlpha < MIN_MEAN_ALPHA) {
      warnings.push({
        code: 'blank',
        message: `Almost every pixel is transparent (mean alpha ${stats.meanAlpha.toFixed(3)}). The renderer treats anything under ${String(MIN_MEAN_ALPHA)} as a failed rasterisation and skips branding entirely, so this video would ship with no mark at all.`,
      })
    }
  }

  return inspection
}
