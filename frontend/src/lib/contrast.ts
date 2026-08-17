/**
 * WCAG contrast, mirrored from the backend.
 *
 * The renderer burns text into pixels, so an unreadable palette cannot be
 * fixed after the fact — it has to be caught before the job is created. The
 * backend gate (`app/core/themes.py: validate_theme`) is the authority; this
 * module is a faithful client-side copy of it so the picker can refuse a bad
 * palette instantly instead of round-tripping for a 422.
 *
 * Verified against the backend implementation (`Theme._luminance` /
 * `Theme.contrast` in `app/core/models.py`) for the default `midnight`
 * palette: text 17.89, muted 7.30, accent 9.17 — see `contrast.test.ts`.
 *
 * Everything here is pure and dependency-free so it can be run under plain
 * `node` by the unit tests.
 */

export const PALETTE_KEYS = ['bg', 'surface', 'text', 'muted', 'accent'] as const

export type PaletteKey = (typeof PALETTE_KEYS)[number]

/** The five colours that make up a brand palette. Always `#RRGGBB`. */
export type Palette = Record<PaletteKey, string>

export interface Rgb {
  r: number
  g: number
  b: number
}

/** Copy for the palette editor — what each colour actually paints on a slide. */
export const PALETTE_LABELS: Record<PaletteKey, { label: string; hint: string }> = {
  bg: { label: 'Background', hint: 'The solid slide colour behind everything.' },
  surface: { label: 'Surface', hint: 'Panel and card fill, lifted slightly off the background.' },
  text: { label: 'Text', hint: 'Headings and bullets. Every word on screen uses this one colour.' },
  muted: { label: 'Muted', hint: 'Secondary tier — kickers, dates, source lines.' },
  accent: { label: 'Accent', hint: 'Heading rule and bullet markers. Graphics only, never text.' },
}

/* ------------------------------------------------------------------ *
 * Hex parsing
 * ------------------------------------------------------------------ */

/** The backend slices fixed pairs out of the string, so `#fff` is not valid there. */
const SIX_DIGIT = /^#[0-9A-Fa-f]{6}$/
const THREE_DIGIT = /^[0-9A-Fa-f]{3}$/
const BARE_SIX = /^[0-9A-Fa-f]{6}$/

/** True only for the canonical `#RRGGBB` form the backend accepts. */
export function isHexColour(value: string): boolean {
  return SIX_DIGIT.test(value.trim())
}

/**
 * Normalises loose user input to `#RRGGBB` (uppercase), or `null` if it is not
 * a colour at all.
 *
 * Shorthand is expanded rather than rejected: someone pasting `#0f8` from a
 * brand sheet means `#00FF88`, and the backend's six-digit-only rule is about
 * its own string slicing, not about what a human is allowed to type.
 */
export function normalizeHex(input: string): string | null {
  const raw = input.trim().replace(/^#/, '')
  if (THREE_DIGIT.test(raw)) {
    return `#${raw.charAt(0).repeat(2)}${raw.charAt(1).repeat(2)}${raw.charAt(2).repeat(2)}`.toUpperCase()
  }
  if (BARE_SIX.test(raw)) return `#${raw.toUpperCase()}`
  return null
}

export function parseHex(input: string): Rgb | null {
  const hex = normalizeHex(input)
  if (hex === null) return null
  return {
    r: Number.parseInt(hex.slice(1, 3), 16),
    g: Number.parseInt(hex.slice(3, 5), 16),
    b: Number.parseInt(hex.slice(5, 7), 16),
  }
}

function toHex(rgb: Rgb): string {
  const pair = (value: number): string =>
    Math.max(0, Math.min(255, value)).toString(16).padStart(2, '0').toUpperCase()
  return `#${pair(rgb.r)}${pair(rgb.g)}${pair(rgb.b)}`
}

/* ------------------------------------------------------------------ *
 * WCAG maths
 * ------------------------------------------------------------------ */

/** sRGB channel (0-255) to its linear-light value. */
function linearise(channel: number): number {
  const c = channel / 255
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

/** WCAG relative luminance, 0 (black) to 1 (white). */
export function relativeLuminance(rgb: Rgb): number {
  return (
    0.2126 * linearise(rgb.r) + 0.7152 * linearise(rgb.g) + 0.0722 * linearise(rgb.b)
  )
}

/** Luminance of a hex colour, or `null` when it cannot be parsed. */
export function luminanceOf(colour: string): number | null {
  const rgb = parseHex(colour)
  return rgb === null ? null : relativeLuminance(rgb)
}

/**
 * WCAG contrast ratio between two colours, 1 to 21. Order does not matter.
 * `null` when either colour is unparseable — a ratio against a colour we
 * cannot read is not a number worth showing.
 */
export function contrastRatio(a: string, b: string): number | null {
  const la = luminanceOf(a)
  const lb = luminanceOf(b)
  if (la === null || lb === null) return null
  const hi = Math.max(la, lb)
  const lo = Math.min(la, lb)
  return (hi + 0.05) / (lo + 0.05)
}

/**
 * True for light palettes. Mirrors `Theme.is_light`, which drives scrim colour
 * on the render side and light/dark grouping in the picker.
 */
export function isLightBackground(bg: string): boolean {
  const luminance = luminanceOf(bg)
  return luminance !== null && luminance > 0.5
}

/** Two decimals, matching the backend's `f"{ratio:.2f}"` in its failure text. */
export function formatRatio(ratio: number | null): string {
  return ratio === null ? '—' : ratio.toFixed(2)
}

/* ------------------------------------------------------------------ *
 * Thresholds — mirrors `THRESHOLDS` in `app/core/themes.py`
 * ------------------------------------------------------------------ */

/** WCAG AA for body text. This is the hard floor for a user's own palette. */
export const WCAG_AA_TEXT = 4.5

/**
 * WCAG AAA. Recommended for `text`, not required: delivered contrast is always
 * below authored contrast once a frame has been through h.264 chroma
 * subsampling and a projector in a lit room, so 7.0 buys the headroom that
 * survives it.
 */
export const WCAG_AAA_TEXT = 7.0

/** SC 1.4.11 non-text contrast / SC 1.4.3 large text. `accent` is a graphic. */
export const WCAG_LARGE_OBJECT = 3.0

export type ContrastPairId = 'text_on_bg' | 'text_on_surface' | 'muted_on_bg' | 'accent_on_bg'

/**
 * The hard floor for a *user-supplied* palette: WCAG AA.
 *
 * Mirrors `REQUIRED_THRESHOLDS` in `app/core/themes.py`. The stricter
 * `RECOMMENDED_THRESHOLDS` below is the bar the shipped presets are held to —
 * applying it to customer brand colours rejects legitimate palettes. Slate
 * `#64748B` on white is 4.76:1, perfectly accessible and an utterly ordinary
 * corporate grey, yet it fails a 7.0 gate. Refusing to render someone's actual
 * brand is worse than rendering it at AA, so this blocks below AA and warns
 * between AA and AAA.
 */
export const REQUIRED_THRESHOLDS: Record<ContrastPairId, number> = {
  text_on_bg: WCAG_AA_TEXT,
  text_on_surface: WCAG_AA_TEXT,
  muted_on_bg: WCAG_AA_TEXT,
  accent_on_bg: WCAG_LARGE_OBJECT,
}

/**
 * What we recommend, and what the shipped presets are held to. Mirrors
 * `THRESHOLDS` in `app/core/themes.py`.
 *
 * `muted` sits at AA rather than AAA on purpose: it is a secondary tier and
 * pushing it to 7.0 collapses the tonal gap that makes it read as secondary at
 * all. `accent` is a graphic, so 3.0 is the correct bar rather than a relaxed
 * text rule.
 */
export const RECOMMENDED_THRESHOLDS: Record<ContrastPairId, number> = {
  text_on_bg: WCAG_AAA_TEXT,
  text_on_surface: WCAG_AAA_TEXT,
  muted_on_bg: WCAG_AA_TEXT,
  accent_on_bg: WCAG_LARGE_OBJECT,
}

export interface ContrastRule {
  id: ContrastPairId
  foreground: PaletteKey
  background: PaletteKey
  /** Blocking floor — `REQUIRED_THRESHOLDS`. Below this, submit is refused. */
  min: number
  /** `RECOMMENDED_THRESHOLDS`. Between `min` and this is a warning, not a block. */
  recommended: number
  /** Which WCAG level `min` is drawn from — shown next to the number. */
  level: 'AA' | 'Non-text'
  /** Short name for the pair. */
  label: string
  /** What this pair is on an actual slide. */
  what: string
  /** Mirrors the backend's `_REMEDY`: the move that fixes a failure. */
  remedy: string
}

/** The four pairs the backend measures, in `contrast_report()` order. */
export const CONTRAST_RULES: readonly ContrastRule[] = [
  {
    id: 'text_on_bg',
    foreground: 'text',
    background: 'bg',
    min: REQUIRED_THRESHOLDS.text_on_bg,
    recommended: RECOMMENDED_THRESHOLDS.text_on_bg,
    level: 'AA',
    label: 'Text on background',
    what: 'Headings and bullets — every word on the slide.',
    remedy: 'lighten or darken Text away from Background',
  },
  {
    id: 'text_on_surface',
    foreground: 'text',
    background: 'surface',
    min: REQUIRED_THRESHOLDS.text_on_surface,
    recommended: RECOMMENDED_THRESHOLDS.text_on_surface,
    level: 'AA',
    label: 'Text on surface',
    what: 'Words that land on a panel rather than the bare background.',
    remedy: 'move Surface closer to Background in tone',
  },
  {
    id: 'muted_on_bg',
    foreground: 'muted',
    background: 'bg',
    min: REQUIRED_THRESHOLDS.muted_on_bg,
    recommended: RECOMMENDED_THRESHOLDS.muted_on_bg,
    level: 'AA',
    label: 'Muted on background',
    what: 'Kickers, dates and source lines.',
    remedy: 'raise Muted toward Text — it is too close to Background',
  },
  {
    id: 'accent_on_bg',
    foreground: 'accent',
    background: 'bg',
    min: REQUIRED_THRESHOLDS.accent_on_bg,
    recommended: RECOMMENDED_THRESHOLDS.accent_on_bg,
    level: 'Non-text',
    label: 'Accent on background',
    what: 'Heading rule and bullet markers. A graphic, so 3.0 is the bar — not 4.5.',
    remedy: 'pick a lighter or darker tint of the same accent hue',
  },
]

export interface ContrastCheck {
  rule: ContrastRule
  /** `null` when either colour in the pair is unparseable. */
  ratio: number | null
  /** `ratio` at two decimals, or an em dash. */
  display: string
  /** Cleared the blocking floor. False means submit is refused. */
  passes: boolean
  /** Cleared the 4.5 web minimum for body text. */
  meetsAA: boolean
  /** Cleared 7.0. Shown as a badge. */
  meetsAAA: boolean
  /**
   * Clears AA but misses the 7.0 we recommend for video. Not blocking — a
   * legitimate brand colour often lands here — but worth saying out loud.
   */
  warns: boolean
  /** Cleared the recommendation as well as the floor. */
  meetsRecommended: boolean
  /** The backend's failure sentence, same shape as `review_theme` emits. */
  message: string | null
  /** The backend's warning sentence, or `null`. */
  warning: string | null
}

export interface PaletteReport {
  checks: ContrastCheck[]
  /** Blocking failures, in `CONTRAST_RULES` order (most important first). */
  failures: ContrastCheck[]
  /** AA-but-under-7.0 pairs. These do not block. */
  warnings: ContrastCheck[]
  /** Palette keys whose value is not a readable hex colour. */
  malformed: PaletteKey[]
  /** True when the palette is submittable: parseable and every pair clears AA. */
  isValid: boolean
  /** True when it also clears every recommendation — no warnings at all. */
  isRecommended: boolean
  isLight: boolean
}

function checkPair(palette: Palette, rule: ContrastRule): ContrastCheck {
  const ratio = contrastRatio(palette[rule.foreground], palette[rule.background])
  const display = formatRatio(ratio)
  const passes = ratio !== null && ratio >= rule.min
  const meetsRecommended = ratio !== null && ratio >= rule.recommended
  const warns = passes && !meetsRecommended
  return {
    rule,
    ratio,
    display,
    passes,
    meetsAA: ratio !== null && ratio >= WCAG_AA_TEXT,
    meetsAAA: ratio !== null && ratio >= WCAG_AAA_TEXT,
    warns,
    meetsRecommended,
    message: passes
      ? null
      : `${rule.id} is ${display}:1, needs >= ${rule.min.toFixed(2)}:1 — ${rule.remedy}`,
    warning: warns
      ? `${rule.id} is ${display}:1, which passes WCAG AA but is under the ${rule.recommended.toFixed(1)}:1 we recommend for video — thin type may smear on a projector or at low bitrate`
      : null,
  }
}

/**
 * Measures every pair and splits the result into blocking failures and
 * non-blocking warnings, mirroring `review_theme(theme) -> (failures, warnings)`.
 *
 * Malformed colours short-circuit the ratios, exactly as the backend does: a
 * ratio measured against a colour we could not parse is meaningless, so it is
 * reported as unknown rather than as a failure of contrast.
 */
export function evaluatePalette(palette: Palette): PaletteReport {
  const malformed = PALETTE_KEYS.filter((key) => !isHexColour(palette[key]))
  const checks = CONTRAST_RULES.map((rule) => checkPair(palette, rule))
  const failures = checks.filter((check) => !check.passes)
  const warnings = checks.filter((check) => check.warns)
  return {
    checks,
    failures,
    warnings,
    malformed,
    isValid: malformed.length === 0 && failures.length === 0,
    isRecommended: malformed.length === 0 && failures.length === 0 && warnings.length === 0,
    isLight: isLightBackground(palette.bg),
  }
}

/** Every ratio as a plain map, in the shape the backend reports them. */
export function contrastReport(palette: Palette): Record<ContrastPairId, number | null> {
  const report = {} as Record<ContrastPairId, number | null>
  for (const rule of CONTRAST_RULES) {
    report[rule.id] = contrastRatio(palette[rule.foreground], palette[rule.background])
  }
  return report
}

/* ------------------------------------------------------------------ *
 * Repair — mirrors `suggest_fix` in `app/core/themes.py`
 *
 * Lightness only, in HLS, with hue and saturation held. A brand is mostly hue
 * and chroma, so a palette repaired this way still looks like the customer's
 * palette. Recolouring to a "safe" blue would clear the gate and be rejected
 * by whoever owns the brand.
 *
 * The backend sends a `suggested_fix` with its 422 and that response wins when
 * we have it. This local copy is what makes the "Fix contrast" button work
 * before any request is made — which is the only time the user actually needs
 * it, since submit is blocked while the palette fails.
 * ------------------------------------------------------------------ */

/** Python's `round()` is half-to-even; matching it keeps the hex identical. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value)
  const diff = value - floor
  if (diff > 0.5) return floor + 1
  if (diff < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

interface Hls {
  h: number
  l: number
  s: number
}

/** Port of `colorsys.rgb_to_hls`, channels 0-1. */
function rgbToHls({ r, g, b }: Rgb): Hls {
  const rf = r / 255
  const gf = g / 255
  const bf = b / 255
  const maxc = Math.max(rf, gf, bf)
  const minc = Math.min(rf, gf, bf)
  const sumc = maxc + minc
  const rangec = maxc - minc
  const l = sumc / 2
  if (minc === maxc) return { h: 0, l, s: 0 }
  const s = l <= 0.5 ? rangec / sumc : rangec / (2 - maxc - minc)
  const rc = (maxc - rf) / rangec
  const gc = (maxc - gf) / rangec
  const bc = (maxc - bf) / rangec
  let h: number
  if (rf === maxc) h = bc - gc
  else if (gf === maxc) h = 2 + rc - bc
  else h = 4 + gc - rc
  h = ((h / 6) % 1 + 1) % 1
  return { h, l, s }
}

function hlsChannel(m1: number, m2: number, hueIn: number): number {
  const hue = ((hueIn % 1) + 1) % 1
  if (hue < 1 / 6) return m1 + (m2 - m1) * hue * 6
  if (hue < 0.5) return m2
  if (hue < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - hue) * 6
  return m1
}

/** Port of `colorsys.hls_to_rgb`, returning a `#RRGGBB` string. */
function hlsToHex({ h, l, s }: Hls): string {
  const clamped = Math.min(1, Math.max(0, l))
  if (s === 0) {
    const grey = roundHalfEven(clamped * 255)
    return toHex({ r: grey, g: grey, b: grey })
  }
  const m2 = clamped <= 0.5 ? clamped * (1 + s) : clamped + s - clamped * s
  const m1 = 2 * clamped - m2
  return toHex({
    r: roundHalfEven(hlsChannel(m1, m2, h + 1 / 3) * 255),
    g: roundHalfEven(hlsChannel(m1, m2, h) * 255),
    b: roundHalfEven(hlsChannel(m1, m2, h - 1 / 3) * 255),
  })
}

/**
 * Smallest lightness move on `colour` that reaches `target` against `against`.
 *
 * Contrast is monotonic in HLS lightness for a fixed hue and saturation, so a
 * binary search between the current lightness and the nearer extreme finds the
 * least brand drift that clears the bar. Returns the clamped extreme when the
 * target is unreachable, so the caller can detect that by re-measuring.
 */
function solveLightness(
  colour: string,
  against: string,
  target: number,
  lighten: boolean,
): string {
  const rgb = parseHex(colour)
  if (rgb === null) return colour
  const { h, l: current, s } = rgbToHls(rgb)
  const bound = lighten ? 1 : 0
  const extreme = hlsToHex({ h, l: bound, s })
  const atExtreme = contrastRatio(extreme, against)
  if (atExtreme === null || atExtreme < target) return extreme

  let lo = current // fails
  let hi = bound // passes
  for (let i = 0; i < 24; i += 1) {
    const mid = (lo + hi) / 2
    const ratio = contrastRatio(hlsToHex({ h, l: mid, s }), against)
    if (ratio !== null && ratio >= target) hi = mid
    else lo = mid
  }
  return hlsToHex({ h, l: hi, s })
}

function ratioOf(a: string, b: string): number {
  return contrastRatio(a, b) ?? 0
}

/**
 * The nearest palette that clears every *recommendation*, hue and saturation
 * kept.
 *
 * Note the tier: like the backend's `suggest_fix`, this targets
 * `RECOMMENDED_THRESHOLDS` (7.0 for text), not the AA floor that gates
 * submission. A repair should leave the palette comfortably good rather than
 * one hundredth above the minimum, so the button also clears warnings.
 *
 * Foregrounds move first and `bg` is held, because the background is the
 * colour a customer actually recognises. Only when a foreground driven all the
 * way to white or black still fails — a mid-tone ground like `#808080`, where
 * the best achievable ratio is about 5.3:1 — does the background give.
 *
 * A palette that already clears everything comes back unchanged. Malformed
 * colours cannot be repaired and are returned untouched: validate first.
 */
export function suggestFix(palette: Palette): Palette {
  if (PALETTE_KEYS.some((key) => !isHexColour(palette[key]))) return { ...palette }

  const fixed: Palette = { ...palette }
  // Recomputed from the original background so every tier moves the same way.
  const lighten = !isLightBackground(fixed.bg)

  const textMin = RECOMMENDED_THRESHOLDS.text_on_bg
  if (ratioOf(fixed.text, fixed.bg) < textMin) {
    let candidate = solveLightness(fixed.text, fixed.bg, textMin, lighten)
    if (ratioOf(candidate, fixed.bg) < textMin) {
      fixed.bg = solveLightness(fixed.bg, candidate, textMin, !lighten)
      candidate = solveLightness(fixed.text, fixed.bg, textMin, lighten)
    }
    fixed.text = candidate
  }

  // `surface` is a ground, so it moves rather than the text sitting on it, and
  // it moves toward bg's polarity to stay a plausible sibling of bg.
  const surfaceMin = RECOMMENDED_THRESHOLDS.text_on_surface
  if (ratioOf(fixed.text, fixed.surface) < surfaceMin) {
    fixed.surface = solveLightness(fixed.surface, fixed.text, surfaceMin, !lighten)
  }

  const mutedMin = RECOMMENDED_THRESHOLDS.muted_on_bg
  if (ratioOf(fixed.muted, fixed.bg) < mutedMin) {
    fixed.muted = solveLightness(fixed.muted, fixed.bg, mutedMin, lighten)
  }
  const accentMin = RECOMMENDED_THRESHOLDS.accent_on_bg
  if (ratioOf(fixed.accent, fixed.bg) < accentMin) {
    fixed.accent = solveLightness(fixed.accent, fixed.bg, accentMin, lighten)
  }

  return fixed
}
