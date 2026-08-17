/**
 * Every network call lives here.
 *
 * The backend is built in parallel, so responses are parsed defensively:
 * we accept both bare arrays and `{ voices: [...] }` / `{ jobs: [...] }`
 * envelopes, and tolerate missing optional fields rather than throwing.
 */

import {
  isLightBackground,
  normalizeHex,
  PALETTE_KEYS,
  type ContrastPairId,
  type Palette,
} from '@/lib/contrast'
import {
  DEEPGRAM_ENGINE_ID,
  DEFAULT_ENGINE_ID,
  DEFAULT_NO_LOGO_VALUE,
  DEFAULT_THEME_ID,
  OPTIONAL_JOB_FIELDS,
  PIPELINE_STAGES,
  POLLY_ENGINE_ID,
  type BrandLogo,
  type Bullet,
  type CreateJobRequest,
  type CreateJobResponse,
  type Job,
  type JobStatus,
  type LogoCatalogue,
  type LogoRejection,
  type PipelineStage,
  type ScenePlan,
  type SpeechEngine,
  type ThemeContrastFailure,
  type ThemePreset,
  type Timeline,
  type TimelineScene,
  type Voice,
  type VoiceEngineMismatch,
} from '@/lib/types'

const API_BASE = '/api'

/** Used when `GET /api/voices` is unreachable or returns something unusable. */
export const FALLBACK_VOICES: Voice[] = [
  {
    id: 'aura-2-draco-en',
    label: 'Draco',
    accent: 'British',
    tags: ['warm', 'baritone', 'storytelling'],
    description: 'Warm British baritone, great for narrative explainers.',
  },
  {
    id: 'aura-2-pluto-en',
    label: 'Pluto',
    accent: 'American',
    tags: ['calm', 'baritone'],
    description: 'Calm American baritone with an even, measured pace.',
  },
  {
    id: 'aura-2-hera-en',
    label: 'Hera',
    accent: 'American',
    tags: ['smooth', 'warm', 'feminine'],
    description: 'Smooth, warm feminine voice.',
  },
  {
    id: 'aura-2-athena-en',
    label: 'Athena',
    accent: 'American',
    tags: ['calm', 'professional', 'feminine'],
    description: 'Calm, professional feminine voice.',
  },
  {
    id: 'aura-2-orpheus-en',
    label: 'Orpheus',
    accent: 'American',
    tags: ['professional', 'masculine'],
    description: 'Professional masculine voice with clear diction.',
  },
  {
    id: 'aura-2-cora-en',
    label: 'Cora',
    accent: 'American',
    tags: ['melodic', 'feminine'],
    description: 'Melodic feminine voice with expressive delivery.',
  },
]

export const DEFAULT_VOICE_ID = 'aura-2-draco-en'

/**
 * Polly voices, used when `GET /api/voices?engine=polly` cannot be trusted.
 *
 * Restricted to the en-US **generative** voices confirmed on our account —
 * these are the ones worth narrating with, and a voice id Polly does not
 * recognise is a hard failure rather than a downgrade. `long-form` is tagged
 * where the voice also supports that engine tier.
 */
export const FALLBACK_POLLY_VOICES: Voice[] = [
  {
    id: 'Danielle',
    label: 'Danielle',
    accent: 'American',
    tags: ['feminine', 'generative', 'long-form'],
    description: 'Generative feminine voice; also supports long-form, so it holds up over a full module.',
  },
  {
    id: 'Ruth',
    label: 'Ruth',
    accent: 'American',
    tags: ['feminine', 'generative', 'long-form'],
    description: 'Generative feminine voice with long-form support.',
  },
  {
    id: 'Joanna',
    label: 'Joanna',
    accent: 'American',
    tags: ['feminine', 'generative'],
    description: 'Generative feminine voice, even and neutral.',
  },
  {
    id: 'Salli',
    label: 'Salli',
    accent: 'American',
    tags: ['feminine', 'generative'],
    description: 'Generative feminine voice, brighter delivery.',
  },
  {
    id: 'Matthew',
    label: 'Matthew',
    accent: 'American',
    tags: ['masculine', 'generative'],
    description: 'Generative masculine voice with clear diction.',
  },
  {
    id: 'Stephen',
    label: 'Stephen',
    accent: 'American',
    tags: ['masculine', 'generative'],
    description: 'Generative masculine voice, measured pace.',
  },
]

/** Built-in catalogue per engine, keyed by engine id. */
export const FALLBACK_VOICES_BY_ENGINE: Record<string, Voice[]> = {
  [DEEPGRAM_ENGINE_ID]: FALLBACK_VOICES,
  [POLLY_ENGINE_ID]: FALLBACK_POLLY_VOICES,
}

/**
 * Preferred voice per engine. Both are narration-first picks: Draco is a warm
 * British baritone built for storytelling; Danielle is generative *and*
 * long-form, which is the closest Polly equivalent for a multi-minute module.
 */
export const DEFAULT_VOICE_BY_ENGINE: Record<string, string> = {
  [DEEPGRAM_ENGINE_ID]: DEFAULT_VOICE_ID,
  [POLLY_ENGINE_ID]: 'Danielle',
}

export function fallbackVoicesFor(engineId: string): Voice[] {
  return FALLBACK_VOICES_BY_ENGINE[engineId] ?? FALLBACK_VOICES
}

/**
 * Engines offered when `GET /api/engines` is unreachable.
 *
 * Deepgram is marked available because it is the engine the pipeline has been
 * narrating with all along. Polly is marked **unknown** (`null`), not
 * available: with no endpoint to ask, nothing has checked whether the AWS
 * credentials are configured, and claiming otherwise would send the user into
 * a render that dies at the `narrating` stage.
 */
export const FALLBACK_ENGINES: SpeechEngine[] = [
  {
    id: DEEPGRAM_ENGINE_ID,
    name: 'Deepgram Aura 2',
    supports_ssml: false,
    available: true,
    is_default: true,
    description: '53 English voices. The engine this pipeline already uses.',
  },
  {
    id: POLLY_ENGINE_ID,
    name: 'AWS Polly',
    supports_ssml: true,
    available: null,
    is_default: false,
    description: 'Generative and neural voices with SSML markup.',
  },
]

/**
 * Whether a voice id is plausibly native to an engine.
 *
 * This exists because the running backend **ignores** `?engine=`: asking for
 * `polly` returns the Deepgram catalogue verbatim. Accepting that list would
 * put `aura-2-draco-en` in the Polly picker and send it to Polly, which is the
 * exact failure the selector is meant to prevent. Deepgram ids are lowercase
 * hyphenated `aura-*` models; Polly ids are bare PascalCase names. Unknown
 * engines accept anything — a future engine must not be gated by this guess.
 */
export function isPlausibleVoiceForEngine(engineId: string, voiceId: string): boolean {
  if (engineId === DEEPGRAM_ENGINE_ID) return voiceId.toLowerCase().startsWith('aura')
  if (engineId === POLLY_ENGINE_ID) return /^[A-Z][A-Za-z]*$/.test(voiceId)
  return true
}

/**
 * Used when `GET /api/themes` is unreachable — the endpoint is being built now,
 * so a 404 is the expected case rather than an exceptional one.
 *
 * This is a snapshot of `app.core.themes.PRESETS` and `THEME_META`, taken from
 * the source. Ratios are deliberately not copied: they are a pure function of
 * the swatches and `lib/contrast.ts` computes them, so the numbers on a
 * fallback card cannot go stale against the colours shown beside them.
 */
export const FALLBACK_THEMES: ThemePreset[] = [
  {
    id: 'midnight',
    name: 'Midnight',
    description:
      'Deep navy with an amber accent — the default; confident and neutral for policy, security and compliance modules.',
    is_light: false,
    is_default: true,
    swatches: {
      bg: '#0B1220',
      surface: '#131F35',
      text: '#F8FAFC',
      muted: '#94A3B8',
      accent: '#F5A524',
    },
    contrast: {},
  },
  {
    id: 'graphite',
    name: 'Graphite',
    description:
      "Neutral charcoal with a cool blue accent — the safest dark option when the footage has to sit under someone else's brand.",
    is_light: false,
    is_default: false,
    swatches: {
      bg: '#15171C',
      surface: '#212530',
      text: '#F4F6F8',
      muted: '#A2A9B6',
      accent: '#38BDF8',
    },
    contrast: {},
  },
  {
    id: 'halo',
    name: 'Halo',
    description:
      "Near-black plum carrying the product's own violet — use for first-party launch, onboarding and enablement content.",
    is_light: false,
    is_default: false,
    swatches: {
      bg: '#140C24',
      surface: '#20153A',
      text: '#F6F2FF',
      muted: '#B7A9DA',
      accent: '#863BFF',
    },
    contrast: {},
  },
  {
    id: 'forest',
    name: 'Forest',
    description:
      'Deep pine with warm gold — for sustainability, operations and field-safety training that should feel grounded rather than technical.',
    is_light: false,
    is_default: false,
    swatches: {
      bg: '#0C2118',
      surface: '#143024',
      text: '#F1F7F3',
      muted: '#96B8A6',
      accent: '#F0B429',
    },
    contrast: {},
  },
  {
    id: 'daylight',
    name: 'Daylight',
    description:
      'Bright near-white with ink text — the clearest option for dense process walkthroughs and anything watched on a projector in a lit room.',
    is_light: true,
    is_default: false,
    swatches: {
      bg: '#F7F8FA',
      surface: '#FFFFFF',
      text: '#111827',
      muted: '#55607A',
      accent: '#1D4ED8',
    },
    contrast: {},
  },
  {
    id: 'boardroom',
    name: 'Boardroom',
    description:
      'Cool grey-blue with a corporate blue accent — for leadership, finance and formal announcement decks.',
    is_light: true,
    is_default: false,
    swatches: {
      bg: '#E8EDF4',
      surface: '#F8FAFD',
      text: '#0F2440',
      muted: '#49597A',
      accent: '#0E6BA8',
    },
    contrast: {},
  },
  {
    id: 'paper',
    name: 'Paper',
    description:
      'Warm off-white with brick red — an editorial, low-glare feel for long-form culture, ethics and HR narrative content.',
    is_light: true,
    is_default: false,
    swatches: {
      bg: '#F6F1E7',
      surface: '#FFFDF8',
      text: '#211C15',
      muted: '#5D5346',
      accent: '#A8442A',
    },
    contrast: {},
  },
  {
    id: 'lilac',
    name: 'Lilac',
    description:
      'Soft violet-tinted white behind the brand purple — the light counterpart to Halo, for customer-facing product education.',
    is_light: true,
    is_default: false,
    swatches: {
      bg: '#F5F1FF',
      surface: '#FFFFFF',
      text: '#1B1230',
      muted: '#574A78',
      accent: '#863BFF',
    },
    contrast: {},
  },
]

export class ApiError extends Error {
  readonly status: number

  /**
   * The parsed `detail` from the error body, untouched. Kept because the
   * contrast gate answers a 422 with a structured object — failures plus a
   * suggested palette — and flattening that to a string would throw away the
   * one-click fix. See `parseThemeContrastFailure`.
   */
  readonly detail: unknown

  constructor(message: string, status: number, detail?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/* ------------------------------------------------------------------ *
 * Narrowing helpers — keeps the parsers free of `any`.
 * ------------------------------------------------------------------ */

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function readString(source: Record<string, unknown>, ...keys: string[]): string | null {
  for (const key of keys) {
    const value = source[key]
    if (typeof value === 'string' && value.trim() !== '') return value
  }
  return null
}

function readNumber(source: Record<string, unknown>, ...keys: string[]): number | null {
  for (const key of keys) {
    const value = source[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value)
      if (Number.isFinite(parsed)) return parsed
    }
  }
  return null
}

function readBoolean(source: Record<string, unknown>, ...keys: string[]): boolean | null {
  for (const key of keys) {
    const value = source[key]
    if (typeof value === 'boolean') return value
  }
  return null
}

function readStringArray(source: Record<string, unknown>, ...keys: string[]): string[] | undefined {
  for (const key of keys) {
    const value = source[key]
    if (Array.isArray(value)) {
      const items = value.filter((item): item is string => typeof item === 'string')
      if (items.length > 0) return items
    }
    // Some APIs hand back a comma-separated string instead of a list.
    if (typeof value === 'string' && value.includes(',')) {
      const items = value
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item !== '')
      if (items.length > 0) return items
    }
  }
  return undefined
}

/** Unwraps `{ key: [...] }` envelopes as well as bare arrays. */
function readList(payload: unknown, ...keys: string[]): unknown[] {
  if (Array.isArray(payload)) return payload
  const record = asRecord(payload)
  if (!record) return []
  for (const key of keys) {
    const value = record[key]
    if (Array.isArray(value)) return value
  }
  return []
}

const STAGE_SET = new Set<string>(PIPELINE_STAGES)

function toStage(value: string | null): PipelineStage | null {
  if (value === null) return null
  const normalized = value.trim().toLowerCase()
  return STAGE_SET.has(normalized) ? (normalized as PipelineStage) : null
}

function toStatus(value: string | null): JobStatus {
  if (value === null) return 'queued'
  const normalized = value.trim().toLowerCase()
  if (normalized === 'failed' || normalized === 'error') return 'failed'
  const stage = toStage(normalized)
  return stage ?? 'queued'
}

/** Title-cases a Deepgram model id: `aura-2-draco-en` -> `Draco`. */
function labelFromVoiceId(id: string): string {
  const parts = id.split('-').filter((part) => part !== '')
  const name = parts.length >= 3 ? parts[2] : parts[0]
  if (name === undefined || name === '') return id
  return name.charAt(0).toUpperCase() + name.slice(1)
}

/* ------------------------------------------------------------------ *
 * Parsers
 * ------------------------------------------------------------------ */

function parseVoice(raw: unknown): Voice | null {
  if (typeof raw === 'string') {
    return { id: raw, label: labelFromVoiceId(raw) }
  }
  const record = asRecord(raw)
  if (!record) return null

  const id = readString(record, 'id', 'voice', 'model', 'name', 'value', 'canonical_name')
  if (id === null) return null

  const label = readString(record, 'label', 'display_name', 'displayName', 'title', 'name')
  const accent = readString(record, 'accent', 'language', 'locale', 'region')
  const description = readString(record, 'description', 'summary', 'blurb')

  // The backend `Voice` schema splits descriptors across `tags` and
  // `use_cases`; both are useful, so they're shown as one de-duped chip set.
  const tags = readStringArray(record, 'tags', 'characteristics', 'traits', 'labels') ?? []
  const useCases = readStringArray(record, 'use_cases', 'useCases') ?? []
  const allTags = [...new Set([...tags, ...useCases])]

  const voice: Voice = {
    id,
    // If `name` doubled as the id, derive a friendlier label instead.
    label: label !== null && label !== id ? label : labelFromVoiceId(id),
  }
  if (accent !== null) voice.accent = accent
  if (description !== null) voice.description = description
  if (allTags.length > 0) voice.tags = allTags
  return voice
}

/** Known SSML behaviour, used only when the payload omits `supports_ssml`. */
const KNOWN_SSML_SUPPORT: Record<string, boolean> = {
  [DEEPGRAM_ENGINE_ID]: false,
  [POLLY_ENGINE_ID]: true,
}

/** Display names for the two engines we know, when the payload omits one. */
const KNOWN_ENGINE_NAMES: Record<string, string> = {
  [DEEPGRAM_ENGINE_ID]: 'Deepgram Aura 2',
  [POLLY_ENGINE_ID]: 'AWS Polly',
}

function parseEngine(raw: unknown): SpeechEngine | null {
  // A bare id is accepted: an early build may answer `["deepgram", "polly"]`.
  if (typeof raw === 'string') {
    const id = raw.trim()
    if (id === '') return null
    return {
      id,
      name: KNOWN_ENGINE_NAMES[id] ?? id,
      supports_ssml: KNOWN_SSML_SUPPORT[id] ?? false,
      // Listed but unqualified: the same "unknown" the fallback catalogue uses.
      available: null,
      is_default: id === DEFAULT_ENGINE_ID,
    }
  }

  const record = asRecord(raw)
  if (record === null) return null

  const id = readString(record, 'id', 'engine', 'tts_engine', 'value', 'key', 'name')
  if (id === null) return null

  const name = readString(record, 'name', 'label', 'display_name', 'displayName', 'title')
  const description = readString(record, 'description', 'blurb', 'summary')
  const reason = readString(
    record,
    'unavailable_reason',
    'unavailableReason',
    'reason',
    'detail',
    'error',
  )

  const engine: SpeechEngine = {
    id,
    name: name !== null && name !== id ? name : (KNOWN_ENGINE_NAMES[id] ?? id),
    supports_ssml:
      readBoolean(record, 'supports_ssml', 'supportsSsml', 'ssml') ??
      KNOWN_SSML_SUPPORT[id] ??
      false,
    // Absent means unverified, not usable.
    available: readBoolean(record, 'available', 'is_available', 'enabled', 'configured'),
    is_default: readBoolean(record, 'default', 'is_default', 'isDefault') ?? false,
  }
  const defaultVoice = readString(record, 'default_voice', 'defaultVoice')
  if (defaultVoice !== null) engine.default_voice = defaultVoice
  if (description !== null) engine.description = description
  if (reason !== null) engine.unavailable_reason = reason
  return engine
}

/**
 * The engine to select, given a catalogue.
 *
 * An engine whose credentials are missing is never preselected — the point of
 * the flag is to keep the user out of a render that cannot narrate. "Unknown"
 * availability is allowed through, because refusing it would leave nothing
 * selected whenever `/api/engines` is missing.
 */
export function preferredEngineId(engines: SpeechEngine[]): string {
  const usable = engines.filter((engine) => engine.available !== false)
  const flagged = usable.find((engine) => engine.is_default)
  if (flagged !== undefined) return flagged.id
  const named = usable.find((engine) => engine.id === DEFAULT_ENGINE_ID)
  if (named !== undefined) return named.id
  return usable[0]?.id ?? engines[0]?.id ?? DEFAULT_ENGINE_ID
}

/**
 * The voice to select for an engine, given that engine's catalogue.
 *
 * @param serverDefault the engine's own `default_voice`, which wins when the
 *   server sends one — it is the voice the render would actually use, so
 *   preferring a local guess over it would misreport the outcome.
 */
export function preferredVoiceId(
  engineId: string,
  voices: Voice[],
  serverDefault?: string,
): string {
  if (serverDefault !== undefined && voices.some((voice) => voice.id === serverDefault)) {
    return serverDefault
  }
  const preferred = DEFAULT_VOICE_BY_ENGINE[engineId]
  if (preferred !== undefined && voices.some((voice) => voice.id === preferred)) return preferred
  return voices[0]?.id ?? serverDefault ?? preferred ?? DEFAULT_VOICE_ID
}

/* ------------------------------------------------------------------ *
 * Theme parsers
 * ------------------------------------------------------------------ */

/**
 * Reads a palette from either shape the backend might send.
 *
 * `app.core.themes.list_themes()` builds `swatches` as an object keyed by
 * colour name, but the `ThemeOut` response model beside it declares
 * `list[str]`. One of the two is wrong and it is not settled yet, so both are
 * accepted: a five-element array is read in `PALETTE_KEYS` order, which is the
 * order `PALETTE_FIELDS` uses on the server.
 *
 * Colours are normalised through `normalizeHex`, so a lowercase or shorthand
 * value from either source lands in the canonical `#RRGGBB` the editor and the
 * `theme_custom` payload both need.
 */
function parsePalette(raw: unknown): Palette | null {
  if (Array.isArray(raw)) {
    const colours = raw.map((item) => (typeof item === 'string' ? normalizeHex(item) : null))
    if (colours.length < PALETTE_KEYS.length) return null
    const palette = {} as Palette
    for (const [index, key] of PALETTE_KEYS.entries()) {
      const colour = colours[index]
      if (colour === undefined || colour === null) return null
      palette[key] = colour
    }
    return palette
  }

  const record = asRecord(raw)
  if (record === null) return null
  const palette = {} as Palette
  for (const key of PALETTE_KEYS) {
    const value = readString(record, key)
    const colour = value === null ? null : normalizeHex(value)
    if (colour === null) return null
    palette[key] = colour
  }
  return palette
}

const CONTRAST_PAIR_IDS: readonly ContrastPairId[] = [
  'text_on_bg',
  'text_on_surface',
  'muted_on_bg',
  'accent_on_bg',
]

/** Whatever subset of the four ratios the payload carried. */
function parseContrastMap(raw: unknown): Partial<Record<ContrastPairId, number>> {
  const record = asRecord(raw)
  if (record === null) return {}
  const map: Partial<Record<ContrastPairId, number>> = {}
  for (const pair of CONTRAST_PAIR_IDS) {
    const value = readNumber(record, pair)
    if (value !== null) map[pair] = value
  }
  return map
}

function parseThemePreset(raw: unknown): ThemePreset | null {
  const record = asRecord(raw)
  if (record === null) return null

  const swatches = parsePalette(record.swatches ?? record.palette ?? record.colours)
  // A preset we cannot draw is worse than a preset we do not offer: the whole
  // point of the card is showing the palette.
  if (swatches === null) return null

  const id = readString(record, 'id', 'name', 'key')
  if (id === null) return null

  const preset: ThemePreset = {
    id,
    name: readString(record, 'name', 'label', 'title') ?? id,
    description: readString(record, 'description', 'blurb', 'summary') ?? '',
    // Trust the server's polarity when it sends one; otherwise derive it the
    // same way `Theme.is_light` does.
    is_light: readBoolean(record, 'is_light', 'isLight') ?? isLightBackground(swatches.bg),
    is_default: readBoolean(record, 'is_default', 'isDefault') ?? id === DEFAULT_THEME_ID,
    swatches,
    contrast: parseContrastMap(record.contrast ?? record.contrast_report),
  }

  // Not serialised by `list_themes()` today, but `ThemeOut` is `extra="allow"`,
  // so it can appear without a client change. Left absent when it does not, and
  // `logoOpacityFor` derives it from `is_light`.
  const logoOpacity = readNumber(record, 'logo_opacity', 'logoOpacity')
  if (logoOpacity !== null && logoOpacity > 0) preset.logo_opacity = Math.min(1, logoOpacity)

  return preset
}

/**
 * Pulls the structured contrast failure out of a 422 body.
 *
 * The gate answers with `detail` as an object — `{error, message, failures,
 * contrast, suggested_fix, suggested_contrast}` — while a plain schema
 * rejection answers with `detail` as a *list* of pydantic errors. Returning
 * `null` for anything that is not the contrast shape is what lets `createJob`
 * tell the two apart.
 *
 * Both `suggested_fix` and `suggest_fix` are read: the endpoint sends the
 * former, the API sketch specified the latter.
 */
export function parseThemeContrastFailure(detail: unknown): ThemeContrastFailure | null {
  const record = asRecord(detail)
  if (record === null) return null
  if (readString(record, 'error') !== 'theme_contrast_failed') return null

  const failures = readList(record, 'failures').filter(
    (item): item is string => typeof item === 'string',
  )

  return {
    message:
      readString(record, 'message') ?? 'The custom palette is not readable on screen.',
    failures,
    contrast: parseContrastMap(record.contrast),
    suggestedFix: parsePalette(record.suggested_fix ?? record.suggest_fix),
    suggestedContrast: parseContrastMap(record.suggested_contrast ?? record.suggest_contrast),
  }
}

/**
 * Pulls the `voice_engine_mismatch` 422 out of a body, or `null`.
 *
 * Same shape of job as `parseThemeContrastFailure`: a tagged object `detail`,
 * as opposed to the list of pydantic errors a plain schema rejection sends.
 */
export function parseVoiceEngineMismatch(detail: unknown): VoiceEngineMismatch | null {
  const record = asRecord(detail)
  if (record === null) return null
  if (readString(record, 'error') !== 'voice_engine_mismatch') return null

  const voice = readString(record, 'voice')
  return {
    message:
      readString(record, 'message') ??
      'That voice belongs to a different speech engine.',
    voice: voice ?? '',
    voiceEngine: readString(record, 'voice_engine', 'voiceEngine'),
    ttsEngine: readString(record, 'tts_engine', 'ttsEngine'),
    engineDefaultVoice: readString(record, 'engine_default_voice', 'engineDefaultVoice'),
  }
}

/* ------------------------------------------------------------------ *
 * Brand-logo parsers
 *
 * `/api/logos` does not exist on the running instance (verified: 404), so these
 * are written against the field names the backend agent published and accept the
 * plausible variants of each, the same way `parseEngine` accepts both `default`
 * and `is_default`.
 * ------------------------------------------------------------------ */

function parseBrandLogo(raw: unknown): BrandLogo | null {
  // A bare id is accepted: an early build may answer `["abc123", "def456"]`.
  if (typeof raw === 'string') {
    const id = raw.trim()
    if (id === '') return null
    return {
      id,
      url: `${API_BASE}/logos/${encodeURIComponent(id)}`,
      width: null,
      height: null,
      format: null,
      has_alpha: null,
      warning: null,
      original_filename: null,
      bytes: null,
      created_at: null,
    }
  }

  const record = asRecord(raw)
  if (record === null) return null

  const id = readString(record, 'id', 'logo_id', 'logoId', 'key', 'name', 'filename')
  if (id === null) return null

  return {
    id,
    // A relative path from the server wins; otherwise the documented route is
    // derived, because a logo we cannot draw is not worth offering.
    url:
      readString(record, 'url', 'href', 'file_url', 'fileUrl', 'path', 'location') ??
      `${API_BASE}/logos/${encodeURIComponent(id)}`,
    width: readNumber(record, 'width', 'w', 'pixel_width'),
    height: readNumber(record, 'height', 'h', 'pixel_height'),
    format: readString(record, 'format', 'content_type', 'contentType', 'mime', 'ext')?.toLowerCase() ?? null,
    // Absent means unknown, never "no": an opaque mark composites as a solid
    // rectangle, so the difference is worth reporting honestly.
    has_alpha: readBoolean(record, 'has_alpha', 'hasAlpha', 'alpha', 'transparent'),
    warning: readString(record, 'warning', 'warnings', 'notice', 'caveat', 'message'),
    original_filename: readString(record, 'original_filename', 'originalFilename', 'filename', 'name'),
    bytes: readNumber(record, 'bytes', 'size', 'size_bytes', 'byte_size'),
    created_at: readString(record, 'created_at', 'createdAt', 'uploaded_at', 'created'),
  }
}

/**
 * Reads the catalogue, including the server's own spelling of "no logo".
 *
 * The backend agent was still choosing between `"none"` and another spelling, so
 * an envelope field naming it is honoured when present and `DEFAULT_NO_LOGO_VALUE`
 * is assumed otherwise. Whatever wins, an entry *matching* it is filtered out of
 * the uploaded list: "no logo" is one of the picker's own fixed options, not a
 * logo someone uploaded.
 */
export function parseLogoCatalogue(payload: unknown): Omit<LogoCatalogue, 'available'> {
  const envelope = asRecord(payload)
  const noneValue =
    (envelope === null
      ? null
      : readString(
          envelope,
          'none_value',
          'noneValue',
          'no_logo_value',
          'noLogoValue',
          'none_id',
          'disabled_value',
          'sentinel',
        )) ?? DEFAULT_NO_LOGO_VALUE

  const logos = readList(payload, 'logos', 'items', 'data', 'results')
    .map(parseBrandLogo)
    .filter((logo): logo is BrandLogo => logo !== null && logo.id !== noneValue)

  return { logos, noneValue }
}

/**
 * Pulls a logo-specific 422 out of a body, or `null`.
 *
 * Two shapes have to be recognised, because they mean different things:
 *
 * 1. A tagged object — `{error: "unknown_logo", …}` — the endpoint saying the id
 *    is not one of its own. The user picks a different logo.
 * 2. A list of pydantic errors whose `loc` names `logo_id`. `extra_forbidden`
 *    there means the backend has no `logo_id` field at all, which is a different
 *    remedy: switch to the built-in mark and the job goes through.
 *
 * Either way the answer is to show it, not to retry without the field.
 */
export function parseLogoRejection(detail: unknown, sent: string | null): LogoRejection | null {
  const record = asRecord(detail)
  if (record !== null) {
    const error = readString(record, 'error', 'code') ?? ''
    if (!/logo/i.test(error)) return null
    return {
      message:
        readString(record, 'message', 'detail') ??
        'The server rejected that brand logo.',
      logoId: sent,
      unsupported: /unsupported|unknown_field|not_supported|extra/i.test(error),
    }
  }

  if (!Array.isArray(detail)) return null
  for (const item of detail) {
    const entry = asRecord(item)
    if (entry === null) continue
    const loc = Array.isArray(entry.loc)
      ? entry.loc.filter((part): part is string => typeof part === 'string')
      : []
    if (!loc.includes('logo_id')) continue
    const type = readString(entry, 'type') ?? ''
    const message = readString(entry, 'msg', 'message')
    const unsupported = type.includes('extra') || type.includes('unexpected')
    return {
      message: unsupported
        ? 'This backend does not accept a brand logo yet.'
        : (message ?? 'The server rejected that brand logo.'),
      logoId: sent,
      unsupported,
    }
  }
  return null
}

/**
 * @param fallbackId used when the payload omits its own id — the detail
 *   endpoint is documented to return only status fields.
 */
export function parseJob(raw: unknown, fallbackId?: string): Job | null {
  const record = asRecord(raw)
  if (!record) return null

  const jobId = readString(record, 'job_id', 'jobId', 'id') ?? fallbackId ?? null
  if (jobId === null) return null

  const status = toStatus(readString(record, 'status', 'state'))
  const stage = toStage(readString(record, 'current_stage', 'currentStage', 'stage'))
  const progressRaw = readNumber(record, 'progress', 'percent', 'percentage')

  // A `done` job with no reported progress is still 100% complete.
  let progress = progressRaw ?? (status === 'done' ? 100 : 0)
  // Tolerate a 0..1 fraction instead of 0..100.
  if (progress > 0 && progress <= 1 && !Number.isInteger(progress)) progress *= 100
  progress = Math.max(0, Math.min(100, Math.round(progress)))

  const job: Job = {
    job_id: jobId,
    status,
    progress,
    current_stage: stage ?? (status === 'failed' ? null : toStage(status)),
    error: readString(record, 'error', 'error_message', 'errorMessage'),
    video_url: readString(record, 'video_url', 'videoUrl', 'url', 'output_url'),
    topic: readString(record, 'topic', 'prompt'),
    created_at: readString(record, 'created_at', 'createdAt', 'created', 'timestamp'),
    title: readString(record, 'title'),
    slide_count: readNumber(record, 'slide_count', 'slideCount', 'scene_count'),
    voice: readString(record, 'voice'),
    music: readBoolean(record, 'music'),
    tts_engine: readString(record, 'tts_engine', 'ttsEngine', 'engine'),
    theme: readString(record, 'theme', 'theme_id', 'themeId'),
    theme_custom: parsePalette(record.theme_custom ?? record.themeCustom),
    bullets_per_slide: readNumber(record, 'bullets_per_slide', 'bulletsPerSlide'),
    tone: readString(record, 'tone'),
  }

  /*
   * `logo_id` is three-state, so `readString` alone is not enough: it collapses
   * "the server sent null" into the same `null` as "the server has no such
   * field". Those mean different things — null is documented as "fall back to the
   * built-in mark", absent means the backend predates the feature — and the badge
   * shows nothing for the latter rather than inventing history.
   */
  const logoKey = ['logo_id', 'logoId', 'logo'].find((key) => key in record)
  if (logoKey !== undefined) {
    job.logo_id = readString(record, 'logo_id', 'logoId', 'logo')
  }

  return job
}

/* ------------------------------------------------------------------ *
 * Timeline parsers
 *
 * Written against the real payload from
 * `GET /api/jobs/43859ea1-.../timeline`, which at the time of writing had
 * NO `bullets` key on any scene and a `plan` missing every animation
 * field. Everything below therefore treats each field as optional.
 * ------------------------------------------------------------------ */

/** Distinguishes "key absent" from "key present but empty". */
function hasKey(source: Record<string, unknown>, ...keys: string[]): boolean {
  return keys.some((key) => key in source && source[key] !== null)
}

function parseBullet(raw: unknown): Bullet | null {
  // Tolerate a bare string: an early script-stage payload may not have
  // attached timings yet (`SceneScript.bullets` is `list[str]`).
  if (typeof raw === 'string') {
    const text = raw.trim()
    return text === '' ? null : { text, appear_at: 0, emphasis: false }
  }

  const record = asRecord(raw)
  if (record === null) return null

  const text = readString(record, 'text', 'label', 'content')
  if (text === null) return null

  return {
    text,
    appear_at: Math.max(0, readNumber(record, 'appear_at', 'appearAt', 'start') ?? 0),
    emphasis: readBoolean(record, 'emphasis', 'emphasise', 'emphasized') ?? false,
  }
}

function parsePlan(raw: unknown): ScenePlan | null {
  const record = asRecord(raw)
  if (record === null) return null

  return {
    layout: readString(record, 'layout', 'slide_layout'),
    motion: readString(record, 'motion'),
    zoom_from: readNumber(record, 'zoom_from', 'zoomFrom'),
    zoom_to: readNumber(record, 'zoom_to', 'zoomTo'),
    easing: readString(record, 'easing'),
    transition_in: readString(record, 'transition_in', 'transitionIn', 'transition'),
    transition_duration: readNumber(record, 'transition_duration', 'transitionDuration'),
    text_position: readString(record, 'text_position', 'textPosition'),
    scrim_opacity: readNumber(record, 'scrim_opacity', 'scrimOpacity'),
    heading_animation: readString(record, 'heading_animation', 'headingAnimation'),
    bullet_animation: readString(record, 'bullet_animation', 'bulletAnimation'),
    anim_duration: readNumber(record, 'anim_duration', 'animDuration'),
    bullet_min_gap: readNumber(record, 'bullet_min_gap', 'bulletMinGap'),
  }
}

function parseScene(raw: unknown, index: number): TimelineScene | null {
  const record = asRecord(raw)
  if (record === null) return null

  const start = readNumber(record, 'start') ?? 0
  const end = readNumber(record, 'end') ?? 0

  const bullets = hasKey(record, 'bullets', 'points')
    ? readList(record, 'bullets', 'points')
        .map(parseBullet)
        .filter((bullet): bullet is Bullet => bullet !== null)
    : null

  const words = readList(record, 'words')

  return {
    id: readNumber(record, 'id', 'index', 'scene_id') ?? index + 1,
    index,
    heading: readString(record, 'heading', 'title'),
    narration: readString(record, 'narration', 'text', 'script'),
    start,
    end,
    duration: Math.max(0, end - start),
    bullets,
    wordCount: words.length,
    hasImage: readString(record, 'image_path', 'imagePath') !== null,
    hasAudio: readString(record, 'audio_path', 'audioPath') !== null,
    hasClip: readString(record, 'clip_path', 'clipPath') !== null,
    plan: parsePlan(record.plan ?? record.visual_plan),
  }
}

export function parseTimeline(raw: unknown, fallbackId?: string): Timeline | null {
  // A bare array of scenes is accepted as well as the `Timeline` envelope.
  const record = asRecord(raw) ?? (Array.isArray(raw) ? { scenes: raw } : null)
  if (record === null) return null

  const scenes = readList(record, 'scenes', 'items')
    .map((item, index) => parseScene(item, index))
    .filter((scene): scene is TimelineScene => scene !== null)

  return {
    job_id: readString(record, 'job_id', 'jobId', 'id') ?? fallbackId ?? null,
    topic: readString(record, 'topic'),
    title: readString(record, 'title'),
    voice: readString(record, 'voice'),
    hasMusic: readString(record, 'music_path', 'musicPath') !== null,
    scenes,
    duration: scenes.reduce((longest, scene) => Math.max(longest, scene.end), 0),
    hasTimings: scenes.some((scene) => scene.duration > 0),
    hasBullets: scenes.some((scene) => scene.bullets !== null && scene.bullets.length > 0),
  }
}

/* ------------------------------------------------------------------ *
 * Requests
 * ------------------------------------------------------------------ */

async function request(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError('Cannot reach the server. Is the backend running?', 0)
  }

  const text = await response.text()
  let payload: unknown = null
  if (text !== '') {
    try {
      payload = JSON.parse(text) as unknown
    } catch {
      payload = null
    }
  }

  if (!response.ok) {
    const record = asRecord(payload)
    const message =
      (record !== null ? readString(record, 'detail', 'error', 'message') : null) ??
      `Request failed (${String(response.status)})`
    // `detail` is carried through structurally as well as flattened into the
    // message: `readString` above finds nothing when the detail is an object
    // (the contrast gate) or a list (a pydantic rejection), and both of those
    // are cases the caller needs to inspect rather than just print.
    throw new ApiError(message, response.status, record === null ? payload : record.detail)
  }

  return payload
}

export interface VoicesResult {
  voices: Voice[]
  usedFallback: boolean
  /**
   * Set when the server answered but the list did not belong to the engine we
   * asked for — currently the normal case for Polly, because `?engine=` is
   * ignored. Distinct from `usedFallback` alone: the endpoint was up, so
   * "unavailable" would be the wrong thing to tell the user.
   */
  engineMismatch: boolean
}

/**
 * Fetches the voices for one engine. Never rejects — falls back to the
 * built-in catalogue for that engine so the form is always usable.
 *
 * The `engine` query param is new and the running backend ignores it, so the
 * response is checked against the engine before it is trusted: see
 * `isPlausibleVoiceForEngine`. Called with no argument it is the original
 * no-param request, which must keep working against older builds.
 */
export async function fetchVoices(engineId?: string): Promise<VoicesResult> {
  const fallback = engineId === undefined ? FALLBACK_VOICES : fallbackVoicesFor(engineId)
  const path =
    engineId === undefined ? '/voices' : `/voices?engine=${encodeURIComponent(engineId)}`

  try {
    const payload = await request(path)
    const voices = readList(payload, 'voices', 'items', 'data')
      .map(parseVoice)
      .filter((voice): voice is Voice => voice !== null)

    if (voices.length === 0) return { voices: fallback, usedFallback: true, engineMismatch: false }

    if (engineId !== undefined) {
      const native = voices.filter((voice) => isPlausibleVoiceForEngine(engineId, voice.id))
      // A partial match is a filtered list; no match at all means the server
      // handed back some other engine's catalogue, so none of it is usable.
      if (native.length === 0) {
        return { voices: fallback, usedFallback: true, engineMismatch: true }
      }
      return {
        voices: native,
        usedFallback: false,
        engineMismatch: native.length !== voices.length,
      }
    }

    return { voices, usedFallback: false, engineMismatch: false }
  } catch {
    return { voices: fallback, usedFallback: true, engineMismatch: false }
  }
}

/**
 * Fetches the speech engines. Never rejects.
 *
 * `GET /api/engines` is being wired as this ships, so a 404 is the expected
 * case: the fallback catalogue offers both engines and marks Polly's
 * availability unknown rather than claiming it is configured.
 */
export async function fetchEngines(): Promise<{
  engines: SpeechEngine[]
  usedFallback: boolean
}> {
  try {
    const payload = await request('/engines')
    const engines = readList(payload, 'engines', 'items', 'data')
      .map(parseEngine)
      .filter((engine): engine is SpeechEngine => engine !== null)

    if (engines.length === 0) return { engines: FALLBACK_ENGINES, usedFallback: true }
    return { engines, usedFallback: false }
  } catch {
    return { engines: FALLBACK_ENGINES, usedFallback: true }
  }
}

/**
 * Fetches the theme catalogue. Never rejects.
 *
 * `GET /api/themes` is being written as this ships, so a 404 is routine: the
 * fallback catalogue keeps the picker fully usable and the caller reports the
 * degradation in the UI rather than hiding it.
 */
export async function fetchThemes(): Promise<{ themes: ThemePreset[]; usedFallback: boolean }> {
  try {
    const payload = await request('/themes')
    const themes = readList(payload, 'themes', 'items', 'data')
      .map(parseThemePreset)
      .filter((theme): theme is ThemePreset => theme !== null)

    if (themes.length === 0) return { themes: FALLBACK_THEMES, usedFallback: true }
    return { themes, usedFallback: false }
  } catch {
    return { themes: FALLBACK_THEMES, usedFallback: true }
  }
}

/* ------------------------------------------------------------------ *
 * Brand logos
 * ------------------------------------------------------------------ */

/**
 * Fetches the uploaded-logo catalogue. Never rejects.
 *
 * `available: false` is the expected state while `/api/logos` is being written:
 * the form keeps the built-in mark selected and says uploads are unavailable
 * rather than breaking. A 404 and an unreachable server are both "not available"
 * — from the form's point of view there is nothing to upload to either way.
 */
export async function fetchLogos(): Promise<LogoCatalogue> {
  try {
    const payload = await request('/logos')
    const { logos, noneValue } = parseLogoCatalogue(payload)
    // An empty list from a live endpoint is a real, useful state: uploading
    // works, nothing has been uploaded yet.
    return { logos, available: true, noneValue }
  } catch {
    return { logos: [], available: false, noneValue: DEFAULT_NO_LOGO_VALUE }
  }
}

/** Thrown by `uploadLogo` when the endpoint is not there at all. */
export class LogoUploadUnavailableError extends Error {
  constructor() {
    super('This backend does not accept logo uploads yet.')
    this.name = 'LogoUploadUnavailableError'
  }
}

/**
 * Uploads a logo, reporting progress.
 *
 * `XMLHttpRequest` rather than `fetch`, for one reason: `fetch` cannot report
 * upload progress (request streams are not available for this in Safari, and
 * `duplex: 'half'` is a different feature). A 4MiB file on a slow link is long
 * enough that a bar beats a spinner.
 *
 * The `Content-Type` is deliberately not set — the browser has to add the
 * multipart boundary, and setting it by hand produces a 422 with no boundary.
 */
export async function uploadLogo(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<BrandLogo> {
  try {
    return await postLogo(file, 'file', onProgress)
  } catch (cause) {
    // FastAPI names the multipart field after its `UploadFile` parameter, and
    // `file` is the convention — but if this build called it something else the
    // 422 says so precisely (`loc: ["body", "<name>"], type: "missing"`), which
    // is worth one retry rather than a dead end. Only the *named* field is
    // retried; nothing is guessed.
    const named = cause instanceof ApiError && cause.status === 422 ? missingFormField(cause.detail) : null
    if (named === null || named === 'file') throw cause
    return postLogo(file, named, onProgress)
  }
}

/** The single missing multipart field a 422 names, or `null`. */
function missingFormField(detail: unknown): string | null {
  if (!Array.isArray(detail)) return null
  for (const item of detail) {
    const entry = asRecord(item)
    if (entry === null) continue
    if (!(readString(entry, 'type') ?? '').includes('missing')) continue
    const loc = Array.isArray(entry.loc)
      ? entry.loc.filter((part): part is string => typeof part === 'string')
      : []
    const name = loc.at(-1)
    if (name !== undefined && name !== 'body') return name
  }
  return null
}

function postLogo(
  file: File,
  field: string,
  onProgress?: (fraction: number) => void,
): Promise<BrandLogo> {
  const form = new FormData()
  form.append(field, file, file.name)

  return new Promise<BrandLogo>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/logos`)
    xhr.responseType = 'text'
    xhr.setRequestHeader('Accept', 'application/json')

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || onProgress === undefined) return
      onProgress(Math.min(1, event.loaded / Math.max(1, event.total)))
    }

    xhr.onerror = () => {
      reject(new ApiError('Cannot reach the server. Is the backend running?', 0))
    }
    xhr.onabort = () => {
      reject(new ApiError('Upload cancelled.', 0))
    }

    xhr.onload = () => {
      let payload: unknown = null
      if (xhr.responseText !== '') {
        try {
          payload = JSON.parse(xhr.responseText) as unknown
        } catch {
          payload = null
        }
      }

      if (xhr.status === 404 || xhr.status === 405) {
        reject(new LogoUploadUnavailableError())
        return
      }

      if (xhr.status < 200 || xhr.status >= 300) {
        const record = asRecord(payload)
        const message =
          (record !== null ? readString(record, 'detail', 'error', 'message') : null) ??
          `Upload failed (${String(xhr.status)})`
        reject(new ApiError(message, xhr.status, record === null ? payload : record.detail))
        return
      }

      // The response may be the logo, or an envelope around it.
      const record = asRecord(payload)
      const logo =
        parseBrandLogo(payload) ??
        (record === null ? null : parseBrandLogo(record.logo ?? record.data ?? record.item))
      if (logo === null) {
        reject(new ApiError('The server did not return a usable logo.', xhr.status, payload))
        return
      }
      onProgress?.(1)
      resolve(logo)
    }

    xhr.send(form)
  })
}

/** Deletes a stored logo. A 404 is treated as success: it is already gone. */
export async function deleteLogo(logoId: string): Promise<void> {
  try {
    await request(`/logos/${encodeURIComponent(logoId)}`, { method: 'DELETE' })
  } catch (cause) {
    if (cause instanceof ApiError && cause.status === 404) return
    throw cause
  }
}

async function postJob(
  body: Record<string, unknown>,
): Promise<{ jobId: string; themeWarnings: string[] }> {
  const payload = await request('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  })

  const record = asRecord(payload)
  const jobId = record !== null ? readString(record, 'job_id', 'jobId', 'id') : null
  if (jobId === null) throw new ApiError('The server did not return a job id.', 500)

  // Advisory, and optional: an older build sends no such field.
  const themeWarnings =
    record === null
      ? []
      : readList(record.theme_warnings ?? record.themeWarnings).filter(
          (item): item is string => typeof item === 'string',
        )

  return { jobId, themeWarnings }
}

/**
 * Thrown when the backend's contrast gate rejects a custom palette.
 *
 * Separate from `ApiError` so the form can react to it specifically: it
 * carries the per-pair failures and a corrected palette to offer as one click,
 * neither of which survives being turned into an error string.
 */
export class ThemeContrastError extends Error {
  readonly failure: ThemeContrastFailure

  constructor(failure: ThemeContrastFailure) {
    super(failure.message)
    this.name = 'ThemeContrastError'
    this.failure = failure
  }
}

/**
 * Thrown when the backend rejects the voice/engine pair.
 *
 * Never retried. The generic 422 path would strip `tts_engine` and `voice` and
 * succeed, which means shipping a render in an engine and voice the user did not
 * choose — the exact substitution the server refused to make. The form prevents
 * this state, so reaching here means the UI and the server disagree about which
 * engine owns a voice, and that needs to be seen rather than smoothed over.
 */
export class VoiceEngineMismatchError extends Error {
  readonly mismatch: VoiceEngineMismatch

  constructor(mismatch: VoiceEngineMismatch) {
    super(mismatch.message)
    this.name = 'VoiceEngineMismatchError'
    this.mismatch = mismatch
  }
}

/**
 * Thrown when the backend refuses our `logo_id`.
 *
 * Never retried without the field, and this is the sharpest case of that rule in
 * the codebase. Stripping `logo_id` always *succeeds*, and the job that comes
 * back has the built-in mark burnt into the bottom-left corner of every frame —
 * either instead of the customer's logo, or in place of the "no logo" the user
 * deliberately chose. Both are a different video from the one requested, and the
 * user would have no way to tell from the form.
 */
export class LogoRejectedError extends Error {
  readonly rejection: LogoRejection

  constructor(rejection: LogoRejection) {
    super(rejection.message)
    this.name = 'LogoRejectedError'
    this.rejection = rejection
  }
}

/**
 * Creates a job, degrading gracefully if the backend has not caught up.
 *
 * Two different 422s have to be told apart here:
 *
 * 1. The contrast gate rejecting a custom palette. Its `detail` is an object
 *    tagged `theme_contrast_failed`. Retrying without the palette would
 *    silently render the video in a preset the user did not choose, so it is
 *    re-thrown as a `ThemeContrastError` for the form to show.
 * 2. A schema rejection from a backend that predates these fields. The running
 *    instance's `JobCreate` has no `extra="forbid"`, so unknown keys are
 *    ignored and this does not currently happen — but if it is ever tightened,
 *    the optional fields are stripped and the job is retried so the form keeps
 *    working. The caller is told what was dropped.
 */
export async function createJob(body: CreateJobRequest): Promise<CreateJobResponse> {
  const full: Record<string, unknown> = { ...body }
  // `theme_custom` is optional in the request type; an explicit `undefined`
  // would serialise away anyway, but dropping the key keeps the payload clean
  // and keeps `field in full` honest below.
  if (body.theme_custom === undefined) delete full.theme_custom
  // Omitting `logo_id` is meaningful — it is how the backend is told to use its
  // configured default mark — so an undefined one is removed rather than sent as
  // JSON `null`, which an older `JobCreate` would read as a bad string.
  if (body.logo_id === undefined) delete full.logo_id

  try {
    const { jobId, themeWarnings } = await postJob(full)
    return { job_id: jobId, droppedFields: [], themeWarnings }
  } catch (cause) {
    if (!(cause instanceof ApiError) || cause.status !== 422) throw cause

    const contrastFailure = parseThemeContrastFailure(cause.detail)
    if (contrastFailure !== null) throw new ThemeContrastError(contrastFailure)

    // Checked before the strip-and-retry: dropping the engine here would be
    // answered with a successful render in the wrong voice.
    const mismatch = parseVoiceEngineMismatch(cause.detail)
    if (mismatch !== null) throw new VoiceEngineMismatchError(mismatch)

    // Same reasoning, one step further: a logo 422 is only ever surfaced.
    // Checked whether or not we sent the field, because `extra_forbidden` on a
    // backend that has never heard of `logo_id` is exactly this error.
    const logoRejection = parseLogoRejection(cause.detail, body.logo_id ?? null)
    if (logoRejection !== null) throw new LogoRejectedError(logoRejection)

    const optional = OPTIONAL_JOB_FIELDS.filter((field) => field in full)
    if (optional.length === 0) throw cause

    const reduced: Record<string, unknown> = { ...full }
    for (const field of optional) delete reduced[field]

    const dropped: string[] = [...optional]

    /*
     * Dropping `tts_engine` makes the voice a liability. The backend falls back
     * to its own default engine, and a voice id is native to one engine only —
     * `Danielle` means nothing to Deepgram, `aura-2-draco-en` nothing to Polly.
     * `voice` defaults to `""` server-side, which selects that engine's own
     * default, so dropping it degrades the choice instead of failing the render.
     * Only done when the user had actually moved off the default engine.
     */
    if (optional.includes('tts_engine') && body.tts_engine !== DEFAULT_ENGINE_ID) {
      delete reduced.voice
      dropped.push('voice')
    }

    // If the 422 was about `topic` (or anything else we still send), this
    // second attempt fails the same way and the original error surfaces.
    const { jobId, themeWarnings } = await postJob(reduced)
    return { job_id: jobId, droppedFields: dropped, themeWarnings }
  }
}

export async function fetchJob(jobId: string): Promise<Job> {
  const payload = await request(`/jobs/${encodeURIComponent(jobId)}`)
  const job = parseJob(payload, jobId)
  if (job === null) throw new ApiError('The server returned an unreadable job.', 500)
  return job
}

/**
 * Fetches the Timeline artifact.
 *
 * Resolves to `null` when the artifact does not exist yet — the endpoint
 * 404s until the `scripting` stage has written one, which is a normal state
 * for a young job, not an error. Other failures still reject so the caller
 * can decide whether to retry.
 */
export async function fetchTimeline(jobId: string): Promise<Timeline | null> {
  let payload: unknown
  try {
    payload = await request(`/jobs/${encodeURIComponent(jobId)}/timeline`)
  } catch (cause) {
    // 404 = not written yet. 409/425 = "too early", if it ever says so.
    if (cause instanceof ApiError && [404, 409, 425].includes(cause.status)) return null
    throw cause
  }

  // Malformed or empty bodies are also treated as "not ready" rather than
  // surfacing a parse error over a job that is still running fine.
  return parseTimeline(payload, jobId)
}

export async function fetchJobs(): Promise<Job[]> {
  const payload = await request('/jobs')
  return readList(payload, 'jobs', 'items', 'data')
    // Wrapped, not passed by reference: `.map` would feed the index as fallbackId.
    .map((item) => parseJob(item))
    .filter((job): job is Job => job !== null)
}
