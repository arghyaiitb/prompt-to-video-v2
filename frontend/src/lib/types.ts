/**
 * API payload contracts shared with the FastAPI backend.
 *
 * The stage list mirrors the backend `JobStatus` enum in
 * `backend/app/core/models.py` exactly.
 */

import type { ContrastPairId, Palette } from '@/lib/contrast'

/** Pipeline stages in execution order. `done` is the terminal success state. */
export const PIPELINE_STAGES = [
  'queued',
  'scripting',
  'imaging',
  'narrating',
  'aligning',
  'scoring',
  'rendering',
  'assembling',
  'done',
] as const

export type PipelineStage = (typeof PIPELINE_STAGES)[number]

/** `failed` sits outside the ordered pipeline: it is a terminal error state. */
export type JobStatus = PipelineStage | 'failed'

export const TERMINAL_STATUSES: readonly JobStatus[] = ['done', 'failed']

export function isTerminal(status: JobStatus): boolean {
  return TERMINAL_STATUSES.includes(status)
}

/** Human-facing copy for each stage, used by the stepper. */
export const STAGE_LABELS: Record<PipelineStage, string> = {
  queued: 'Queued',
  scripting: 'Writing script',
  imaging: 'Generating images',
  narrating: 'Recording narration',
  aligning: 'Aligning words',
  scoring: 'Scoring music',
  rendering: 'Rendering scenes',
  assembling: 'Assembling video',
  done: 'Done',
}

export const STAGE_BLURBS: Record<PipelineStage, string> = {
  queued: 'Waiting for a worker to pick up the job.',
  scripting: 'Turning your topic into a scene-by-scene script.',
  imaging: 'Painting a still for every scene.',
  narrating: 'Synthesising the voiceover track.',
  aligning: 'Timing each word against the audio.',
  scoring: 'Laying a light music bed under the narration.',
  rendering: 'Applying camera moves to each still.',
  assembling: 'Stitching scenes, audio and transitions together.',
  done: 'Your video is ready to watch.',
}

/* ------------------------------------------------------------------ *
 * Brand themes — `GET /api/themes`
 *
 * Mirrors `ThemeOut` in `backend/app/api/themes.py` / the dicts built by
 * `app.core.themes.list_themes()`. Note the two disagree about `swatches`:
 * the response model declares `list[str]` while `list_themes()` returns a
 * `{bg, surface, ...}` object. `parseThemePreset` accepts either — see the
 * note there.
 * ------------------------------------------------------------------ */

export interface ThemePreset {
  /** Preset id — what `POST /api/jobs` wants in `theme`. */
  id: string
  name: string
  description: string
  /** `Theme.is_light`: luminance(bg) > 0.5. Drives the light/dark grouping. */
  is_light: boolean
  is_default: boolean
  swatches: Palette
  /**
   * Ratios as measured by the backend, rounded to 2dp. Possibly partial (or
   * absent), so the UI falls back to computing them from `swatches`.
   */
  contrast: Partial<Record<ContrastPairId, number>>
}

/** Sentinel `theme` value meaning "the palette is in `theme_custom`". */
export const CUSTOM_THEME_ID = 'custom'

/** `app.core.themes.DEFAULT_THEME_NAME`. */
export const DEFAULT_THEME_ID = 'midnight'

/**
 * A palette failing the backend's contrast gate: the parsed body of the 422
 * from `POST /api/jobs`. `suggestedFix` is the one-click correction.
 */
export interface ThemeContrastFailure {
  message: string
  /** One sentence per failing pair, naming the ratio and the fix. */
  failures: string[]
  contrast: Partial<Record<ContrastPairId, number>>
  suggestedFix: Palette | null
  suggestedContrast: Partial<Record<ContrastPairId, number>>
}

export interface Voice {
  /** Engine-native voice id — `aura-2-draco-en` (Deepgram) or `Danielle` (Polly). */
  id: string
  /** Display name, e.g. "Draco". */
  label: string
  accent?: string
  tags?: string[]
  description?: string
}

/* ------------------------------------------------------------------ *
 * Speech engines — `GET /api/engines`
 *
 * Mirrors the contract being wired now:
 * `[{id, name, supports_ssml, available, default}]`.
 * ------------------------------------------------------------------ */

export const DEEPGRAM_ENGINE_ID = 'deepgram'
export const POLLY_ENGINE_ID = 'polly'

/** Engine used when the selector has not resolved a catalogue yet. */
export const DEFAULT_ENGINE_ID = DEEPGRAM_ENGINE_ID

export interface SpeechEngine {
  /** What `POST /api/jobs` wants in `tts_engine`. */
  id: string
  name: string
  /**
   * Whether narration may be marked up. False means the engine reads tags
   * aloud, so the backend must send plain text — see `SSML_SUMMARY`.
   */
  supports_ssml: boolean
  /**
   * Credentials configured server-side.
   *
   * `null` means *unknown*, not "yes": it is what the fallback catalogue uses
   * when `GET /api/engines` is missing, because nothing has checked the
   * credentials in that case. Only an explicit `false` blocks selection.
   */
  available: boolean | null
  is_default: boolean
  /**
   * The voice this engine narrates with when a job names none — `default_voice`
   * in the server's `EngineOut`. Authoritative when present: it beats the
   * built-in preference, because it is what the render would actually use.
   */
  default_voice?: string
  description?: string
  /** Why `available` is false, as reported by the server if it says. */
  unavailable_reason?: string
}

/**
 * The backend's `voice_engine_mismatch` 422 — a voice belonging to a different
 * engine than `tts_engine`.
 *
 * The server refuses to normalise this rather than substituting a voice into a
 * six-minute render nobody chose, so the UI must not paper over it either. The
 * form makes the state unreachable by deriving the voice from the engine, which
 * makes this a tripwire: if it ever fires, the two sides disagree about which
 * engine owns a voice.
 */
export interface VoiceEngineMismatch {
  message: string
  voice: string
  /** Engine the voice actually belongs to, per the server. */
  voiceEngine: string | null
  ttsEngine: string | null
  /** The voice the server would have used — offered as the correction. */
  engineDefaultVoice: string | null
}

/** SSML tags the pipeline cares about, in the order the table shows them. */
export const SSML_TAGS = ['break', 'emphasis', 'prosody', 'say-as'] as const

export type SsmlTag = (typeof SSML_TAGS)[number]

export interface SsmlTierSupport {
  /** Polly `Engine` parameter value. */
  tier: string
  label: string
  supports: Record<SsmlTag, boolean>
  note?: string
}

/**
 * Measured against our own AWS account, per Polly engine tier.
 *
 * `<emphasis>` is the one that differs, and it differs in the awkward
 * direction: it fails on exactly the two tiers worth using. On generative and
 * neural voices Polly answers `InvalidSsmlException: Unsupported Generative
 * feature`, so the UI must not promise emphasis just because SSML is on.
 */
export const POLLY_SSML_TIERS: readonly SsmlTierSupport[] = [
  {
    tier: 'generative',
    label: 'Generative',
    supports: { break: true, emphasis: false, prosody: true, 'say-as': true },
    note: 'Most natural delivery. Rejects <emphasis>.',
  },
  {
    tier: 'neural',
    label: 'Neural',
    supports: { break: true, emphasis: false, prosody: true, 'say-as': true },
    note: 'Rejects <emphasis>.',
  },
  {
    tier: 'standard',
    label: 'Standard',
    supports: { break: true, emphasis: true, prosody: true, 'say-as': true },
    note: 'The only tier that accepts <emphasis>, and the least natural.',
  },
]

/**
 * One honest sentence per engine about what happens to the narration markup.
 *
 * Deepgram's entry is not a guess: given `<break time="800ms"/>` it says
 * "break time equals eight hundred milliseconds" out loud.
 */
export const SSML_SUMMARY: Record<string, string> = {
  [DEEPGRAM_ENGINE_ID]:
    'No SSML. Deepgram reads the tags aloud, so narration is sent as plain text — pacing comes from the script alone.',
  [POLLY_ENGINE_ID]:
    'SSML is active: pauses, pacing and number/date reading are marked up. <emphasis> is the exception — Polly rejects it on generative and neural voices, so it only applies on standard.',
}

export function ssmlSummary(engine: SpeechEngine): string {
  return (
    SSML_SUMMARY[engine.id] ??
    (engine.supports_ssml
      ? 'Narration is marked up with SSML.'
      : 'Narration is sent as plain text.')
  )
}

/* ------------------------------------------------------------------ *
 * Corporate-training options
 *
 * `theme`, `theme_custom`, `bullets_per_slide` and `tone` are all in the
 * backend's `JobCreate` on the branch being built now, but the running
 * instance may be older (verified: its /openapi.json lists only topic,
 * slide_count, voice, music). They are sent regardless — `JobCreate` has no
 * `extra="forbid"`, so an older build ignores them — and `createJob` retries
 * without them if one ever 422s. See OPTIONAL_JOB_FIELDS.
 * ------------------------------------------------------------------ */

export const TONE_OPTIONS = [
  {
    value: 'new_hires',
    label: 'New hires',
    hint: 'Assumes no prior context. Defines terms as it goes.',
  },
  {
    value: 'all_staff',
    label: 'All staff',
    hint: 'Plain language, everyday examples. The safe default.',
  },
  {
    value: 'technical',
    label: 'Technical',
    hint: 'Precise, uses domain vocabulary without hedging.',
  },
  {
    value: 'executives',
    label: 'Executives',
    hint: 'Impact and risk first, brief, decision-oriented.',
  },
] as const

export type Tone = (typeof TONE_OPTIONS)[number]['value']

export const DEFAULT_TONE: Tone = 'all_staff'

export const MIN_BULLETS = 3
export const MAX_BULLETS = 5
export const DEFAULT_BULLETS = 4

/** Fields the backend may not accept yet — dropped on a 422 retry. */
export const OPTIONAL_JOB_FIELDS = [
  'theme',
  'theme_custom',
  'bullets_per_slide',
  'tone',
  'tts_engine',
] as const

/**
 * Human-facing names for the fields above, for the "settings ignored" toast.
 *
 * `voice` is in here despite not being optional: dropping `tts_engine` can
 * force the voice to be dropped with it, because an engine-native voice id
 * means nothing to a different engine. See `createJob`.
 */
export const OPTIONAL_FIELD_LABELS: Record<
  (typeof OPTIONAL_JOB_FIELDS)[number] | 'voice',
  string
> = {
  theme: 'the brand theme',
  theme_custom: 'your custom colours',
  bullets_per_slide: 'bullets per slide',
  tone: 'the audience',
  tts_engine: 'the speech engine',
  voice: 'the narrator voice',
}

export interface CreateJobRequest {
  topic: string
  slide_count: number
  /**
   * A voice id belonging to `tts_engine`. The two travel together: a Deepgram
   * id sent with `polly` is a server-side failure, so the form derives this
   * from the selected engine rather than storing it independently.
   */
  voice: string
  music: boolean
  /** Engine id from `GET /api/engines`. */
  tts_engine: string
  /** Preset id from `GET /api/themes`, or `custom` when `theme_custom` is set. */
  theme: string
  /**
   * Own-colours override. Every colour is required — the backend's
   * `ThemeCustom` is `extra="forbid"` with all five fields mandatory, so a
   * half palette is a 422, not a merge with the preset.
   */
  theme_custom?: Palette
  /** 3-5 on-screen points per slide. */
  bullets_per_slide: number
  /** Audience the script should be written for. */
  tone: Tone
}

export interface CreateJobResponse {
  job_id: string
  /**
   * Names of request fields the backend rejected, forcing a retry without
   * them. Empty when the whole payload was accepted.
   */
  droppedFields: string[]
  /**
   * `theme_warnings` from the 202: pairs that clear WCAG AA but sit under the
   * 7.0 recommended for video. The job was accepted — these are advisory.
   */
  themeWarnings: string[]
}

export interface Job {
  job_id: string
  status: JobStatus
  /** 0-100. */
  progress: number
  current_stage: PipelineStage | null
  error: string | null
  /** Relative path, e.g. `/api/jobs/{id}/video` (proxied in dev). */
  video_url: string | null
  topic: string | null
  /**
   * ISO-8601 timestamp, e.g. `2026-08-17T06:48:33.221471`. Present on both
   * `GET /api/jobs` and `GET /api/jobs/{id}` (verified). Note it carries no
   * timezone suffix, so it is interpreted as local time — see
   * `parseTimestamp` in `lib/format.ts`. Still nullable: the UI hides the
   * field rather than printing a placeholder.
   */
  created_at: string | null
  /** Not in `JobStatusOut` today — the UI falls back to `topic`. */
  title?: string | null
  slide_count?: number | null
  voice?: string | null
  music?: boolean | null
  /**
   * Engine the narration was actually synthesised with. Absent on builds that
   * predate the selector, in which case the UI shows nothing rather than
   * assuming Deepgram.
   */
  tts_engine?: string | null
  /**
   * Preset id the job was actually rendered with — normalised by the backend,
   * so an unknown request id comes back as the default. `custom` means the
   * palette is in `theme_custom`. Absent on older builds.
   */
  theme?: string | null
  theme_custom?: Palette | null
  bullets_per_slide?: number | null
  tone?: string | null
}

/* ------------------------------------------------------------------ *
 * Timeline artifact — `GET /api/jobs/{id}/timeline`
 *
 * Mirrors `Timeline` / `Scene` / `BulletPoint` / `VisualPlan` in
 * `backend/app/core/models.py`. Enum-valued plan fields are typed as
 * plain strings on purpose: the backend's StrEnums are still growing
 * (`SlideLayout` and the animation fields landed after the sample job
 * was rendered), and an unrecognised value must render as a badge, not
 * crash or vanish. Known values get pretty labels below; anything else
 * falls back to a de-slugged version of the raw token.
 * ------------------------------------------------------------------ */

/** `Motion` in the backend enum. */
export const MOTION_LABELS: Record<string, string> = {
  zoom_in: 'Zoom in',
  zoom_out: 'Zoom out',
  pan_left: 'Pan left',
  pan_right: 'Pan right',
  static: 'Static',
}

/** `Transition`. Note the backend spells these `slideleft` / `wiperight`. */
export const TRANSITION_LABELS: Record<string, string> = {
  fade: 'Fade',
  dissolve: 'Dissolve',
  slideleft: 'Slide left',
  wiperight: 'Wipe right',
  cut: 'Cut',
}

/** `TextPosition`. */
export const TEXT_POSITION_LABELS: Record<string, string> = {
  center: 'Centre',
  lower_third: 'Lower third',
  upper_third: 'Upper third',
  left_panel: 'Left panel',
}

/** `TextAnimation`. */
export const TEXT_ANIMATION_LABELS: Record<string, string> = {
  none: 'None',
  fade_in: 'Fade in',
  slide_up: 'Slide up',
  slide_left: 'Slide left',
  pop: 'Pop',
  typewriter: 'Typewriter',
}

/** `SlideLayout` — added to `VisualPlan` after the sample job rendered. */
export const SLIDE_LAYOUT_LABELS: Record<string, string> = {
  title_card: 'Title card',
  hero_right: 'Hero right',
  hero_left: 'Hero left',
  image_band: 'Image band',
  full_bleed: 'Full bleed',
}

/** One on-screen point, revealed while the narration says it. */
export interface Bullet {
  text: string
  /** Seconds **relative to the scene start**, not the global timeline. */
  appear_at: number
  /** Rendered in the accent colour by the backend. */
  emphasis: boolean
}

export interface ScenePlan {
  layout: string | null
  motion: string | null
  zoom_from: number | null
  zoom_to: number | null
  easing: string | null
  transition_in: string | null
  transition_duration: number | null
  text_position: string | null
  scrim_opacity: number | null
  heading_animation: string | null
  bullet_animation: string | null
  anim_duration: number | null
  /** Floor on the spacing between bullet reveals. Default 0.6s. */
  bullet_min_gap: number | null
}

export const DEFAULT_BULLET_MIN_GAP = 0.6

export interface TimelineScene {
  /** Backend scene id (1-based in practice). */
  id: number
  /** Position in the `scenes` array — always usable as a React key. */
  index: number
  heading: string | null
  narration: string | null
  /** Global timeline seconds. */
  start: number
  end: number
  /** `end - start`, clamped at 0. */
  duration: number
  /**
   * `null` means the payload had no `bullets` key at all (pre-bullets
   * backend); `[]` means the key was there but empty. Both render as an
   * empty state, with different copy.
   */
  bullets: Bullet[] | null
  /** Aligned word count — 0 until the `aligning` stage has run. */
  wordCount: number
  hasImage: boolean
  hasAudio: boolean
  hasClip: boolean
  plan: ScenePlan | null
}

export interface Timeline {
  job_id: string | null
  topic: string | null
  title: string | null
  voice: string | null
  hasMusic: boolean
  scenes: TimelineScene[]
  /** Narration duration: the largest scene `end`. */
  duration: number
  /** True once at least one scene reports a non-zero `end`. */
  hasTimings: boolean
  /** True once at least one scene reports a bullet. */
  hasBullets: boolean
}

/** Stages before which the timeline artifact cannot exist yet. */
export function timelineCouldExist(status: JobStatus | null): boolean {
  return status !== null && status !== 'queued'
}
