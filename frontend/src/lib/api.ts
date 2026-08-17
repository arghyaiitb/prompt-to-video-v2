/**
 * Every network call lives here.
 *
 * The backend is built in parallel, so responses are parsed defensively:
 * we accept both bare arrays and `{ voices: [...] }` / `{ jobs: [...] }`
 * envelopes, and tolerate missing optional fields rather than throwing.
 */

import {
  PIPELINE_STAGES,
  type CreateJobRequest,
  type CreateJobResponse,
  type Job,
  type JobStatus,
  type PipelineStage,
  type Voice,
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

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
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

  return {
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
    throw new ApiError(message, response.status)
  }

  return payload
}

/**
 * Fetches available voices. Never rejects — falls back to the verified
 * hardcoded list so the form is always usable.
 */
export async function fetchVoices(): Promise<{ voices: Voice[]; usedFallback: boolean }> {
  try {
    const payload = await request('/voices')
    const voices = readList(payload, 'voices', 'items', 'data')
      .map(parseVoice)
      .filter((voice): voice is Voice => voice !== null)

    if (voices.length === 0) return { voices: FALLBACK_VOICES, usedFallback: true }
    return { voices, usedFallback: false }
  } catch {
    return { voices: FALLBACK_VOICES, usedFallback: true }
  }
}

export async function createJob(body: CreateJobRequest): Promise<CreateJobResponse> {
  const payload = await request('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  })

  const record = asRecord(payload)
  const jobId = record !== null ? readString(record, 'job_id', 'jobId', 'id') : null
  if (jobId === null) throw new ApiError('The server did not return a job id.', 500)
  return { job_id: jobId }
}

export async function fetchJob(jobId: string): Promise<Job> {
  const payload = await request(`/jobs/${encodeURIComponent(jobId)}`)
  const job = parseJob(payload, jobId)
  if (job === null) throw new ApiError('The server returned an unreadable job.', 500)
  return job
}

export async function fetchJobs(): Promise<Job[]> {
  const payload = await request('/jobs')
  return readList(payload, 'jobs', 'items', 'data')
    // Wrapped, not passed by reference: `.map` would feed the index as fallbackId.
    .map((item) => parseJob(item))
    .filter((job): job is Job => job !== null)
}
