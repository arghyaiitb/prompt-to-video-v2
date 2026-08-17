/**
 * API payload contracts shared with the FastAPI backend.
 *
 * The stage list mirrors the backend `JobStatus` enum in
 * `backend/app/core/models.py` exactly.
 */

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

export interface Voice {
  /** Deepgram model id, e.g. `aura-2-draco-en`. */
  id: string
  /** Display name, e.g. "Draco". */
  label: string
  accent?: string
  tags?: string[]
  description?: string
}

export interface CreateJobRequest {
  topic: string
  slide_count: number
  voice: string
  music: boolean
}

export interface CreateJobResponse {
  job_id: string
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
   * ISO-8601 timestamp. The backend's current `JobStatusOut` schema does NOT
   * include a timestamp, so this is null in practice; the UI hides the field
   * rather than showing a placeholder. Kept so it lights up if one is added.
   */
  created_at: string | null
  /** Not in `JobStatusOut` today — the UI falls back to `topic`. */
  title?: string | null
  slide_count?: number | null
  voice?: string | null
  music?: boolean | null
}
