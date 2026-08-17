import { BracesIcon, TypeIcon } from 'lucide-react'

import { FALLBACK_ENGINES } from '@/lib/api'
import { cn } from '@/lib/utils'
import type { Job, SpeechEngine } from '@/lib/types'

interface EngineBadgeProps {
  job: Job
  engines: SpeechEngine[]
  /** Compact form for the history list. */
  small?: boolean
  className?: string
}

/**
 * The speech engine a job was narrated with.
 *
 * Returns `null` when the job carries no `tts_engine` — a backend that predates
 * the field, where guessing Deepgram would be inventing history for jobs
 * rendered before the choice existed.
 */
export function EngineBadge({ job, engines, small = false, className }: EngineBadgeProps) {
  const id = job.tts_engine ?? null
  if (id === null || id === '') return null

  const catalogue = engines.length > 0 ? engines : FALLBACK_ENGINES
  const engine = catalogue.find((candidate) => candidate.id === id) ?? null
  // An id we do not have a card for still gets a badge: it is what the server
  // says was used, and hiding it would be less honest than showing the raw id.
  const label = engine?.name ?? id
  const ssml = engine?.supports_ssml ?? null

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03]',
        small ? 'px-1.5 py-px text-[10px]' : 'px-2 py-1 text-xs',
        className,
      )}
      title={
        ssml === null
          ? `Narrated with ${label}`
          : ssml
            ? `Narrated with ${label} — narration was marked up with SSML`
            : `Narrated with ${label} — narration was sent as plain text`
      }
    >
      {!small &&
        (ssml === false ? (
          <TypeIcon className="size-3 shrink-0 text-amber-300/60" />
        ) : (
          <BracesIcon className="size-3 shrink-0 text-white/40" />
        ))}
      <span className={small ? 'text-white/45' : 'text-white/60'}>{label}</span>
      {!small && ssml !== null && (
        <span
          className={cn(
            'rounded px-1 py-px text-[9px] font-medium tracking-wide uppercase',
            ssml
              ? 'bg-emerald-500/15 text-emerald-300/90'
              : 'bg-amber-500/15 text-amber-300/90',
          )}
        >
          {ssml ? 'SSML' : 'Plain text'}
        </span>
      )}
    </span>
  )
}
