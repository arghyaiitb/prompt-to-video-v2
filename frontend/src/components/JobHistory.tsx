import {
  AlertTriangleIcon,
  ClockIcon,
  FilmIcon,
  Loader2Icon,
  RefreshCwIcon,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { formatRelativeTime } from '@/lib/format'
import { cn } from '@/lib/utils'
import { STAGE_LABELS, type Job, type JobStatus } from '@/lib/types'

interface JobHistoryProps {
  jobs: Job[]
  isLoading: boolean
  error: string | null
  activeJobId: string | null
  onSelect: (jobId: string) => void
  onReload: () => void
}

function statusChip(status: JobStatus): { label: string; className: string } {
  if (status === 'done') {
    return {
      label: 'Ready',
      className: 'border-emerald-400/25 bg-emerald-500/10 text-emerald-200',
    }
  }
  if (status === 'failed') {
    return { label: 'Failed', className: 'border-red-400/25 bg-red-500/10 text-red-200' }
  }
  return {
    label: STAGE_LABELS[status],
    className: 'border-violet-400/25 bg-violet-500/10 text-violet-200',
  }
}

export function JobHistory({
  jobs,
  isLoading,
  error,
  activeJobId,
  onSelect,
  onReload,
}: JobHistoryProps) {
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-xs font-medium tracking-wide text-white/40 uppercase">
          <ClockIcon className="size-3.5" />
          Recent videos
        </h2>
        <Button
          variant="ghost"
          size="icon"
          onClick={onReload}
          aria-label="Refresh history"
          className="size-7 text-white/30 hover:bg-white/5 hover:text-white/70"
        >
          <RefreshCwIcon className={cn('size-3.5', isLoading && 'animate-spin')} />
        </Button>
      </div>

      {isLoading && jobs.length === 0 && (
        <div className="space-y-2">
          {[0, 1, 2].map((key) => (
            <Skeleton key={key} className="h-[74px] w-full rounded-xl bg-white/[0.04]" />
          ))}
        </div>
      )}

      {error !== null && jobs.length === 0 && !isLoading && (
        <p className="flex items-start gap-2 rounded-xl border border-white/[0.07] bg-white/[0.015] p-3 text-xs leading-relaxed text-white/35">
          <AlertTriangleIcon className="mt-px size-3.5 shrink-0 text-amber-300/60" />
          History unavailable — {error}
        </p>
      )}

      {error === null && !isLoading && jobs.length === 0 && (
        <p className="rounded-xl border border-dashed border-white/10 p-6 text-center text-xs text-white/30">
          No videos yet. Your first one will show up here.
        </p>
      )}

      <ul className="space-y-2">
        {jobs.map((job) => {
          const chip = statusChip(job.status)
          const isActive = job.job_id === activeJobId
          const inFlight = job.status !== 'done' && job.status !== 'failed'

          return (
            <li key={job.job_id}>
              <button
                type="button"
                onClick={() => {
                  onSelect(job.job_id)
                }}
                aria-current={isActive}
                className={cn(
                  'group w-full rounded-xl border p-3 text-left transition-all',
                  isActive
                    ? 'border-violet-400/40 bg-violet-500/[0.08]'
                    : 'border-white/[0.07] bg-white/[0.015] hover:border-white/15 hover:bg-white/[0.04]',
                )}
              >
                <div className="flex items-start gap-2.5">
                  <span
                    className={cn(
                      'mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg border',
                      isActive
                        ? 'border-violet-400/30 bg-violet-500/15 text-violet-200'
                        : 'border-white/10 bg-white/[0.03] text-white/40',
                    )}
                  >
                    {inFlight ? (
                      <Loader2Icon className="size-3.5 animate-spin" />
                    ) : (
                      <FilmIcon className="size-3.5" />
                    )}
                  </span>

                  <div className="min-w-0 flex-1 space-y-1.5">
                    <p className="line-clamp-2 text-sm leading-snug font-medium text-white/80 group-hover:text-white">
                      {job.topic ?? job.title ?? job.job_id}
                    </p>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span
                        className={cn(
                          'rounded-full border px-1.5 py-px text-[10px] font-medium',
                          chip.className,
                        )}
                      >
                        {chip.label}
                      </span>
                      {/* The backend's JobStatusOut carries no timestamp, so
                          this only renders if one is ever added. */}
                      {job.created_at !== null && (
                        <span className="text-[10px] text-white/30">
                          {formatRelativeTime(job.created_at)}
                        </span>
                      )}
                      {job.status !== 'done' && job.status !== 'failed' && (
                        <span className="text-[10px] tabular-nums text-white/30">
                          {job.progress}%
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
