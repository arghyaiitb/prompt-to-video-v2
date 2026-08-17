import { AlertTriangleIcon, Loader2Icon, RotateCcwIcon } from 'lucide-react'

import { StageStepper } from '@/components/StageStepper'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { STAGE_LABELS, type Job } from '@/lib/types'

interface ProgressViewProps {
  job: Job
  /** Set when polling itself failed (as opposed to the job failing). */
  pollError: string | null
  onRetryPoll: () => void
  onReset: () => void
}

export function ProgressView({ job, pollError, onRetryPoll, onReset }: ProgressViewProps) {
  const failed = job.status === 'failed'
  const stageLabel =
    job.current_stage !== null ? STAGE_LABELS[job.current_stage] : 'Waiting for the pipeline'

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <div className="flex items-center gap-2">
          {failed ? (
            <span className="flex items-center gap-1.5 rounded-full border border-red-400/30 bg-red-500/10 px-2.5 py-1 text-xs font-medium text-red-200">
              <AlertTriangleIcon className="size-3" />
              Failed
            </span>
          ) : (
            <span className="flex items-center gap-1.5 rounded-full border border-violet-400/30 bg-violet-500/10 px-2.5 py-1 text-xs font-medium text-violet-200">
              <Loader2Icon className="size-3 animate-spin" />
              {stageLabel}
            </span>
          )}
          <span className="font-mono text-xs text-white/25">{job.job_id}</span>
        </div>

        {job.topic !== null && (
          <h2 className="text-xl leading-snug font-semibold text-white/90">{job.topic}</h2>
        )}
      </header>

      {/* Progress bar ---------------------------------------------------- */}
      <div className="space-y-2">
        <div className="flex items-end justify-between">
          <span className="text-xs tracking-wide text-white/40 uppercase">Progress</span>
          <span className="font-mono text-2xl font-semibold text-white tabular-nums">
            {job.progress}
            <span className="text-sm text-white/30">%</span>
          </span>
        </div>
        <Progress
          value={job.progress}
          className="h-2 bg-white/[0.06] [&>[data-slot=progress-indicator]]:bg-gradient-to-r [&>[data-slot=progress-indicator]]:from-violet-500 [&>[data-slot=progress-indicator]]:to-indigo-400"
        />
      </div>

      {/* Error surface --------------------------------------------------- */}
      {failed && (
        <div className="space-y-3 rounded-xl border border-red-400/25 bg-red-500/[0.07] p-4">
          <p className="text-sm font-medium text-red-200">This job failed.</p>
          <p className="font-mono text-xs leading-relaxed break-words whitespace-pre-wrap text-red-200/70">
            {job.error ?? 'The backend did not report a reason.'}
          </p>
          <Button
            variant="outline"
            size="sm"
            onClick={onReset}
            className="border-red-400/30 bg-transparent text-red-100 hover:bg-red-500/10 hover:text-white"
          >
            <RotateCcwIcon className="size-3.5" />
            Start over
          </Button>
        </div>
      )}

      {pollError !== null && !failed && (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-amber-400/25 bg-amber-500/[0.07] p-4">
          <p className="text-sm text-amber-100/80">{pollError}</p>
          <Button
            variant="outline"
            size="sm"
            onClick={onRetryPoll}
            className="shrink-0 border-amber-400/30 bg-transparent text-amber-100 hover:bg-amber-500/10 hover:text-white"
          >
            <RotateCcwIcon className="size-3.5" />
            Retry
          </Button>
        </div>
      )}

      {/* Stepper --------------------------------------------------------- */}
      <div className="rounded-xl border border-white/[0.07] bg-white/[0.015] p-5">
        <p className="mb-4 text-xs tracking-wide text-white/35 uppercase">Pipeline</p>
        <StageStepper status={job.status} currentStage={job.current_stage} />
      </div>
    </div>
  )
}
