import { CheckIcon, Loader2Icon, XIcon } from 'lucide-react'

import { cn } from '@/lib/utils'
import {
  PIPELINE_STAGES,
  STAGE_BLURBS,
  STAGE_LABELS,
  type JobStatus,
  type PipelineStage,
} from '@/lib/types'

type StageState = 'complete' | 'active' | 'failed' | 'pending'

interface StageStepperProps {
  status: JobStatus
  currentStage: PipelineStage | null
}

/** Index of the stage the job is sitting on, or -1 if unknown. */
function activeIndex(status: JobStatus, currentStage: PipelineStage | null): number {
  if (status === 'done') return PIPELINE_STAGES.length - 1
  const stage = currentStage ?? (status === 'failed' ? null : status)
  if (stage === null) return -1
  return PIPELINE_STAGES.indexOf(stage)
}

function stateFor(index: number, active: number, status: JobStatus): StageState {
  if (status === 'failed') {
    if (index === active) return 'failed'
    return index < active ? 'complete' : 'pending'
  }
  if (status === 'done') return 'complete'
  if (index < active) return 'complete'
  if (index === active) return 'active'
  return 'pending'
}

const DOT_STYLES: Record<StageState, string> = {
  complete: 'border-emerald-400/40 bg-emerald-400/15 text-emerald-300',
  active: 'border-violet-400/60 bg-violet-500/20 text-violet-200 shadow-[0_0_0_4px_rgba(139,92,246,0.12)]',
  failed: 'border-red-400/50 bg-red-500/20 text-red-300',
  pending: 'border-white/10 bg-white/[0.03] text-white/30',
}

const LABEL_STYLES: Record<StageState, string> = {
  complete: 'text-white/70',
  active: 'text-white',
  failed: 'text-red-200',
  pending: 'text-white/35',
}

/** Vertical timeline of pipeline stages with the current one highlighted. */
export function StageStepper({ status, currentStage }: StageStepperProps) {
  const active = activeIndex(status, currentStage)

  return (
    <ol className="relative space-y-1">
      {PIPELINE_STAGES.map((stage, index) => {
        const state = stateFor(index, active, status)
        const isLast = index === PIPELINE_STAGES.length - 1

        return (
          <li key={stage} className="relative flex gap-4 pb-1">
            {/* Connector rail, drawn behind the dot. */}
            {!isLast && (
              <span
                aria-hidden
                className={cn(
                  'absolute left-[15px] top-8 h-[calc(100%-1rem)] w-px transition-colors duration-500',
                  state === 'complete' ? 'bg-emerald-400/30' : 'bg-white/10',
                )}
              />
            )}

            <span
              className={cn(
                'relative z-10 flex size-8 shrink-0 items-center justify-center rounded-full border transition-all duration-500',
                DOT_STYLES[state],
              )}
            >
              {state === 'complete' && <CheckIcon className="size-4" strokeWidth={2.5} />}
              {state === 'active' && <Loader2Icon className="size-4 animate-spin" />}
              {state === 'failed' && <XIcon className="size-4" strokeWidth={2.5} />}
              {state === 'pending' && (
                <span className="size-1.5 rounded-full bg-current" aria-hidden />
              )}
            </span>

            <div className="min-w-0 pt-1 pb-3">
              <p
                className={cn(
                  'text-sm font-medium transition-colors duration-300',
                  LABEL_STYLES[state],
                )}
              >
                {STAGE_LABELS[stage]}
              </p>
              {state === 'active' && (
                <p className="mt-0.5 text-xs text-white/45">{STAGE_BLURBS[stage]}</p>
              )}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
