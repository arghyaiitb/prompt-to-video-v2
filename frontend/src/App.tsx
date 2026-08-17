import { useCallback, useEffect, useState } from 'react'
import { FilmIcon, Loader2Icon } from 'lucide-react'
import { toast } from 'sonner'

import { CreateForm } from '@/components/CreateForm'
import { JobHistory } from '@/components/JobHistory'
import { ProgressView } from '@/components/ProgressView'
import { ResultView } from '@/components/ResultView'
import { Skeleton } from '@/components/ui/skeleton'
import { Toaster } from '@/components/ui/sonner'
import { useEngines } from '@/hooks/useEngines'
import { useJobHistory } from '@/hooks/useJobHistory'
import { useJobPolling } from '@/hooks/useJobPolling'
import { useThemes } from '@/hooks/useThemes'
import { useTimeline } from '@/hooks/useTimeline'
import { useVoices } from '@/hooks/useVoices'
import { createJob, ThemeContrastError, VoiceEngineMismatchError } from '@/lib/api'
import { OPTIONAL_FIELD_LABELS, type CreateJobRequest, type ThemeContrastFailure } from '@/lib/types'

/** `?job=<id>` opens straight into that job — handy for sharing a render. */
function initialJobId(): string | null {
  const value = new URLSearchParams(window.location.search).get('job')?.trim() ?? ''
  return value === '' ? null : value
}

export default function App() {
  const [activeJobId, setActiveJobId] = useState<string | null>(initialJobId)
  const [isSubmitting, setIsSubmitting] = useState(false)
  /**
   * The server's verdict on a custom palette. Held here rather than in the form
   * because it arrives with the response to a submission, and it carries the
   * corrected palette the "Fix contrast" button applies.
   */
  const [contrastFailure, setContrastFailure] = useState<ThemeContrastFailure | null>(null)

  // The engine drives the voice request, so it is resolved before `useVoices`.
  const {
    engines,
    isLoading: enginesLoading,
    usedFallback: usedFallbackEngines,
    selectedId: engineId,
    select: selectEngine,
  } = useEngines()
  const {
    voices,
    isLoading: voicesLoading,
    usedFallback,
    engineMismatch: voicesEngineMismatch,
  } = useVoices(engineId)
  const {
    themes,
    isLoading: themesLoading,
    usedFallback: usedFallbackThemes,
  } = useThemes()
  const { job, isLoading: jobLoading, error: pollError, refresh } = useJobPolling(activeJobId)
  const { jobs, isLoading: historyLoading, error: historyError, reload } = useJobHistory()

  // Polled alongside the status: the scene breakdown is the useful thing to
  // watch while a render grinds through eight stages.
  const {
    timeline,
    isLoading: timelineLoading,
    isPending: timelinePending,
    error: timelineError,
    refresh: refreshTimeline,
  } = useTimeline(activeJobId, job?.status ?? null)

  // Mirror the selection into the URL (replace, not push: the back button
  // should leave the app rather than walk a trail of job ids).
  useEffect(() => {
    const url = new URL(window.location.href)
    if (activeJobId === null) url.searchParams.delete('job')
    else url.searchParams.set('job', activeJobId)
    window.history.replaceState(null, '', url)
  }, [activeJobId])

  // Keep the history list honest as the active job moves through the pipeline.
  useEffect(() => {
    if (job !== null && (job.status === 'done' || job.status === 'failed')) reload()
  }, [job?.status, job, reload])

  const handleSubmit = useCallback(
    (request: CreateJobRequest) => {
      setIsSubmitting(true)
      setContrastFailure(null)
      void createJob(request)
        .then((response) => {
          setActiveJobId(response.job_id)
          reload()

          // The backend accepted the job only after we dropped the newer
          // fields — say so rather than silently ignoring the settings.
          // Accepted, but the palette sits between AA and our recommendation.
          if (response.themeWarnings.length > 0) {
            toast.warning('Palette accepted with a caveat', {
              description: response.themeWarnings.join(' '),
            })
          }

          if (response.droppedFields.length > 0) {
            const names = response.droppedFields.map(
              (field) => OPTIONAL_FIELD_LABELS[field as keyof typeof OPTIONAL_FIELD_LABELS] ?? field,
            )
            toast.warning('Some settings were ignored', {
              description: `This backend does not accept ${names.join(', ')} yet. The video is being made with the remaining settings.`,
            })
          }
        })
        .catch((cause: unknown) => {
          // The contrast gate is not a generic failure: it comes back with the
          // failing pairs and a corrected palette, so it belongs in the form
          // next to the colours rather than in a toast that scrolls away.
          if (cause instanceof ThemeContrastError) {
            setContrastFailure(cause.failure)
            toast.error('That palette is not readable on screen', {
              description:
                cause.failure.failures[0] ??
                'Use "Fix contrast" to apply the nearest palette that passes.',
            })
            return
          }
          // The voice/engine pair is derived, so this should be unreachable —
          // it means the UI and server disagree about which engine owns a
          // voice. Named explicitly so that shows up as a bug, not as noise.
          if (cause instanceof VoiceEngineMismatchError) {
            toast.error('That voice does not belong to the selected engine', {
              description: cause.mismatch.message,
            })
            return
          }
          toast.error('Could not start the video', {
            description: cause instanceof Error ? cause.message : 'Unknown error.',
          })
        })
        .finally(() => {
          setIsSubmitting(false)
        })
    },
    [reload],
  )

  const dismissContrastFailure = useCallback(() => {
    setContrastFailure(null)
  }, [])

  const handleReset = useCallback(() => {
    setActiveJobId(null)
    reload()
  }, [reload])

  const handleSelect = useCallback((jobId: string) => {
    setActiveJobId(jobId)
  }, [])

  return (
    <div className="dark relative min-h-screen bg-neutral-950 text-white antialiased">
      {/* Ambient backdrop */}
      <div aria-hidden className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 h-[32rem] w-[52rem] -translate-x-1/2 rounded-full bg-violet-600/15 blur-[120px]" />
        <div className="absolute top-1/3 -right-40 h-[26rem] w-[26rem] rounded-full bg-indigo-600/10 blur-[110px]" />
        <div className="absolute bottom-0 -left-32 h-[22rem] w-[22rem] rounded-full bg-fuchsia-600/[0.07] blur-[110px]" />
      </div>

      <div className="relative mx-auto max-w-6xl px-6 py-12 lg:py-16">
        {/* Header ------------------------------------------------------- */}
        <header className="mb-12 space-y-4">
          <div className="flex items-center gap-3">
            <span className="flex size-10 items-center justify-center rounded-xl border border-white/10 bg-gradient-to-br from-violet-500/25 to-indigo-500/10 shadow-lg shadow-violet-950/30">
              <FilmIcon className="size-5 text-violet-200" />
            </span>
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-white">Topic to Video</h1>
              <p className="text-xs text-white/40">
                Narrated, scored and rendered from a single prompt
              </p>
            </div>
          </div>
        </header>

        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem] lg:gap-10">
          {/* Main panel ------------------------------------------------- */}
          <main>
            <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-6 shadow-2xl shadow-black/40 backdrop-blur-sm sm:p-8">
              {activeJobId === null ? (
                <CreateForm
                  engines={engines}
                  enginesLoading={enginesLoading}
                  usedFallbackEngines={usedFallbackEngines}
                  engineId={engineId}
                  onSelectEngine={selectEngine}
                  voices={voices}
                  voicesLoading={voicesLoading}
                  usedFallbackVoices={usedFallback}
                  voicesEngineMismatch={voicesEngineMismatch}
                  themes={themes}
                  themesLoading={themesLoading}
                  usedFallbackThemes={usedFallbackThemes}
                  contrastFailure={contrastFailure}
                  onDismissContrastFailure={dismissContrastFailure}
                  isSubmitting={isSubmitting}
                  onSubmit={handleSubmit}
                />
              ) : job === null ? (
                jobLoading ? (
                  <div className="space-y-6">
                    <Skeleton className="h-6 w-2/3 rounded bg-white/5" />
                    <Skeleton className="h-2 w-full rounded bg-white/5" />
                    <Skeleton className="h-64 w-full rounded-xl bg-white/5" />
                    <p className="flex items-center justify-center gap-2 text-xs text-white/30">
                      <Loader2Icon className="size-3 animate-spin" />
                      Loading job…
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4 py-8 text-center">
                    <p className="text-sm text-white/60">
                      {pollError ?? 'That job could not be loaded.'}
                    </p>
                    <button
                      type="button"
                      onClick={handleReset}
                      className="text-sm text-violet-300 underline underline-offset-4 hover:text-violet-200"
                    >
                      Back to the form
                    </button>
                  </div>
                )
              ) : job.status === 'done' ? (
                <ResultView
                  job={job}
                  themes={themes}
                  engines={engines}
                  timeline={timeline}
                  timelineLoading={timelineLoading}
                  timelinePending={timelinePending}
                  timelineError={timelineError}
                  onRetryTimeline={refreshTimeline}
                  onReset={handleReset}
                />
              ) : (
                <ProgressView
                  job={job}
                  pollError={pollError}
                  onRetryPoll={refresh}
                  onReset={handleReset}
                  timeline={timeline}
                  timelineLoading={timelineLoading}
                  timelinePending={timelinePending}
                  timelineError={timelineError}
                  onRetryTimeline={refreshTimeline}
                />
              )}
            </div>
          </main>

          {/* Sidebar ---------------------------------------------------- */}
          <aside className="lg:sticky lg:top-16 lg:self-start">
            <JobHistory
              jobs={jobs}
              themes={themes}
              engines={engines}
              isLoading={historyLoading}
              error={historyError}
              activeJobId={activeJobId}
              onSelect={handleSelect}
              onReload={reload}
            />
          </aside>
        </div>
      </div>

      <Toaster position="bottom-right" />
    </div>
  )
}
