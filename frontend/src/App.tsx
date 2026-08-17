import { useCallback, useEffect, useState } from 'react'
import { FilmIcon, Loader2Icon } from 'lucide-react'
import { toast } from 'sonner'

import { CreateForm } from '@/components/CreateForm'
import { JobHistory } from '@/components/JobHistory'
import { ProgressView } from '@/components/ProgressView'
import { ResultView } from '@/components/ResultView'
import { Skeleton } from '@/components/ui/skeleton'
import { Toaster } from '@/components/ui/sonner'
import { useJobHistory } from '@/hooks/useJobHistory'
import { useJobPolling } from '@/hooks/useJobPolling'
import { useVoices } from '@/hooks/useVoices'
import { createJob } from '@/lib/api'
import type { CreateJobRequest } from '@/lib/types'

export default function App() {
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const { voices, isLoading: voicesLoading, usedFallback } = useVoices()
  const { job, isLoading: jobLoading, error: pollError, refresh } = useJobPolling(activeJobId)
  const { jobs, isLoading: historyLoading, error: historyError, reload } = useJobHistory()

  // Keep the history list honest as the active job moves through the pipeline.
  useEffect(() => {
    if (job !== null && (job.status === 'done' || job.status === 'failed')) reload()
  }, [job?.status, job, reload])

  const handleSubmit = useCallback(
    (request: CreateJobRequest) => {
      setIsSubmitting(true)
      void createJob(request)
        .then((response) => {
          setActiveJobId(response.job_id)
          reload()
        })
        .catch((cause: unknown) => {
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
                  voices={voices}
                  voicesLoading={voicesLoading}
                  usedFallbackVoices={usedFallback}
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
                <ResultView job={job} onReset={handleReset} />
              ) : (
                <ProgressView
                  job={job}
                  pollError={pollError}
                  onRetryPoll={refresh}
                  onReset={handleReset}
                />
              )}
            </div>
          </main>

          {/* Sidebar ---------------------------------------------------- */}
          <aside className="lg:sticky lg:top-16 lg:self-start">
            <JobHistory
              jobs={jobs}
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
