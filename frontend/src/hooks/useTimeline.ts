import { useCallback, useEffect, useRef, useState } from 'react'

import { fetchTimeline } from '@/lib/api'
import { isTerminal, timelineCouldExist, type JobStatus, type Timeline } from '@/lib/types'

/** Slower than the status poll: the artifact only changes between stages. */
const POLL_INTERVAL_MS = 2500
const MAX_CONSECUTIVE_ERRORS = 4

export interface TimelineState {
  timeline: Timeline | null
  /** True until the first answer of any kind for the current job id. */
  isLoading: boolean
  /**
   * The artifact does not exist yet (404 / no scenes). Expected while the
   * script is still being written — render an explanatory state, not an error.
   */
  isPending: boolean
  /** Set only when fetching failed repeatedly. */
  error: string | null
  refresh: () => void
}

/**
 * Polls `GET /api/jobs/{id}/timeline` alongside the status poll.
 *
 * Disabled while the job is `queued` (nothing can exist yet) and stops once a
 * terminal status has produced one successful read, so a finished job settles
 * on a single fetch. `status` is taken as a primitive rather than the whole
 * job so the effect restarts on stage changes only — each transition is
 * exactly when a fresh read is worth doing.
 */
export function useTimeline(jobId: string | null, status: JobStatus | null): TimelineState {
  const [timeline, setTimeline] = useState<Timeline | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)
  const lastJobId = useRef<string | null>(null)

  const refresh = useCallback(() => {
    setNonce((value) => value + 1)
  }, [])

  useEffect(() => {
    // Switching jobs must not show the previous job's scenes for a frame.
    if (lastJobId.current !== jobId) {
      lastJobId.current = jobId
      setTimeline(null)
      setError(null)
    }

    if (jobId === null || !timelineCouldExist(status)) {
      setIsLoading(false)
      return
    }

    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let failures = 0

    setIsLoading(true)

    const stop = (): void => {
      if (timer !== undefined) {
        clearTimeout(timer)
        timer = undefined
      }
    }

    const tick = async (): Promise<void> => {
      try {
        const next = await fetchTimeline(jobId)
        if (cancelled) return

        failures = 0
        setIsLoading(false)
        setError(null)
        // A null/empty read on a job that already had scenes is ignored:
        // never regress a populated inspector to its empty state.
        if (next !== null && next.scenes.length > 0) setTimeline(next)

        if (status !== null && isTerminal(status)) {
          // Finished job: one good read is all there is to get.
          if (next !== null || status === 'failed') {
            stop()
            return
          }
        }
      } catch (cause) {
        if (cancelled) return

        failures += 1
        if (failures >= MAX_CONSECUTIVE_ERRORS) {
          setIsLoading(false)
          setError(cause instanceof Error ? cause.message : 'Could not load the scene breakdown.')
          stop()
          return
        }
      }

      if (!cancelled) {
        timer = setTimeout(() => void tick(), POLL_INTERVAL_MS)
      }
    }

    void tick()

    return () => {
      cancelled = true
      stop()
    }
  }, [jobId, status, nonce])

  const isPending =
    !isLoading && error === null && (timeline === null || timeline.scenes.length === 0)

  return { timeline, isLoading, isPending, error, refresh }
}
