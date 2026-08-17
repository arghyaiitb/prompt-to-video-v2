import { useCallback, useEffect, useState } from 'react'

import { ApiError, fetchJob } from '@/lib/api'
import { isTerminal, type Job } from '@/lib/types'

const POLL_INTERVAL_MS = 1500
/** Transient network blips shouldn't kill a long render. */
const MAX_CONSECUTIVE_ERRORS = 5

export interface JobPollingState {
  job: Job | null
  /** True until the first response for the current job id arrives. */
  isLoading: boolean
  /** Set only when polling has given up. */
  error: string | null
  /** Forces an immediate refetch. */
  refresh: () => void
}

/**
 * Polls `GET /api/jobs/{id}` every 1.5s.
 *
 * Stops on terminal states (`done` / `failed`), on repeated failures, and on
 * unmount. Passing `null` disables polling entirely.
 */
export function useJobPolling(jobId: string | null): JobPollingState {
  const [job, setJob] = useState<Job | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  const refresh = useCallback(() => {
    setNonce((value) => value + 1)
  }, [])

  useEffect(() => {
    if (jobId === null) {
      setJob(null)
      setIsLoading(false)
      setError(null)
      return
    }

    // `cancelled` guards against a late response from a previous job id
    // overwriting state after the effect has been torn down.
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let failures = 0

    setIsLoading(true)
    setError(null)

    const stop = (): void => {
      if (timer !== undefined) {
        clearTimeout(timer)
        timer = undefined
      }
    }

    const tick = async (): Promise<void> => {
      try {
        const next = await fetchJob(jobId)
        if (cancelled) return

        failures = 0
        setJob(next)
        setIsLoading(false)
        setError(null)

        if (isTerminal(next.status)) {
          stop()
          return
        }
      } catch (cause) {
        if (cancelled) return

        // A 404 will never resolve itself — give up immediately.
        if (cause instanceof ApiError && cause.status === 404) {
          setIsLoading(false)
          setError('That job no longer exists.')
          stop()
          return
        }

        failures += 1
        if (failures >= MAX_CONSECUTIVE_ERRORS) {
          setIsLoading(false)
          setError(cause instanceof Error ? cause.message : 'Lost contact with the server.')
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
  }, [jobId, nonce])

  return { job, isLoading, error, refresh }
}
