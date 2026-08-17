import { useCallback, useEffect, useRef, useState } from 'react'

import { fetchJobs } from '@/lib/api'
import type { Job } from '@/lib/types'

export interface JobHistoryState {
  jobs: Job[]
  isLoading: boolean
  /** Non-null when the history list could not be loaded. */
  error: string | null
  reload: () => void
}

/** Loads recent jobs from `GET /api/jobs`. Call `reload` after mutations. */
export function useJobHistory(): JobHistoryState {
  const [jobs, setJobs] = useState<Job[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const reload = useCallback(() => {
    setIsLoading(true)
    void fetchJobs()
      .then((result) => {
        if (!mounted.current) return
        setJobs(result)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (!mounted.current) return
        setJobs([])
        setError(cause instanceof Error ? cause.message : 'Could not load history.')
      })
      .finally(() => {
        if (mounted.current) setIsLoading(false)
      })
  }, [])

  useEffect(() => {
    reload()
  }, [reload])

  return { jobs, isLoading, error, reload }
}
