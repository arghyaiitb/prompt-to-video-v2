import { useEffect, useState } from 'react'

import { FALLBACK_VOICES, fetchVoices } from '@/lib/api'
import type { Voice } from '@/lib/types'

export interface VoicesState {
  voices: Voice[]
  isLoading: boolean
  /** True when `/api/voices` was unreachable and the built-in list is in use. */
  usedFallback: boolean
}

/** Loads the voice catalogue once, degrading to the verified built-in list. */
export function useVoices(): VoicesState {
  const [voices, setVoices] = useState<Voice[]>(FALLBACK_VOICES)
  const [isLoading, setIsLoading] = useState(true)
  const [usedFallback, setUsedFallback] = useState(false)

  useEffect(() => {
    let cancelled = false

    void fetchVoices().then((result) => {
      if (cancelled) return
      setVoices(result.voices)
      setUsedFallback(result.usedFallback)
      setIsLoading(false)
    })

    return () => {
      cancelled = true
    }
  }, [])

  return { voices, isLoading, usedFallback }
}
