import { useEffect, useState } from 'react'

import { fallbackVoicesFor, fetchVoices } from '@/lib/api'
import type { Voice } from '@/lib/types'

export interface VoicesState {
  voices: Voice[]
  isLoading: boolean
  /** True when `/api/voices` was unreachable and the built-in list is in use. */
  usedFallback: boolean
  /**
   * True when the server answered but returned another engine's voices, so the
   * built-in catalogue is standing in. The endpoint is up — the `engine`
   * parameter is not honoured yet.
   */
  engineMismatch: boolean
}

/**
 * Loads the voice catalogue for one engine, degrading to the built-in list.
 *
 * Refetches whenever the engine changes. `voices` switches to that engine's
 * fallback catalogue *synchronously* on the change rather than holding the
 * previous engine's list while the request is in flight: the caller derives the
 * selected voice from this list, and a Deepgram voice must never be sitting in
 * the form while Polly is selected.
 */
export function useVoices(engineId?: string): VoicesState {
  // `fallbackVoicesFor` answers the Deepgram list for an unknown key, which is
  // the right catalogue for the engine-less call an older backend expects.
  const [state, setState] = useState<{ engineId: string | undefined; value: VoicesState }>({
    engineId,
    value: {
      voices: fallbackVoicesFor(engineId ?? ''),
      isLoading: true,
      usedFallback: false,
      engineMismatch: false,
    },
  })

  useEffect(() => {
    let cancelled = false

    void fetchVoices(engineId).then((result) => {
      if (cancelled) return
      setState({
        engineId,
        value: {
          voices: result.voices,
          isLoading: false,
          usedFallback: result.usedFallback,
          engineMismatch: result.engineMismatch,
        },
      })
    })

    return () => {
      cancelled = true
    }
  }, [engineId])

  // Rendered before the effect for a new engine has resolved: report that
  // engine's built-in list, not the one belonging to the engine just left.
  if (state.engineId !== engineId) {
    return {
      voices: fallbackVoicesFor(engineId ?? ''),
      isLoading: true,
      usedFallback: false,
      engineMismatch: false,
    }
  }

  return state.value
}
