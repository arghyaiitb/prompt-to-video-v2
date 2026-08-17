import { useCallback, useEffect, useState } from 'react'

import { FALLBACK_ENGINES, fetchEngines, preferredEngineId } from '@/lib/api'
import { DEFAULT_ENGINE_ID, type SpeechEngine } from '@/lib/types'

export interface EnginesState {
  engines: SpeechEngine[]
  isLoading: boolean
  /** True when `/api/engines` was unreachable and the built-in list is in use. */
  usedFallback: boolean
  /** Currently selected engine id — always one of `engines`. */
  selectedId: string
  selected: SpeechEngine | null
  select: (engineId: string) => void
}

/**
 * Loads the speech-engine catalogue and owns the selection.
 *
 * The selection lives here rather than in the form because the voice list is
 * fetched from it: `useVoices(engineId)` has to see the change in the same
 * render the user makes it. Mirrors `useVoices`/`useThemes` in degrading to a
 * built-in list instead of throwing — `/api/engines` may not exist yet.
 */
export function useEngines(): EnginesState {
  const [engines, setEngines] = useState<SpeechEngine[]>(FALLBACK_ENGINES)
  const [isLoading, setIsLoading] = useState(true)
  const [usedFallback, setUsedFallback] = useState(false)
  const [selectedId, setSelectedId] = useState<string>(() => preferredEngineId(FALLBACK_ENGINES))

  useEffect(() => {
    let cancelled = false

    void fetchEngines().then((result) => {
      if (cancelled) return
      setEngines(result.engines)
      setUsedFallback(result.usedFallback)
      setIsLoading(false)
      // Re-resolve against the real catalogue: the id chosen from the fallback
      // list may not be offered, or may have turned out to be unavailable.
      setSelectedId((current) => {
        const match = result.engines.find((engine) => engine.id === current)
        return match !== undefined && match.available !== false
          ? current
          : preferredEngineId(result.engines)
      })
    })

    return () => {
      cancelled = true
    }
  }, [])

  const select = useCallback(
    (engineId: string) => {
      // An engine with no credentials is not a valid destination; the selector
      // disables it, and this is the backstop for a keyboard or programmatic hit.
      const target = engines.find((engine) => engine.id === engineId)
      if (target === undefined || target.available === false) return
      setSelectedId(engineId)
    },
    [engines],
  )

  const selected = engines.find((engine) => engine.id === selectedId) ?? null

  return {
    engines,
    isLoading,
    usedFallback,
    selectedId: selected?.id ?? DEFAULT_ENGINE_ID,
    selected,
    select,
  }
}
