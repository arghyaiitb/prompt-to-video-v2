import { useEffect, useState } from 'react'

import { FALLBACK_THEMES, fetchThemes } from '@/lib/api'
import type { ThemePreset } from '@/lib/types'

export interface ThemesState {
  themes: ThemePreset[]
  isLoading: boolean
  /** True when `/api/themes` was unreachable and the built-in snapshot is in use. */
  usedFallback: boolean
}

/**
 * Loads the theme catalogue once, degrading to the built-in snapshot.
 *
 * Mirrors `useVoices`: the picker must be usable before the endpoint exists, so
 * a failure is a state to report, not an error to throw.
 */
export function useThemes(): ThemesState {
  const [themes, setThemes] = useState<ThemePreset[]>(FALLBACK_THEMES)
  const [isLoading, setIsLoading] = useState(true)
  const [usedFallback, setUsedFallback] = useState(false)

  useEffect(() => {
    let cancelled = false

    void fetchThemes().then((result) => {
      if (cancelled) return
      setThemes(result.themes)
      setUsedFallback(result.usedFallback)
      setIsLoading(false)
    })

    return () => {
      cancelled = true
    }
  }, [])

  return { themes, isLoading, usedFallback }
}
