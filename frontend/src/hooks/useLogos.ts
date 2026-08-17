import { useCallback, useEffect, useRef, useState } from 'react'

import {
  deleteLogo,
  fetchLogos,
  LogoUploadUnavailableError,
  uploadLogo,
} from '@/lib/api'
import {
  BUILT_IN_LOGO_ID,
  DEFAULT_NO_LOGO_VALUE,
  type BrandLogo,
  type LogoSelection,
} from '@/lib/types'

export interface LogosState {
  /** Previously uploaded marks, newest first. */
  logos: BrandLogo[]
  isLoading: boolean
  /**
   * False when `/api/logos` is not there. The built-in mark stays selected and
   * the uploader says so — a missing endpoint must not break the form.
   */
  available: boolean
  /** The server's spelling of "no brand mark", or our default. */
  noneValue: string
  /** `BUILT_IN_LOGO_ID`, `noneValue`, or an uploaded logo id. */
  selection: LogoSelection
  select: (selection: LogoSelection) => void
  /** 0..1 while an upload is in flight, `null` otherwise. */
  uploadProgress: number | null
  /** The last upload failure, already human-readable. */
  uploadError: string | null
  clearUploadError: () => void
  /** Resolves to the stored logo, which is also selected. Rejects never. */
  upload: (file: File) => Promise<BrandLogo | null>
  remove: (logoId: string) => Promise<void>
  reload: () => void
}

/**
 * Loads the uploaded-logo catalogue and owns the selection.
 *
 * Mirrors `useEngines`: a missing endpoint is a state to report, not an error to
 * throw, and the selection is re-resolved against the real catalogue once it
 * arrives so a stale id can never be submitted.
 *
 * The default selection is `BUILT_IN_LOGO_ID`, which sends no `logo_id` at all —
 * matching the backend, where an absent value means "use the configured mark".
 */
export function useLogos(): LogosState {
  const [logos, setLogos] = useState<BrandLogo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [available, setAvailable] = useState(false)
  const [noneValue, setNoneValue] = useState(DEFAULT_NO_LOGO_VALUE)
  const [selection, setSelection] = useState<LogoSelection>(BUILT_IN_LOGO_ID)
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [reloadToken, setReloadToken] = useState(0)

  /** Guards a state update after unmount, and lets `upload` see live values. */
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    void fetchLogos().then((result) => {
      if (cancelled) return
      setLogos(result.logos)
      setAvailable(result.available)
      setNoneValue(result.noneValue)
      setIsLoading(false)
      // A selected id the catalogue does not offer would be a 422 at submit
      // time; the built-in mark is the safe resolution. The no-logo value and
      // the built-in sentinel are always valid, whatever the list says.
      setSelection((current) => {
        if (current === BUILT_IN_LOGO_ID || current === result.noneValue) return current
        return result.logos.some((logo) => logo.id === current) ? current : BUILT_IN_LOGO_ID
      })
    })

    return () => {
      cancelled = true
    }
  }, [reloadToken])

  const select = useCallback((next: LogoSelection) => {
    setSelection(next)
    setUploadError(null)
  }, [])

  const reload = useCallback(() => {
    setIsLoading(true)
    setReloadToken((token) => token + 1)
  }, [])

  const clearUploadError = useCallback(() => {
    setUploadError(null)
  }, [])

  const upload = useCallback(async (file: File): Promise<BrandLogo | null> => {
    setUploadError(null)
    setUploadProgress(0)
    try {
      const logo = await uploadLogo(file, (fraction) => {
        if (mounted.current) setUploadProgress(fraction)
      })
      if (!mounted.current) return logo
      // Prepended rather than refetched: the list is ours to keep current, and a
      // refetch would drop the caller into a loading state right after success.
      setLogos((current) => [logo, ...current.filter((entry) => entry.id !== logo.id)])
      setAvailable(true)
      setSelection(logo.id)
      return logo
    } catch (cause) {
      if (mounted.current) {
        if (cause instanceof LogoUploadUnavailableError) {
          // The list request may have succeeded against an older route while the
          // upload route is missing; believe the failure.
          setAvailable(false)
          setUploadError(
            'This backend does not accept logo uploads yet. The built-in mark will be used.',
          )
        } else {
          setUploadError(cause instanceof Error ? cause.message : 'The upload failed.')
        }
      }
      return null
    } finally {
      if (mounted.current) setUploadProgress(null)
    }
  }, [])

  const remove = useCallback(
    async (logoId: string): Promise<void> => {
      try {
        await deleteLogo(logoId)
      } catch (cause) {
        if (mounted.current) {
          setUploadError(cause instanceof Error ? cause.message : 'Could not delete that logo.')
        }
        return
      }
      if (!mounted.current) return
      setLogos((current) => current.filter((logo) => logo.id !== logoId))
      // Deleting what was selected must not leave a dangling id in the form.
      setSelection((current) => (current === logoId ? BUILT_IN_LOGO_ID : current))
    },
    [],
  )

  return {
    logos,
    isLoading,
    available,
    noneValue,
    selection,
    select,
    uploadProgress,
    uploadError,
    clearUploadError,
    upload,
    remove,
    reload,
  }
}

/**
 * The `logo_id` to put in the request body for a selection.
 *
 * `undefined` for the built-in mark — the field is *omitted*, which is how the
 * backend is told to use its configured default. Anything else is sent verbatim.
 */
export function logoIdForRequest(selection: LogoSelection): string | undefined {
  return selection === BUILT_IN_LOGO_ID ? undefined : selection
}
