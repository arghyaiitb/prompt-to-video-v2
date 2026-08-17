import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import {
  AlertTriangleIcon,
  CheckIcon,
  InfoIcon,
  Loader2Icon,
  UploadIcon,
  XIcon,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'
import {
  formatBytes,
  inspectLogoFile,
  isUploadable,
  LOGO_ACCEPT_ATTRIBUTE,
  LOGO_RENDER_HEIGHT,
  MAX_LOGO_BYTES,
  MAX_LOGO_DIMENSION,
  type LogoInspection,
} from '@/lib/logo'

interface LogoUploaderProps {
  /** False when `/api/logos` is absent — the zone explains instead of accepting. */
  available: boolean
  /** 0..1 while the POST is in flight, `null` otherwise. */
  progress: number | null
  /** The last server-side failure, if any. */
  uploadError: string | null
  onUpload: (file: File) => Promise<unknown>
  /**
   * The inspected, not-yet-uploaded file — or `null`. Lifted so the picker can
   * preview it at render scale *before* it is sent anywhere, which is the whole
   * point: a 49px mark is judged locally, in a tenth of a second.
   */
  onPendingChange: (inspection: LogoInspection | null) => void
  className?: string
}

/**
 * Drag-and-drop (plus a file picker) for a brand mark.
 *
 * Everything checkable is checked in the browser first — type, byte size,
 * decoded dimensions, alpha channel, mean alpha, and for an SVG the presence of
 * constructs the render box's ImageMagick cannot reproduce. A 4MiB round trip to
 * be told "that is 6000px wide" is a worse experience than being told instantly,
 * and the limits are the server's own (`video_logo_max_bytes`,
 * `video_logo_max_dimension`) rather than invented here.
 */
export function LogoUploader({
  available,
  progress,
  uploadError,
  onUpload,
  onPendingChange,
  className,
}: LogoUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [inspecting, setInspecting] = useState(false)
  const [pending, setPending] = useState<LogoInspection | null>(null)

  const isUploading = progress !== null

  /** Object URLs are a leak if they are not revoked; the latest one is tracked. */
  const pendingUrl = useRef<string | null>(null)
  const releasePending = useCallback(() => {
    if (pendingUrl.current !== null) {
      URL.revokeObjectURL(pendingUrl.current)
      pendingUrl.current = null
    }
  }, [])
  useEffect(() => releasePending, [releasePending])

  const setPendingBoth = useCallback(
    (next: LogoInspection | null) => {
      setPending(next)
      onPendingChange(next)
    },
    [onPendingChange],
  )

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      const file = files?.[0]
      if (file === undefined) return
      releasePending()
      setPendingBoth(null)
      setInspecting(true)
      const inspection = await inspectLogoFile(file)
      pendingUrl.current = inspection.objectUrl
      setInspecting(false)
      setPendingBoth(inspection)
    },
    [releasePending, setPendingBoth],
  )

  const discard = useCallback(() => {
    releasePending()
    setPendingBoth(null)
    if (inputRef.current !== null) inputRef.current.value = ''
  }, [releasePending, setPendingBoth])

  const handleDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault()
    setIsDragging(false)
    if (!available || isUploading) return
    void handleFiles(event.dataTransfer.files)
  }

  const confirm = useCallback(async () => {
    if (pending === null || !isUploadable(pending)) return
    const result = await onUpload(pending.file)
    // Kept on failure so the warnings and the preview stay on screen to retry
    // from; the error is rendered beneath.
    if (result !== null) discard()
  }, [pending, onUpload, discard])

  return (
    <div className={cn('space-y-3', className)}>
      <div
        onDragOver={(event) => {
          event.preventDefault()
          if (available && !isUploading) setIsDragging(true)
        }}
        onDragLeave={() => {
          setIsDragging(false)
        }}
        onDrop={handleDrop}
        data-testid="logo-dropzone"
        data-dragging={isDragging}
        className={cn(
          'rounded-xl border border-dashed p-4 transition-colors',
          isDragging
            ? 'border-violet-400/60 bg-violet-500/[0.08]'
            : 'border-white/[0.12] bg-white/[0.015]',
          !available && 'opacity-60',
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={LOGO_ACCEPT_ATTRIBUTE}
          className="sr-only"
          data-testid="logo-file-input"
          disabled={!available || isUploading}
          onChange={(event) => {
            void handleFiles(event.target.files)
          }}
        />

        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] text-white/40">
            {inspecting || isUploading ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : (
              <UploadIcon className="size-4" />
            )}
          </span>

          <div className="min-w-0 flex-1 space-y-1">
            {available ? (
              <>
                <p className="text-sm text-white/70">
                  Drop a logo here, or{' '}
                  <button
                    type="button"
                    data-testid="logo-browse"
                    disabled={isUploading}
                    onClick={() => {
                      inputRef.current?.click()
                    }}
                    className="font-medium text-violet-300 underline underline-offset-4 hover:text-violet-200 disabled:opacity-50"
                  >
                    choose a file
                  </button>
                  .
                </p>
                <p className="text-xs text-white/35">
                  PNG with transparency, up to {formatBytes(MAX_LOGO_BYTES)} and{' '}
                  {MAX_LOGO_DIMENSION}px a side. SVG works but is rasterised on the server
                  — PNG is the safer choice.
                </p>
              </>
            ) : (
              <>
                <p className="text-sm text-white/70">Logo uploads are unavailable.</p>
                <p className="text-xs text-white/35" data-testid="logo-unavailable">
                  This backend has no <span className="font-mono">/api/logos</span> endpoint
                  yet, so videos are branded with the built-in mark. Everything else on this
                  form works as normal.
                </p>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Upload progress ---------------------------------------------- */}
      {isUploading && (
        <div className="space-y-1.5" data-testid="logo-progress">
          <Progress value={Math.round(progress * 100)} className="h-1.5 bg-white/[0.06]" />
          <p className="text-xs text-white/40 tabular-nums">
            Uploading… {Math.round(progress * 100)}%
          </p>
        </div>
      )}

      {uploadError !== null && (
        <p
          data-testid="logo-upload-error"
          className="flex items-start gap-2 rounded-xl border border-red-400/25 bg-red-500/[0.07] p-3 text-xs leading-relaxed text-red-200/90"
        >
          <AlertTriangleIcon className="mt-px size-3.5 shrink-0" />
          {uploadError}
        </p>
      )}

      {/* Pending file -------------------------------------------------- */}
      {pending !== null && (
        <div
          data-testid="logo-pending"
          className="space-y-3 rounded-xl border border-white/[0.08] bg-white/[0.02] p-3"
        >
          <div className="flex items-start gap-3">
            {/* Deliberately small: it is a file chip, not the preview. The
                honest preview is the framed one above the picker. */}
            <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-white/10 bg-white/[0.04]">
              <img src={pending.objectUrl} alt="" className="max-h-full max-w-full object-contain" />
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-white/80">{pending.file.name}</p>
              <p className="text-xs text-white/35 tabular-nums">
                {pending.width !== null && pending.height !== null
                  ? `${String(pending.width)}x${String(pending.height)} · `
                  : ''}
                {formatBytes(pending.file.size)}
                {pending.isSvg ? ' · SVG' : ''}
                {pending.hasAlpha === true ? ' · transparent' : ''}
                {pending.hasAlpha === false && !pending.isSvg ? ' · opaque' : ''}
              </p>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Discard this file"
              onClick={discard}
              disabled={isUploading}
              className="size-7 shrink-0 text-white/30 hover:bg-white/5 hover:text-white/70"
            >
              <XIcon className="size-3.5" />
            </Button>
          </div>

          {pending.errors.map((problem) => (
            <p
              key={problem.code}
              data-testid="logo-error"
              className="flex items-start gap-2 text-xs leading-relaxed text-red-200/90"
            >
              <AlertTriangleIcon className="mt-px size-3.5 shrink-0" />
              {problem.message}
            </p>
          ))}

          {pending.warnings.map((problem) => (
            <p
              key={problem.code}
              data-testid="logo-warning"
              className="flex items-start gap-2 text-xs leading-relaxed text-amber-300/80"
            >
              <InfoIcon className="mt-px size-3.5 shrink-0" />
              {problem.message}
            </p>
          ))}

          {isUploadable(pending) ? (
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                size="sm"
                data-testid="logo-confirm"
                disabled={isUploading}
                onClick={() => {
                  void confirm()
                }}
                className="bg-violet-600 text-white hover:bg-violet-500"
              >
                <CheckIcon className="size-3.5" />
                Use this logo
              </Button>
              <span className="text-[11px] text-white/30">
                The frame above shows it at {LOGO_RENDER_HEIGHT}px — check it before uploading.
              </span>
            </div>
          ) : (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => {
                inputRef.current?.click()
              }}
              className="border-white/15 bg-white/[0.03] text-white hover:bg-white/[0.08] hover:text-white"
            >
              Choose a different file
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
