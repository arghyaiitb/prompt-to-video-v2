import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  AlertTriangleIcon,
  BanIcon,
  CheckIcon,
  InfoIcon,
  SparklesIcon,
  Trash2Icon,
} from 'lucide-react'

import { LogoUploader } from '@/components/LogoUploader'
import { SlidePreview, type PreviewLogo } from '@/components/SlidePreview'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { formatRatio, WCAG_LARGE_OBJECT, type Palette } from '@/lib/contrast'
import {
  BUILT_IN_LOGO_URL,
  BUILT_IN_MARK_COLOUR,
  evaluateLogoContrast,
  formatBytes,
  LOGO_HEIGHT_FRACTION,
  LOGO_RENDER_HEIGHT,
  sampleImageUrl,
  type LogoInspection,
} from '@/lib/logo'
import { cn } from '@/lib/utils'
import {
  BUILT_IN_LOGO_ID,
  type BrandLogo,
  type LogoRejection,
  type LogoSelection,
} from '@/lib/types'

interface LogoPickerProps {
  logos: BrandLogo[]
  isLoading: boolean
  available: boolean
  /** The server's spelling of "no brand mark". */
  noneValue: string
  selection: LogoSelection
  onSelect: (selection: LogoSelection) => void
  onRemove: (logoId: string) => void
  progress: number | null
  uploadError: string | null
  onUpload: (file: File) => Promise<unknown>
  /** The palette currently selected in the form — the preview's real ground. */
  palette: Palette
  themeName: string
  /** `Theme.logo_opacity` for that palette. */
  opacity: number
  /** A `logo_id` the server has refused. */
  rejection: LogoRejection | null
  /**
   * An inspected file that has not been uploaded yet.
   *
   * Owned by the form rather than here so the theme preview above shows the same
   * mark this one does — two frames disagreeing about what is on the slide would
   * be worse than no second frame at all.
   */
  pending: LogoInspection | null
  onPendingChange: (pending: LogoInspection | null) => void
  className?: string
}

/**
 * Choosing the brand mark, and seeing what it will actually look like.
 *
 * The preview is the reason this component exists. A logo is picked from a file
 * browser at 400px and composited into the video at 49px, 85% opaque, over
 * whatever ground the selected theme paints — and those are three separate ways
 * for it to fall apart. So the mark is never shown at its natural size: it is
 * shown in the frame at the renderer's scale, and again at exactly 49px, which is
 * the size it will be.
 */
export function LogoPicker({
  logos,
  isLoading,
  available,
  noneValue,
  selection,
  onSelect,
  onRemove,
  progress,
  uploadError,
  onUpload,
  palette,
  themeName,
  opacity,
  rejection,
  pending,
  onPendingChange,
  className,
}: LogoPickerProps) {
  const selectedLogo = logos.find((logo) => logo.id === selection) ?? null
  const isNone = selection === noneValue
  const isBuiltIn = selection === BUILT_IN_LOGO_ID

  /*
   * What the frame shows. A pending upload wins over the selection: the user is
   * looking at it to decide, and that decision has to be made before the bytes
   * are sent, not after.
   */
  const previewSrc: string | null = pending?.objectUrl ?? (
    isNone ? null : isBuiltIn ? BUILT_IN_LOGO_URL : (selectedLogo?.renderUrl ?? null)
  )

  const previewLogo: PreviewLogo | null = previewSrc === null ? null : { src: previewSrc, opacity }

  /*
   * The mark's dominant colour, measured from the pixels rather than assumed.
   *
   * With one deliberate exception: the built-in mark uses its documented `fill`.
   * `favicon.svg` declares the same purple twice — `#863bff` and then
   * `color(display-p3 .5252 .23 1)` — and a browser honours the second, which
   * clips to a noticeably more vivid sRGB value (sampling it here returns about
   * `#9A49FF`). ImageMagick has no display-p3, so it rasterises `#863BFF`, and
   * the number beside the preview has to be the one the *render* will produce.
   * PNG pixels have no such ambiguity, so uploads are measured.
   */
  const [sampledColour, setSampledColour] = useState<string | null>(null)
  useEffect(() => {
    if (pending !== null) {
      setSampledColour(pending.dominant)
      return
    }
    if (previewSrc === null) {
      setSampledColour(null)
      return
    }
    if (previewSrc === BUILT_IN_LOGO_URL) {
      setSampledColour(BUILT_IN_MARK_COLOUR)
      return
    }
    let cancelled = false
    void sampleImageUrl(previewSrc).then((sample) => {
      if (cancelled) return
      setSampledColour(sample?.dominant ?? null)
    })
    return () => {
      cancelled = true
    }
  }, [previewSrc, pending])

  const contrast = useMemo(
    () => (sampledColour === null ? null : evaluateLogoContrast(sampledColour, palette, opacity)),
    [sampledColour, palette, opacity],
  )

  const label = pending !== null
    ? pending.file.name
    : isNone
      ? 'No brand mark'
      : isBuiltIn
        ? 'Built-in mark'
        : (selectedLogo?.filename ?? selectedLogo?.id ?? '—')

  return (
    <div className={cn('space-y-4', className)}>
      {/* The truthful preview ------------------------------------------ */}
      <figure className="space-y-2">
        {/* `compact` on purpose: the full frame lives in the theme section
            above and also carries the mark. This one is here so the corner
            being judged is on screen while the file is being chosen. */}
        <div
          data-testid="logo-frame-preview"
          className="overflow-hidden rounded-xl border border-white/10"
        >
          <SlidePreview palette={palette} logo={previewLogo} compact />
        </div>

        <figcaption className="space-y-2 text-[11px] leading-relaxed text-white/30">
          <p data-testid="logo-preview-caption">
            <span className="text-white/45">{label}</span>
            {pending !== null && (
              <span className="ml-1 rounded bg-amber-500/15 px-1 py-px text-[10px] tracking-wide text-amber-200/90 uppercase">
                Not uploaded yet
              </span>
            )}{' '}
            —{' '}
            {previewLogo === null ? (
              <>
                No mark is composited — the bottom-left corner stays empty for the whole
                video.
              </>
            ) : (
              <>
                Bottom-left, {(LOGO_HEIGHT_FRACTION * 100).toFixed(1)}% of frame height (
                {LOGO_RENDER_HEIGHT}px at 1080p) at {Math.round(opacity * 100)}% opacity, over{' '}
                {themeName}. Constant for the whole video.
              </>
            )}
          </p>

          {/* Actual size. The framed mock above gets the *proportions* right but
              is drawn a few hundred pixels wide, so the mark there is smaller
              than it will be in the file. This strip is 1:1 for a 1080p render —
              the one place fine detail and small type can be judged. */}
          {previewLogo !== null && (
            <div className="space-y-1.5">
              <div
                data-testid="logo-actual-size"
                className="flex items-end gap-3 overflow-hidden rounded-lg border border-white/10 p-3"
                style={{ backgroundColor: palette.bg }}
              >
                <img
                  src={previewLogo.src}
                  alt=""
                  className="w-auto max-w-none object-contain"
                  style={{ height: `${String(LOGO_RENDER_HEIGHT)}px`, opacity }}
                />
                <span
                  className="mb-0.5 font-mono text-[10px] tracking-wide"
                  style={{ color: palette.muted }}
                >
                  {LOGO_RENDER_HEIGHT}px — actual size at 1080p
                </span>
              </div>
              <p className="text-white/30">
                This is the real size in the video. Fine detail and small text will not
                survive it — a symbol reads better than a full lockup.
              </p>
            </div>
          )}

          {/* Contrast, measured on the composite. Advisory only: a brand colour
              is not ours to refuse. */}
          {contrast !== null && (
            <p
              data-testid="logo-contrast"
              className={cn(
                'flex items-start gap-1.5',
                contrast.passes ? 'text-white/40' : 'text-amber-300/80',
              )}
            >
              {contrast.passes ? (
                <span
                  className="mt-px size-3 shrink-0 rounded-full ring-1 ring-white/20"
                  style={{ backgroundColor: contrast.composite }}
                />
              ) : (
                <AlertTriangleIcon className="mt-px size-3.5 shrink-0" />
              )}
              <span>
                <span className="font-mono tabular-nums">{formatRatio(contrast.ratio)}:1</span>{' '}
                against the {themeName} background, measured on the{' '}
                {Math.round(opacity * 100)}% composite ({contrast.composite}).{' '}
                {contrast.passes
                  ? `Clears the ${WCAG_LARGE_OBJECT.toFixed(1)}:1 WCAG bar for a non-text graphic.`
                  : `Under the ${WCAG_LARGE_OBJECT.toFixed(1)}:1 WCAG bar for a non-text graphic — the mark will be hard to make out on this theme. A lighter or darker version of it, or a different theme, fixes that.`}
              </span>
            </p>
          )}
        </figcaption>
      </figure>

      {/* A logo_id the server refused --------------------------------- */}
      {rejection !== null && (
        <p
          data-testid="logo-rejection"
          className="flex items-start gap-2 rounded-xl border border-red-400/25 bg-red-500/[0.07] p-3 text-xs leading-relaxed text-red-200/90"
        >
          <AlertTriangleIcon className="mt-px size-3.5 shrink-0" />
          <span>
            {rejection.message}{' '}
            {rejection.unsupported
              ? 'Switch to the built-in mark to start the video — nothing was created, and no other setting was changed.'
              : 'Pick another mark, or upload it again.'}{' '}
            The job was <span className="font-medium">not</span> started with a different
            logo.
          </span>
        </p>
      )}

      {/* Options ------------------------------------------------------- */}
      {isLoading ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {[0, 1].map((key) => (
            <Skeleton key={key} className="h-[62px] w-full rounded-xl bg-white/[0.04]" />
          ))}
        </div>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2" data-testid="logo-options">
          <OptionCard
            isSelected={isBuiltIn}
            onSelect={() => {
              onSelect(BUILT_IN_LOGO_ID)
            }}
            testId="logo-option-built-in"
            title="Built-in mark"
            hint="The app's own logo. What you get if you choose nothing."
            thumbnail={
              <img src={BUILT_IN_LOGO_URL} alt="" className="max-h-full max-w-full object-contain" />
            }
            palette={palette}
          />

          <OptionCard
            isSelected={isNone}
            onSelect={() => {
              onSelect(noneValue)
            }}
            testId="logo-option-none"
            title="No logo"
            hint="Ship the video unbranded."
            thumbnail={<BanIcon className="size-4 text-white/30" />}
            palette={palette}
          />

          {logos.map((logo) => (
            <OptionCard
              key={logo.id}
              isSelected={selection === logo.id}
              onSelect={() => {
                onSelect(logo.id)
              }}
              testId="logo-option-uploaded"
              title={logo.filename ?? logo.id}
              hint={[
                logo.width !== null && logo.height !== null
                  ? `${String(logo.width)}x${String(logo.height)}`
                  : null,
                logo.format,
                logo.size_bytes !== null ? formatBytes(logo.size_bytes) : null,
                logo.has_alpha === false ? 'no transparency' : null,
              ]
                .filter((part): part is string => part !== null && part !== '')
                .join(' · ')}
              warnings={logo.warnings}
              thumbnail={
                <img src={logo.renderUrl} alt="" className="max-h-full max-w-full object-contain" />
              }
              palette={palette}
              onRemove={() => {
                onRemove(logo.id)
              }}
            />
          ))}
        </div>
      )}

      {available && logos.length === 0 && !isLoading && (
        <p className="flex items-start gap-1.5 text-xs text-white/30">
          <SparklesIcon className="mt-px size-3.5 shrink-0" />
          No logos uploaded yet. The built-in mark is used until you add one.
        </p>
      )}

      <LogoUploader
        available={available}
        progress={progress}
        uploadError={uploadError}
        onUpload={onUpload}
        onPendingChange={onPendingChange}
      />
    </div>
  )
}

interface OptionCardProps {
  isSelected: boolean
  onSelect: () => void
  title: string
  hint: string
  /** The server's caveats about this specific file — `warnings` from the upload. */
  warnings?: string[]
  thumbnail: ReactNode
  palette: Palette
  onRemove?: () => void
  testId: string
}

function OptionCard({
  isSelected,
  onSelect,
  title,
  hint,
  warnings = [],
  thumbnail,
  palette,
  onRemove,
  testId,
}: OptionCardProps) {
  return (
    <div
      data-testid={testId}
      data-selected={isSelected}
      className={cn(
        'relative flex items-start gap-2.5 rounded-xl border p-2.5 transition-colors',
        isSelected
          ? 'border-violet-400/50 bg-violet-500/[0.08]'
          : 'border-white/[0.08] bg-white/[0.015] hover:border-white/15 hover:bg-white/[0.04]',
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={isSelected}
        className="flex min-w-0 flex-1 items-start gap-2.5 text-left"
      >
        {/* The chip sits on the theme background, because a mark that vanishes
            into the ground should look like it vanishes here too. */}
        <span
          className="flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-lg ring-1 ring-white/10"
          style={{ backgroundColor: palette.bg }}
        >
          {thumbnail}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-sm font-medium text-white/80">{title}</span>
            {isSelected && <CheckIcon className="size-3.5 shrink-0 text-violet-300" />}
          </span>
          {hint !== '' && (
            <span className="mt-0.5 block truncate text-[11px] text-white/35">{hint}</span>
          )}
          {/* One line per caveat. These come from the server's own rasteriser,
              so they are the authoritative version of what the browser's
              pre-flight SVG scan can only guess at. */}
          {warnings.map((warning) => (
            <span
              key={warning}
              data-testid="logo-server-warning"
              className="mt-1 flex items-start gap-1 text-[11px] leading-snug text-amber-300/80"
            >
              <InfoIcon className="mt-px size-3 shrink-0" />
              {warning}
            </span>
          ))}
        </span>
      </button>

      {onRemove !== undefined && (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label={`Delete ${title}`}
          onClick={onRemove}
          className="size-7 shrink-0 text-white/25 hover:bg-red-500/10 hover:text-red-300"
        >
          <Trash2Icon className="size-3.5" />
        </Button>
      )}
    </div>
  )
}
