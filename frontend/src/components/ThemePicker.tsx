import { CheckIcon, InfoIcon, SlidersHorizontalIcon, SunIcon } from 'lucide-react'

import { PaletteEditor } from '@/components/PaletteEditor'
import { SlidePreview } from '@/components/SlidePreview'
import { Skeleton } from '@/components/ui/skeleton'
import { contrastRatio, formatRatio, WCAG_AAA_TEXT, type Palette } from '@/lib/contrast'
import { cn } from '@/lib/utils'
import { CUSTOM_THEME_ID, type ThemeContrastFailure, type ThemePreset } from '@/lib/types'

/**
 * A UI recommendation, not a server default: bright grounds survive a projector
 * in a lit room better than dark ones, which is most of where training video is
 * watched. The server default (`midnight`) is badged separately.
 */
const RECOMMENDED_ID = 'daylight'

interface ThemePickerProps {
  themes: ThemePreset[]
  isLoading: boolean
  usedFallback: boolean
  /** A preset id, or `custom`. */
  value: string
  customPalette: Palette
  onSelectPreset: (id: string) => void
  onSelectCustom: () => void
  onChangeCustom: (palette: Palette) => void
  /** Set once the backend has rejected a palette — drives the fix button. */
  contrastFailure: ThemeContrastFailure | null
}

/**
 * Renders each preset as the slide it produces.
 *
 * Five colour chips do not tell anyone whether a palette works; a heading, a
 * rule and two bullets drawn in that palette do. The `text_on_bg` ratio is on
 * every card because the picker is also the place where an unreadable choice
 * would otherwise be made silently.
 */
export function ThemePicker({
  themes,
  isLoading,
  usedFallback,
  value,
  customPalette,
  onSelectPreset,
  onSelectCustom,
  onChangeCustom,
  contrastFailure,
}: ThemePickerProps) {
  const isCustom = value === CUSTOM_THEME_ID
  const dark = themes.filter((theme) => !theme.is_light)
  const light = themes.filter((theme) => theme.is_light)

  if (isLoading) {
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        {[0, 1, 2, 3].map((key) => (
          <Skeleton key={key} className="aspect-[16/11] w-full rounded-xl bg-white/[0.04]" />
        ))}
      </div>
    )
  }

  const renderGroup = (label: string, hint: string, group: ThemePreset[]) => {
    if (group.length === 0) return null
    return (
      <div className="space-y-2.5">
        <div className="flex items-baseline gap-2">
          <h3 className="text-[11px] font-medium tracking-wide text-white/40 uppercase">{label}</h3>
          <span className="text-[11px] text-white/25">{hint}</span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {group.map((theme) => (
            <PresetCard
              key={theme.id}
              theme={theme}
              isSelected={!isCustom && theme.id === value}
              onSelect={() => {
                onSelectPreset(theme.id)
              }}
            />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {usedFallback && (
        <p className="flex items-start gap-1.5 text-xs text-amber-300/70">
          <InfoIcon className="mt-px size-3.5 shrink-0" />
          Theme list unavailable — showing the {themes.length} built-in presets. Contrast
          ratios below are measured in the browser, so they are still accurate.
        </p>
      )}

      {renderGroup('Dark', 'text on a deep ground', dark)}
      {renderGroup('Light', 'best under room lighting', light)}

      {/* Custom ------------------------------------------------------- */}
      <div className="space-y-3 border-t border-white/[0.06] pt-4">
        <button
          type="button"
          onClick={onSelectCustom}
          aria-pressed={isCustom}
          className={cn(
            'flex w-full items-center gap-3 rounded-xl border p-3 text-left transition-colors',
            isCustom
              ? 'border-violet-400/40 bg-violet-500/[0.08]'
              : 'border-white/[0.08] bg-white/[0.015] hover:border-white/15 hover:bg-white/[0.04]',
          )}
        >
          <span
            className={cn(
              'flex size-8 shrink-0 items-center justify-center rounded-lg border',
              isCustom
                ? 'border-violet-400/30 bg-violet-500/15 text-violet-200'
                : 'border-white/10 bg-white/[0.03] text-white/40',
            )}
          >
            <SlidersHorizontalIcon className="size-4" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-medium text-white/80">Custom colours</span>
            <span className="block text-xs text-white/40">
              Your own five-colour palette, checked against WCAG as you pick.
            </span>
          </span>
          {isCustom && <CheckIcon className="size-4 shrink-0 text-violet-300" />}
        </button>

        {isCustom && (
          <PaletteEditor
            palette={customPalette}
            onChange={onChangeCustom}
            contrastFailure={contrastFailure}
          />
        )}
      </div>
    </div>
  )
}

interface PresetCardProps {
  theme: ThemePreset
  isSelected: boolean
  onSelect: () => void
}

function PresetCard({ theme, isSelected, onSelect }: PresetCardProps) {
  // Computed from the swatches rather than read from `contrast`, so the number
  // can never disagree with the colours drawn beside it — and so a fallback
  // card, which ships no ratios, shows one anyway. The maths is the backend's,
  // verified against it in `contrast.test.ts`.
  const ratio = contrastRatio(theme.swatches.text, theme.swatches.bg)
  const clearsAAA = ratio !== null && ratio >= WCAG_AAA_TEXT

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={isSelected}
      title={theme.description}
      className={cn(
        'group overflow-hidden rounded-xl border text-left transition-all',
        isSelected
          ? 'border-violet-400/60 ring-2 ring-violet-500/30'
          : 'border-white/[0.08] hover:border-white/25',
      )}
    >
      <SlidePreview palette={theme.swatches} compact />

      <div className="space-y-1.5 bg-white/[0.02] p-2.5">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-white/85">{theme.name}</span>
          {theme.is_default && (
            <span className="shrink-0 rounded border border-white/10 px-1 py-px text-[9px] tracking-wide text-white/40 uppercase">
              Default
            </span>
          )}
          {theme.id === RECOMMENDED_ID && (
            <span className="flex shrink-0 items-center gap-0.5 rounded border border-amber-400/25 bg-amber-500/10 px-1 py-px text-[9px] tracking-wide text-amber-200/80 uppercase">
              <SunIcon className="size-2.5" />
              Projector
            </span>
          )}
          {isSelected && <CheckIcon className="ml-auto size-3.5 shrink-0 text-violet-300" />}
        </div>

        <div className="flex items-center gap-1.5">
          <span
            className="font-mono text-[10px] text-white/45 tabular-nums"
            title="Contrast of text on the background"
          >
            {formatRatio(ratio)}:1
          </span>
          <span
            className={cn(
              'rounded px-1 py-px text-[9px] font-medium tracking-wide uppercase',
              clearsAAA
                ? 'bg-emerald-500/15 text-emerald-300/90'
                : 'bg-amber-500/15 text-amber-300/90',
            )}
          >
            {clearsAAA ? 'AAA' : 'AA'}
          </span>
          <span className="ml-auto flex shrink-0 gap-0.5">
            {/* The raw swatches, small — the frame above is the real signal. */}
            {[theme.swatches.bg, theme.swatches.surface, theme.swatches.text, theme.swatches.muted, theme.swatches.accent].map(
              (colour, index) => (
                <span
                  key={`${colour}-${String(index)}`}
                  className="size-2 rounded-full ring-1 ring-white/10"
                  style={{ backgroundColor: colour }}
                />
              ),
            )}
          </span>
        </div>
      </div>
    </button>
  )
}
