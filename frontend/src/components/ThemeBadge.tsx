import { PaletteIcon } from 'lucide-react'

import { FALLBACK_THEMES } from '@/lib/api'
import { cn } from '@/lib/utils'
import { CUSTOM_THEME_ID, type Job, type ThemePreset } from '@/lib/types'
import { PALETTE_KEYS, type Palette } from '@/lib/contrast'

/**
 * Resolves what a job was actually rendered with.
 *
 * `theme_custom` wins over `theme`, matching `Job.resolved_theme()` on the
 * server. Returns `null` when the payload carries no theme at all — an older
 * backend — so the caller can omit the badge rather than invent one.
 */
function resolveJobTheme(
  job: Job,
  themes: ThemePreset[],
): { label: string; palette: Palette | null; isCustom: boolean } | null {
  const custom = job.theme_custom ?? null
  const id = job.theme ?? null
  if (custom === null && id === null) return null

  if (custom !== null) return { label: 'Custom colours', palette: custom, isCustom: true }
  if (id === CUSTOM_THEME_ID) return { label: 'Custom colours', palette: null, isCustom: true }

  const catalogue = themes.length > 0 ? themes : FALLBACK_THEMES
  const preset = catalogue.find((theme) => theme.id === id)
  return {
    label: preset?.name ?? id ?? 'Unknown theme',
    palette: preset?.swatches ?? null,
    isCustom: false,
  }
}

interface ThemeBadgeProps {
  job: Job
  themes: ThemePreset[]
  /** Compact form for the history list. */
  small?: boolean
  className?: string
}

/** The palette a finished job used, as a name plus its five colours. */
export function ThemeBadge({ job, themes, small = false, className }: ThemeBadgeProps) {
  const resolved = resolveJobTheme(job, themes)
  if (resolved === null) return null

  const dots =
    resolved.palette === null ? [] : PALETTE_KEYS.map((key) => resolved.palette?.[key] ?? '#000000')

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03]',
        small ? 'px-1.5 py-px text-[10px]' : 'px-2 py-1 text-xs',
        className,
      )}
      title={`Rendered with the ${resolved.label} palette`}
    >
      {!small && <PaletteIcon className="size-3 shrink-0 text-white/40" />}
      <span className={small ? 'text-white/45' : 'text-white/60'}>{resolved.label}</span>
      {dots.length > 0 && (
        <span className="flex shrink-0 gap-0.5">
          {dots.map((colour, index) => (
            <span
              key={`${colour}-${String(index)}`}
              className={cn('rounded-full ring-1 ring-white/10', small ? 'size-1.5' : 'size-2')}
              style={{ backgroundColor: colour }}
            />
          ))}
        </span>
      )}
    </span>
  )
}
