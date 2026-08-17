import { useState } from 'react'
import { CheckIcon, WandSparklesIcon, XIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  evaluatePalette,
  formatRatio,
  isHexColour,
  normalizeHex,
  PALETTE_KEYS,
  PALETTE_LABELS,
  suggestFix,
  WCAG_AAA_TEXT,
  WCAG_AA_TEXT,
  type ContrastCheck,
  type Palette,
  type PaletteKey,
} from '@/lib/contrast'
import { cn } from '@/lib/utils'
import type { ThemeContrastFailure } from '@/lib/types'

interface PaletteEditorProps {
  palette: Palette
  onChange: (palette: Palette) => void
  /** Present once the backend has rejected this palette. */
  contrastFailure: ThemeContrastFailure | null
}

/**
 * Five colour inputs and a live WCAG readout.
 *
 * The committed palette only ever holds valid `#RRGGBB`: a hex field keeps its
 * own draft string while it is being typed, so `#0B12` half-way to `#0B1220`
 * does not turn the whole panel red, and an unparseable draft reverts on blur.
 * Contrast is therefore always measured against colours that exist, and the
 * submit gate in `CreateForm` can trust the report.
 */
export function PaletteEditor({ palette, onChange, contrastFailure }: PaletteEditorProps) {
  const [drafts, setDrafts] = useState<Partial<Record<PaletteKey, string>>>({})
  const report = evaluatePalette(palette)

  const commit = (key: PaletteKey, raw: string): void => {
    const colour = normalizeHex(raw)
    if (colour !== null) onChange({ ...palette, [key]: colour })
  }

  /**
   * The backend's own correction when it has answered with one, otherwise the
   * local mirror of the same lightness-only repair. Both keep hue and
   * saturation, so the palette still looks like the customer's brand.
   */
  const applyFix = (): void => {
    setDrafts({})
    onChange(contrastFailure?.suggestedFix ?? suggestFix(palette))
  }

  // Offered for warnings too: the repair targets the 7.0 recommendation, so it
  // is useful even when the palette is already submittable.
  const fixable = !report.isRecommended || contrastFailure !== null

  return (
    <div className="space-y-4 rounded-xl border border-white/[0.08] bg-white/[0.015] p-3.5">
      {/* Colour inputs ------------------------------------------------- */}
      <div className="space-y-2.5">
        {PALETTE_KEYS.map((key) => {
          const draft = drafts[key]
          const value = draft ?? palette[key]
          const invalidDraft = draft !== undefined && normalizeHex(draft) === null

          return (
            <div key={key} className="flex items-center gap-3">
              {/* Native picker and hex field are two views of one value. */}
              <label className="relative size-9 shrink-0 cursor-pointer overflow-hidden rounded-lg ring-1 ring-white/15 transition-shadow hover:ring-white/30">
                <span className="sr-only">{PALETTE_LABELS[key].label} colour picker</span>
                <input
                  type="color"
                  value={isHexColour(palette[key]) ? palette[key] : '#000000'}
                  onChange={(event) => {
                    setDrafts((current) => ({ ...current, [key]: undefined }))
                    commit(key, event.target.value)
                  }}
                  aria-label={`${PALETTE_LABELS[key].label} colour`}
                  // The native swatch is drawn by the UA with its own padding;
                  // oversizing and clipping makes the whole tile the colour.
                  className="absolute -inset-2 h-[calc(100%+1rem)] w-[calc(100%+1rem)] cursor-pointer border-0 bg-transparent p-0"
                />
              </label>

              <div className="min-w-0 flex-1">
                <label
                  htmlFor={`palette-${key}`}
                  className="block text-xs font-medium text-white/70"
                >
                  {PALETTE_LABELS[key].label}
                </label>
                <p className="truncate text-[11px] text-white/30">{PALETTE_LABELS[key].hint}</p>
              </div>

              <input
                id={`palette-${key}`}
                value={value}
                spellCheck={false}
                autoComplete="off"
                maxLength={7}
                aria-invalid={invalidDraft}
                onChange={(event) => {
                  const next = event.target.value
                  setDrafts((current) => ({ ...current, [key]: next }))
                  commit(key, next)
                }}
                onBlur={() => {
                  // Revert a draft that never became a colour, rather than
                  // leaving the field disagreeing with the swatch beside it.
                  setDrafts((current) => ({ ...current, [key]: undefined }))
                }}
                className={cn(
                  'w-[7.5rem] shrink-0 rounded-md border bg-black/20 px-2 py-1.5 font-mono text-xs tracking-wide text-white uppercase outline-none',
                  invalidDraft
                    ? 'border-amber-400/50 text-amber-200'
                    : 'border-white/10 focus:border-violet-400/50',
                )}
              />
            </div>
          )
        })}
      </div>

      {/* Contrast readout --------------------------------------------- */}
      <div className="space-y-2 border-t border-white/[0.06] pt-3">
        <div className="flex items-baseline justify-between">
          <h4 className="text-[11px] font-medium tracking-wide text-white/40 uppercase">
            Contrast
          </h4>
          <span className="text-[11px] text-white/25">
            {!report.isValid
              ? `${report.failures.length} to fix`
              : report.isRecommended
                ? 'All pairs clear'
                : `Passes AA · ${report.warnings.length} below our recommendation`}
          </span>
        </div>

        <ul className="space-y-1.5">
          {report.checks.map((check) => (
            <ContrastRow key={check.rule.id} check={check} />
          ))}
        </ul>

        {report.malformed.length > 0 && (
          <p className="text-xs text-red-300">
            {report.malformed.map((key) => PALETTE_LABELS[key].label).join(', ')} is not a
            6-digit hex colour, so its contrast cannot be measured.
          </p>
        )}

        {/* What to change. The backend's remedy copy, shown before the
            request rather than after a rejection. */}
        {report.failures.length > 0 && (
          <div className="space-y-1.5 rounded-lg border border-red-400/20 bg-red-500/[0.07] p-2.5">
            <p className="text-xs font-medium text-red-200">
              This palette is below WCAG AA and cannot be rendered.
            </p>
            <ul className="space-y-1">
              {report.failures.map((check) => (
                <li key={check.rule.id} className="text-[11px] leading-relaxed text-red-200/80">
                  <span className="font-medium">{check.rule.label}</span> is{' '}
                  {check.display}:1, needs {check.rule.min.toFixed(1)}:1 —{' '}
                  {check.rule.remedy}.
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Between AA and our 7.0 recommendation. Explicitly not a blocker:
            plenty of real brand colours live here, and refusing to render
            someone's actual brand is worse than rendering it at AA. */}
        {report.isValid && report.warnings.length > 0 && (
          <div className="space-y-1.5 rounded-lg border border-amber-400/20 bg-amber-500/[0.06] p-2.5">
            <p className="text-xs font-medium text-amber-100/90">
              Usable, but below what we recommend for video.
            </p>
            <ul className="space-y-1">
              {report.warnings.map((check) => (
                <li key={check.rule.id} className="text-[11px] leading-relaxed text-amber-100/70">
                  <span className="font-medium">{check.rule.label}</span> is {check.display}:1,
                  which passes WCAG AA but is under the {check.rule.recommended.toFixed(1)}:1 we
                  recommend for video — thin type may smear on a projector or at low bitrate.
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* A rejection from the server. Only reachable if its rules are
            stricter than the mirror above, since submit is blocked otherwise. */}
        {contrastFailure !== null && (
          <div className="space-y-1 rounded-lg border border-amber-400/25 bg-amber-500/[0.07] p-2.5">
            <p className="text-xs font-medium text-amber-100">The server rejected this palette.</p>
            <ul className="space-y-1">
              {contrastFailure.failures.map((failure) => (
                <li key={failure} className="text-[11px] leading-relaxed text-amber-100/75">
                  {failure}
                </li>
              ))}
            </ul>
          </div>
        )}

        {fixable && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={applyFix}
            className="w-full border-white/15 bg-white/[0.03] text-xs text-white hover:bg-white/[0.08] hover:text-white"
          >
            <WandSparklesIcon className="size-3.5" />
            {report.isValid ? 'Raise contrast' : 'Fix contrast'}
            <span className="text-white/40">
              {contrastFailure?.suggestedFix != null
                ? "— server's suggestion"
                : `— nearest palette clearing ${formatRatio(WCAG_AAA_TEXT)}:1`}
            </span>
          </Button>
        )}

        <p className="text-[11px] leading-relaxed text-white/25">
          Blocked below WCAG AA ({formatRatio(WCAG_AA_TEXT)}:1 for text, 3.0:1 for the accent,
          which paints graphics rather than words). Between AA and{' '}
          {formatRatio(WCAG_AAA_TEXT)}:1 you get a warning, not a refusal: the text is burned
          into the video and then compressed, so more headroom survives a projector — but a real
          brand colour often sits in that band and is still perfectly readable.
        </p>
      </div>
    </div>
  )
}

/**
 * Three states, not two: a failure blocks, a warning does not. Collapsing them
 * would either block a legitimate brand colour or hide a real readability risk.
 */
function ContrastRow({ check }: { check: ContrastCheck }) {
  const { rule, passes, warns, meetsAAA, display } = check

  return (
    <li className="flex items-center gap-2">
      <span
        className={cn(
          'flex size-4 shrink-0 items-center justify-center rounded-full',
          !passes
            ? 'bg-red-500/20 text-red-300'
            : warns
              ? 'bg-amber-500/20 text-amber-300'
              : 'bg-emerald-500/20 text-emerald-300',
        )}
      >
        {passes ? <CheckIcon className="size-2.5" /> : <XIcon className="size-2.5" />}
      </span>

      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs text-white/70">{rule.label}</span>
        <span className="block truncate text-[10px] text-white/30">{rule.what}</span>
      </span>

      <span
        className={cn(
          'shrink-0 font-mono text-xs tabular-nums',
          !passes ? 'text-red-300' : warns ? 'text-amber-200/90' : 'text-white/70',
        )}
      >
        {display}:1
      </span>

      <span className="w-16 shrink-0 text-right">
        {!passes ? (
          <span className="text-[9px] font-medium tracking-wide text-red-300/70 uppercase">
            needs {rule.min.toFixed(1)}
          </span>
        ) : meetsAAA ? (
          <span className="rounded bg-emerald-500/15 px-1 py-px text-[9px] font-medium tracking-wide text-emerald-300/90 uppercase">
            AAA
          </span>
        ) : warns ? (
          <span
            className="rounded bg-amber-500/15 px-1 py-px text-[9px] font-medium tracking-wide text-amber-300/90 uppercase"
            title={`Passes AA. Under the ${rule.recommended.toFixed(1)}:1 we recommend for video.`}
          >
            AA only
          </span>
        ) : (
          <span className="rounded bg-white/[0.06] px-1 py-px text-[9px] font-medium tracking-wide text-white/40 uppercase">
            {rule.level}
          </span>
        )}
      </span>
    </li>
  )
}
