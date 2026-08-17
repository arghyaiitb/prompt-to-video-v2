import {
  BracesIcon,
  CheckIcon,
  CircleSlashIcon,
  HelpCircleIcon,
  InfoIcon,
  TypeIcon,
} from 'lucide-react'

import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import {
  POLLY_ENGINE_ID,
  POLLY_SSML_TIERS,
  SSML_TAGS,
  ssmlSummary,
  type SpeechEngine,
} from '@/lib/types'

interface EngineSelectorProps {
  engines: SpeechEngine[]
  isLoading: boolean
  /** True when `/api/engines` 404'd and the built-in list is on screen. */
  usedFallback: boolean
  value: string
  onSelect: (engineId: string) => void
}

/**
 * Picks the speech engine, and says what that choice costs.
 *
 * The SSML capability is on the card rather than in a tooltip because it is the
 * consequence the user cannot see anywhere else: with Deepgram the narration is
 * plain text, and pacing has to come from the script. The Polly tier table is
 * shown when Polly is selected because "supports SSML" is not true uniformly —
 * `<emphasis>` fails on the two tiers worth using.
 */
export function EngineSelector({
  engines,
  isLoading,
  usedFallback,
  value,
  onSelect,
}: EngineSelectorProps) {
  if (isLoading && engines.length === 0) {
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        {[0, 1].map((key) => (
          <Skeleton key={key} className="h-28 w-full rounded-xl bg-white/[0.04]" />
        ))}
      </div>
    )
  }

  const selected = engines.find((engine) => engine.id === value) ?? null

  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        {engines.map((engine) => (
          <EngineCard
            key={engine.id}
            engine={engine}
            isSelected={engine.id === value}
            onSelect={() => {
              onSelect(engine.id)
            }}
          />
        ))}
      </div>

      {usedFallback && (
        <p className="flex items-start gap-1.5 text-xs text-amber-300/70">
          <InfoIcon className="mt-px size-3.5 shrink-0" />
          Engine list unavailable — showing the built-in options. Whether AWS credentials are
          configured could not be checked, so Polly may fail at the narration stage.
        </p>
      )}

      {/* The honest version of "supports SSML". */}
      {selected !== null && (
        <p className="flex items-start gap-1.5 rounded-xl border border-white/[0.07] bg-white/[0.015] p-3 text-xs leading-relaxed text-white/45">
          {selected.supports_ssml ? (
            <BracesIcon className="mt-px size-3.5 shrink-0 text-emerald-300/70" />
          ) : (
            <TypeIcon className="mt-px size-3.5 shrink-0 text-amber-300/70" />
          )}
          <span>{ssmlSummary(selected)}</span>
        </p>
      )}

      {selected?.id === POLLY_ENGINE_ID && <SsmlTierTable />}
    </div>
  )
}

interface EngineCardProps {
  engine: SpeechEngine
  isSelected: boolean
  onSelect: () => void
}

function EngineCard({ engine, isSelected, onSelect }: EngineCardProps) {
  const unavailable = engine.available === false
  const unknown = engine.available === null

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={unavailable}
      aria-pressed={isSelected}
      data-engine={engine.id}
      data-availability={unavailable ? 'unavailable' : unknown ? 'unknown' : 'available'}
      title={unavailable ? (engine.unavailable_reason ?? 'Credentials not configured') : engine.description}
      className={cn(
        'rounded-xl border p-3 text-left transition-all',
        unavailable
          ? 'cursor-not-allowed border-white/[0.06] bg-white/[0.01] opacity-60'
          : isSelected
            ? 'border-violet-400/60 bg-violet-500/[0.08] ring-2 ring-violet-500/30'
            : 'border-white/[0.08] bg-white/[0.015] hover:border-white/25 hover:bg-white/[0.04]',
      )}
    >
      <span className="flex items-center gap-1.5">
        <span
          className={cn(
            'truncate text-sm font-medium',
            unavailable ? 'text-white/40' : 'text-white/85',
          )}
        >
          {engine.name}
        </span>
        {engine.is_default && (
          <span className="shrink-0 rounded border border-white/10 px-1 py-px text-[9px] tracking-wide text-white/40 uppercase">
            Default
          </span>
        )}
        {isSelected && !unavailable && (
          <CheckIcon className="ml-auto size-3.5 shrink-0 text-violet-300" />
        )}
      </span>

      <span className="mt-1.5 flex flex-wrap items-center gap-1">
        {/* Capability first: it is the reason to choose one over the other. */}
        <span
          className={cn(
            'flex items-center gap-1 rounded border px-1.5 py-px text-[10px] font-medium tracking-wide uppercase',
            engine.supports_ssml
              ? 'border-emerald-400/25 bg-emerald-500/10 text-emerald-200/90'
              : 'border-amber-400/25 bg-amber-500/10 text-amber-200/90',
          )}
        >
          {engine.supports_ssml ? (
            <>
              <BracesIcon className="size-2.5" />
              SSML
            </>
          ) : (
            <>
              <TypeIcon className="size-2.5" />
              Plain text
            </>
          )}
        </span>

        {unavailable && (
          <span className="flex items-center gap-1 rounded border border-red-400/25 bg-red-500/10 px-1.5 py-px text-[10px] font-medium tracking-wide text-red-200/90 uppercase">
            <CircleSlashIcon className="size-2.5" />
            Unavailable
          </span>
        )}
        {unknown && (
          <span className="flex items-center gap-1 rounded border border-white/12 bg-white/[0.04] px-1.5 py-px text-[10px] font-medium tracking-wide text-white/50 uppercase">
            <HelpCircleIcon className="size-2.5" />
            Unverified
          </span>
        )}
      </span>

      {/* Why it cannot be picked beats a greyed-out card with no explanation. */}
      {unavailable ? (
        <span className="mt-1.5 block text-[11px] leading-relaxed text-red-200/70">
          {engine.unavailable_reason ??
            'Credentials not configured on the server — this engine cannot be used.'}
        </span>
      ) : unknown ? (
        <span className="mt-1.5 block text-[11px] leading-relaxed text-white/35">
          {engine.description !== undefined ? `${engine.description} ` : ''}
          Credentials not verified — the server did not report availability.
        </span>
      ) : (
        engine.description !== undefined && (
          <span className="mt-1.5 block text-[11px] leading-relaxed text-white/35">
            {engine.description}
          </span>
        )
      )}
    </button>
  )
}

/**
 * Per-tier SSML support, as measured on our account.
 *
 * Exists so the `<emphasis>` gap is stated rather than implied: a user who
 * reads "SSML" on the Polly card would otherwise expect all four tags to work
 * on the generative voices we default to.
 */
function SsmlTierTable() {
  return (
    <div className="overflow-hidden rounded-xl border border-white/[0.07] bg-white/[0.015]">
      <table className="w-full border-collapse text-left">
        <caption className="px-3 pt-2.5 text-left text-[11px] text-white/40">
          SSML support by Polly voice tier, measured on this account.
        </caption>
        <thead>
          <tr>
            <th className="px-3 py-2 text-[10px] font-medium tracking-wide text-white/40 uppercase">
              Tier
            </th>
            {SSML_TAGS.map((tag) => (
              <th
                key={tag}
                className="px-1.5 py-2 text-center font-mono text-[10px] font-normal text-white/40"
              >
                {tag}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {POLLY_SSML_TIERS.map((tier) => (
            <tr key={tier.tier} className="border-t border-white/[0.06]">
              <th className="px-3 py-2 text-xs font-medium text-white/70">
                <span className="block">{tier.label}</span>
                {tier.note !== undefined && (
                  <span className="block text-[10px] font-normal text-white/30">{tier.note}</span>
                )}
              </th>
              {SSML_TAGS.map((tag) => (
                <td key={tag} className="px-1.5 py-2 text-center">
                  <span
                    className={cn(
                      'text-xs',
                      tier.supports[tag] ? 'text-emerald-300/80' : 'text-red-300/70',
                    )}
                    title={
                      tier.supports[tag]
                        ? `<${tag}> works on ${tier.label.toLowerCase()} voices`
                        : `<${tag}> is rejected on ${tier.label.toLowerCase()} voices`
                    }
                  >
                    {tier.supports[tag] ? 'yes' : 'no'}
                    <span className="sr-only">
                      {tier.supports[tag] ? ' supported' : ' not supported'}
                    </span>
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
