import { MicIcon } from 'lucide-react'

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import type { SpeechEngine, Voice } from '@/lib/types'

interface VoicePickerProps {
  value: string
  voices: Voice[]
  isLoading: boolean
  /**
   * The engine these voices belong to. Voice ids are engine-native, so the
   * picker names the engine: it is the difference between `aura-2-draco-en`
   * and `Danielle` being the right answer.
   */
  engine: SpeechEngine | null
  onChange: (voiceId: string) => void
}

export function VoicePicker({ value, voices, isLoading, engine, onChange }: VoicePickerProps) {
  if (isLoading) return <Skeleton className="h-10 w-full rounded-md bg-white/5" />

  // If the default voice isn't in the catalogue, fall back to the first entry
  // so the trigger never renders empty. The parent derives `value` from this
  // same list per engine, so this is a backstop rather than the mechanism.
  const selected =
    voices.find((voice) => voice.id === value) ?? voices[0] ?? null
  const effectiveValue = selected?.id ?? value

  return (
    <Select
      // Remounted per engine: the option set is entirely different, and Radix
      // should not carry highlight or scroll position across the swap.
      key={engine?.id ?? 'unknown'}
      value={effectiveValue}
      onValueChange={onChange}
    >
      <SelectTrigger
        id="voice"
        data-engine={engine?.id ?? 'unknown'}
        data-voice={effectiveValue}
        className="h-auto w-full border-white/10 bg-white/[0.03] py-2.5 text-left text-white data-[placeholder]:text-white/30 focus-visible:border-violet-400/50 focus-visible:ring-violet-500/20 [&>span]:min-w-0 [&>span]:flex-1"
      >
        <SelectValue
          placeholder={
            engine === null ? 'Choose a voice' : `Choose a ${engine.name} voice`
          }
        >
          {selected !== null && (
            <span className="flex min-w-0 items-center gap-2">
              <MicIcon className="size-3.5 shrink-0 text-violet-300/70" />
              <span className="truncate font-medium">{selected.label}</span>
              {selected.accent !== undefined && (
                <span className="shrink-0 text-xs text-white/40">{selected.accent}</span>
              )}
            </span>
          )}
        </SelectValue>
      </SelectTrigger>

      <SelectContent className="max-h-80 border-white/10 bg-neutral-900/95 backdrop-blur">
        {engine !== null && (
          <div className="flex items-baseline justify-between gap-2 border-b border-white/[0.06] px-2 pb-1.5 text-[10px] tracking-wide text-white/30 uppercase">
            <span>{engine.name}</span>
            <span className="tabular-nums">
              {voices.length} {voices.length === 1 ? 'voice' : 'voices'}
            </span>
          </div>
        )}

        {voices.map((voice) => (
          <SelectItem
            key={voice.id}
            value={voice.id}
            className="items-start py-2.5 text-white focus:bg-violet-500/15"
          >
            <span className="flex flex-col gap-1">
              <span className="flex items-center gap-2">
                <span className="text-sm font-medium">{voice.label}</span>
                {voice.accent !== undefined && (
                  <span className="rounded border border-white/10 px-1.5 py-px text-[10px] tracking-wide text-white/50 uppercase">
                    {voice.accent}
                  </span>
                )}
              </span>

              {voice.tags !== undefined && voice.tags.length > 0 && (
                <span className="flex flex-wrap gap-1">
                  {/* Capped so long descriptor lists don't stretch the row. */}
                  {voice.tags.slice(0, 4).map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full bg-violet-500/10 px-1.5 py-px text-[10px] text-violet-200/80"
                    >
                      {tag}
                    </span>
                  ))}
                </span>
              )}

              <span className="font-mono text-[10px] text-white/25">{voice.id}</span>
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
