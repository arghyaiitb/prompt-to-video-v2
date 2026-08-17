import { useState, type FormEvent } from 'react'
import { InfoIcon, Loader2Icon, MusicIcon, SparklesIcon, WandSparklesIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { VoicePicker } from '@/components/VoicePicker'
import { DEFAULT_VOICE_ID } from '@/lib/api'
import type { CreateJobRequest, Voice } from '@/lib/types'

const MIN_SLIDES = 2
const MAX_SLIDES = 10
const DEFAULT_SLIDES = 4

const EXAMPLE_TOPICS = [
  'How black holes bend time',
  'The rise and fall of the Silk Road',
  'Why the ocean is salty',
  'How sourdough starter actually works',
]

interface CreateFormProps {
  voices: Voice[]
  voicesLoading: boolean
  usedFallbackVoices: boolean
  isSubmitting: boolean
  onSubmit: (request: CreateJobRequest) => void
}

export function CreateForm({
  voices,
  voicesLoading,
  usedFallbackVoices,
  isSubmitting,
  onSubmit,
}: CreateFormProps) {
  const [topic, setTopic] = useState('')
  const [slideCount, setSlideCount] = useState(DEFAULT_SLIDES)
  const [voice, setVoice] = useState(DEFAULT_VOICE_ID)
  const [music, setMusic] = useState(true)
  const [touched, setTouched] = useState(false)

  const trimmedTopic = topic.trim()
  const topicInvalid = touched && trimmedTopic === ''

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    setTouched(true)
    if (trimmedTopic === '') return
    onSubmit({ topic: trimmedTopic, slide_count: slideCount, voice, music })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Topic --------------------------------------------------------- */}
      <div className="space-y-3">
        <div className="flex items-baseline justify-between">
          <Label htmlFor="topic" className="text-sm font-medium text-white/80">
            What should the video be about?
          </Label>
          <span className="text-xs text-white/30">{trimmedTopic.length} chars</span>
        </div>

        <Textarea
          id="topic"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          onBlur={() => setTouched(true)}
          placeholder="Explain how the James Webb telescope sees back in time, for a curious teenager."
          rows={4}
          aria-invalid={topicInvalid}
          aria-describedby="topic-help"
          className="resize-none border-white/10 bg-white/[0.03] text-base leading-relaxed text-white placeholder:text-white/25 focus-visible:border-violet-400/50 focus-visible:ring-violet-500/20 md:text-sm"
        />

        {topicInvalid ? (
          <p id="topic-help" className="text-xs text-red-300">
            Give the video a topic to get started.
          </p>
        ) : (
          <div id="topic-help" className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-white/30">Try:</span>
            {EXAMPLE_TOPICS.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => {
                  setTopic(example)
                }}
                className="rounded-full border border-white/10 bg-white/[0.02] px-2.5 py-1 text-xs text-white/50 transition-colors hover:border-violet-400/30 hover:bg-violet-500/10 hover:text-violet-200"
              >
                {example}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Slides -------------------------------------------------------- */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="slides" className="text-sm font-medium text-white/80">
            Number of slides
          </Label>
          <span className="rounded-md border border-violet-400/20 bg-violet-500/10 px-2 py-0.5 font-mono text-sm text-violet-200 tabular-nums">
            {slideCount}
          </span>
        </div>

        <Slider
          id="slides"
          min={MIN_SLIDES}
          max={MAX_SLIDES}
          step={1}
          value={[slideCount]}
          onValueChange={(values) => {
            const next = values[0]
            if (next !== undefined) setSlideCount(next)
          }}
          className="py-1"
        />

        <div className="flex justify-between text-xs text-white/25">
          <span>{MIN_SLIDES} — quick take</span>
          <span>{MAX_SLIDES} — deep dive</span>
        </div>
      </div>

      {/* Voice --------------------------------------------------------- */}
      <div className="space-y-3">
        <Label htmlFor="voice" className="text-sm font-medium text-white/80">
          Narrator voice
        </Label>
        <VoicePicker
          value={voice}
          voices={voices}
          isLoading={voicesLoading}
          onChange={setVoice}
        />
        {usedFallbackVoices && (
          <p className="flex items-center gap-1.5 text-xs text-amber-300/70">
            <InfoIcon className="size-3.5 shrink-0" />
            Voice list unavailable — showing built-in Deepgram voices.
          </p>
        )}
      </div>

      {/* Music --------------------------------------------------------- */}
      <label className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-white/10 bg-white/[0.02] p-4 transition-colors hover:border-white/15">
        <span className="flex items-start gap-3">
          <MusicIcon className="mt-0.5 size-4 shrink-0 text-violet-300/70" />
          <span className="space-y-0.5">
            <span className="block text-sm font-medium text-white/80">Light background music</span>
            <span className="block text-xs text-white/40">
              Adds a quiet music bed, ducked under the narration.
            </span>
          </span>
        </span>
        <Switch
          checked={music}
          onCheckedChange={setMusic}
          aria-label="Light background music"
          className="data-[state=checked]:bg-violet-500"
        />
      </label>

      {/* Submit -------------------------------------------------------- */}
      <Button
        type="submit"
        size="lg"
        disabled={isSubmitting}
        className="group relative w-full overflow-hidden bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-lg shadow-violet-950/40 transition-all hover:from-violet-500 hover:to-indigo-500 hover:shadow-violet-900/50 disabled:opacity-60"
      >
        {isSubmitting ? (
          <>
            <Loader2Icon className="size-4 animate-spin" />
            Starting your video…
          </>
        ) : (
          <>
            <WandSparklesIcon className="size-4 transition-transform group-hover:rotate-12" />
            Generate video
          </>
        )}
      </Button>

      <p className="flex items-center justify-center gap-1.5 text-center text-xs text-white/25">
        <SparklesIcon className="size-3" />
        Script, images, narration and music are generated for you.
      </p>
    </form>
  )
}
