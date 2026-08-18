import { useMemo, useState, type FormEvent } from 'react'
import {
  AlertTriangleIcon,
  AudioLinesIcon,
  InfoIcon,
  ListChecksIcon,
  Loader2Icon,
  MusicIcon,
  PaletteIcon,
  SparklesIcon,
  StampIcon,
  UsersIcon,
  WandSparklesIcon,
} from 'lucide-react'

import { EngineSelector } from '@/components/EngineSelector'
import { LogoPicker } from '@/components/LogoPicker'
import { SlidePreview } from '@/components/SlidePreview'
import { ThemePicker } from '@/components/ThemePicker'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { VoicePicker } from '@/components/VoicePicker'
import { logoIdForRequest } from '@/hooks/useLogos'
import { FALLBACK_THEMES, preferredVoiceId } from '@/lib/api'
import { evaluatePalette, formatRatio, type Palette } from '@/lib/contrast'
import {
  BUILT_IN_LOGO_URL,
  logoOpacityFor,
  LOGO_HEIGHT_FRACTION,
  LOGO_RENDER_HEIGHT,
  type LogoInspection,
} from '@/lib/logo'
import {
  BUILT_IN_LOGO_ID,
  CUSTOM_THEME_ID,
  DEFAULT_BULLETS,
  DEFAULT_THEME_ID,
  DEFAULT_TONE,
  MAX_BULLETS,
  MIN_BULLETS,
  TONE_OPTIONS,
  type BrandLogo,
  type CreateJobRequest,
  type LogoRejection,
  type LogoSelection,
  type SpeechEngine,
  type ThemeContrastFailure,
  type ThemePreset,
  type Tone,
  type Voice,
} from '@/lib/types'

const MIN_SLIDES = 2
const MAX_SLIDES = 10
const DEFAULT_SLIDES = 4

const EXAMPLE_TOPICS = [
  'Spotting business email compromise',
  'Handling customer data safely',
  'What to do in a security incident',
  'Password managers and why we use one',
]

const TONE_VALUES = new Set<string>(TONE_OPTIONS.map((option) => option.value))

function isTone(value: string): value is Tone {
  return TONE_VALUES.has(value)
}

/** Seeds the custom editor and the preview before the catalogue has loaded. */
const SEED_PALETTE: Palette = FALLBACK_THEMES.find(
  (theme) => theme.id === DEFAULT_THEME_ID,
)?.swatches ?? {
  bg: '#0B1220',
  surface: '#131F35',
  text: '#F8FAFC',
  muted: '#94A3B8',
  accent: '#F5A524',
}

interface CreateFormProps {
  engines: SpeechEngine[]
  enginesLoading: boolean
  usedFallbackEngines: boolean
  /** Selected engine id — owned above, because the voice list is fetched from it. */
  engineId: string
  onSelectEngine: (engineId: string) => void
  voices: Voice[]
  voicesLoading: boolean
  usedFallbackVoices: boolean
  /** The server returned another engine's voices; the built-in list stood in. */
  voicesEngineMismatch: boolean
  themes: ThemePreset[]
  themesLoading: boolean
  usedFallbackThemes: boolean
  /** A palette the server has rejected — cleared when the palette changes. */
  contrastFailure: ThemeContrastFailure | null
  onDismissContrastFailure: () => void
  /* Brand logo — owned by `useLogos` above, because the upload and the
     catalogue outlive any one render of this form. */
  logos: BrandLogo[]
  logosLoading: boolean
  logosAvailable: boolean
  logoNoneValue: string
  logoSelection: LogoSelection
  onSelectLogo: (selection: LogoSelection) => void
  onRemoveLogo: (logoId: string) => void
  logoUploadProgress: number | null
  logoUploadError: string | null
  onUploadLogo: (file: File) => Promise<unknown>
  /** A `logo_id` the server refused. Never retried around — see `createJob`. */
  logoRejection: LogoRejection | null
  isSubmitting: boolean
  onSubmit: (request: CreateJobRequest) => void
}

export function CreateForm({
  engines,
  enginesLoading,
  usedFallbackEngines,
  engineId,
  onSelectEngine,
  voices,
  voicesLoading,
  usedFallbackVoices,
  voicesEngineMismatch,
  themes,
  themesLoading,
  usedFallbackThemes,
  contrastFailure,
  onDismissContrastFailure,
  logos,
  logosLoading,
  logosAvailable,
  logoNoneValue,
  logoSelection,
  onSelectLogo,
  onRemoveLogo,
  logoUploadProgress,
  logoUploadError,
  onUploadLogo,
  logoRejection,
  isSubmitting,
  onSubmit,
}: CreateFormProps) {
  const [topic, setTopic] = useState('')
  const [slideCount, setSlideCount] = useState(DEFAULT_SLIDES)
  const [bulletCount, setBulletCount] = useState(DEFAULT_BULLETS)
  const [tone, setTone] = useState<Tone>(DEFAULT_TONE)
  const [music, setMusic] = useState(true)
  const [touched, setTouched] = useState(false)

  /**
   * The voice the user picked, tagged with the engine it was picked for.
   *
   * Stored as a pair and *derived* below rather than reset by an effect: a voice
   * id is only meaningful to one engine, so the selection has to be invalidated
   * the instant the engine changes. An effect would leave one render — and one
   * possible submit — in which a Deepgram id sat in the form under Polly, which
   * is exactly the 400 this selector exists to prevent.
   */
  const [voiceChoice, setVoiceChoice] = useState<{ engine: string; id: string } | null>(null)

  /**
   * A logo that has been inspected locally but not uploaded.
   *
   * Held here, above both previews, so the framed slide and the logo section
   * never disagree about which mark is on the frame. It does not change the
   * *selection*: `logo_id` is still whatever the picker says until an upload
   * succeeds, which is why the caption badges it as "not uploaded yet".
   */
  const [pendingLogo, setPendingLogo] = useState<LogoInspection | null>(null)

  const selectedEngine = engines.find((engine) => engine.id === engineId) ?? null

  const voice = useMemo(() => {
    if (
      voiceChoice !== null &&
      voiceChoice.engine === engineId &&
      voices.some((candidate) => candidate.id === voiceChoice.id)
    ) {
      return voiceChoice.id
    }
    // The server's `default_voice` for this engine wins, then the built-in
    // preference, then the head of the catalogue.
    return preferredVoiceId(engineId, voices, selectedEngine?.default_voice)
  }, [voiceChoice, engineId, voices, selectedEngine?.default_voice])

  /** A preset id, or `custom`. */
  const [themeId, setThemeId] = useState<string>(DEFAULT_THEME_ID)
  const [customPalette, setCustomPalette] = useState<Palette>(SEED_PALETTE)
  /** Until the editor is opened, the custom palette follows the selected preset. */
  const [customSeeded, setCustomSeeded] = useState(false)

  const isCustomTheme = themeId === CUSTOM_THEME_ID

  // The catalogue arrives asynchronously and its ids are the server's, so the
  // selection is resolved at render rather than assumed to exist.
  const selectedPreset =
    themes.find((theme) => theme.id === themeId) ??
    themes.find((theme) => theme.is_default) ??
    themes[0] ??
    null

  const activePalette: Palette = isCustomTheme
    ? customPalette
    : (selectedPreset?.swatches ?? SEED_PALETTE)

  const report = useMemo(() => evaluatePalette(activePalette), [activePalette])

  /*
   * The opacity the mark composites at, for whatever palette is active.
   *
   * `Theme.logo_opacity` is 0.85 by default and 0.92 on every light preset, and
   * the catalogue does not serialise it, so it is derived from the polarity —
   * from the *live* polarity for a custom palette, which is how the renderer
   * would resolve it too. It matters because the number the contrast note shows
   * is measured on the composite, not on the raw brand colour.
   */
  const logoOpacity = isCustomTheme
    ? logoOpacityFor({ is_light: report.isLight })
    : logoOpacityFor(selectedPreset)

  const activeThemeName = isCustomTheme ? 'your custom palette' : (selectedPreset?.name ?? 'the theme')

  /**
   * The mark to draw in the theme preview.
   *
   * Kept in step with the picker so the two frames never disagree — the theme
   * preview claims to show the slide, and a slide has the watermark on it.
   */
  const previewLogoSrc: string | null =
    pendingLogo?.objectUrl ??
    (logoSelection === BUILT_IN_LOGO_ID
      ? BUILT_IN_LOGO_URL
      : logoSelection === logoNoneValue
        ? null
        : (logos.find((logo) => logo.id === logoSelection)?.renderUrl ?? null))

  /**
   * Only a custom palette can block submission, and only below WCAG AA — the
   * same floor the server enforces. Anything between AA and the 7.0 we
   * recommend is a warning shown in the editor, not a refusal: plenty of real
   * brand colours live there.
   *
   * Presets never block: they come from the server's own validated registry, so
   * refusing one it offered would be a dead end.
   */
  const paletteBlocks = isCustomTheme && !report.isValid

  const trimmedTopic = topic.trim()
  const topicInvalid = touched && trimmedTopic === ''
  const selectedTone = TONE_OPTIONS.find((option) => option.value === tone) ?? TONE_OPTIONS[1]

  const handleSelectPreset = (id: string): void => {
    setThemeId(id)
    onDismissContrastFailure()
    // Keep the unopened editor tracking the preset, so opening it starts from
    // the palette on screen rather than from an unrelated default.
    if (!customSeeded) {
      const preset = themes.find((theme) => theme.id === id)
      if (preset !== undefined) setCustomPalette(preset.swatches)
    }
  }

  const handleSelectCustom = (): void => {
    if (!customSeeded && selectedPreset !== null) setCustomPalette(selectedPreset.swatches)
    setThemeId(CUSTOM_THEME_ID)
  }

  const handleChangeCustom = (palette: Palette): void => {
    setCustomPalette(palette)
    setCustomSeeded(true)
    // The rejection belongs to the palette that caused it, not to this one.
    onDismissContrastFailure()
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    setTouched(true)
    if (trimmedTopic === '' || paletteBlocks) return

    const request: CreateJobRequest = {
      topic: trimmedTopic,
      slide_count: slideCount,
      // `voice` is derived from `engineId`, so the pair is consistent by
      // construction — there is no state in which they can disagree.
      voice,
      music,
      tts_engine: engineId,
      theme: isCustomTheme ? CUSTOM_THEME_ID : (selectedPreset?.id ?? DEFAULT_THEME_ID),
      bullets_per_slide: bulletCount,
      tone,
    }
    // Sent only in custom mode: `ThemeCustom` forbids extra keys and requires
    // all five colours, so it is all or nothing.
    if (isCustomTheme) request.theme_custom = customPalette

    // `undefined` for the built-in mark: the field is omitted, which is exactly
    // how the backend is told to use its configured default.
    const logoId = logoIdForRequest(logoSelection)
    if (logoId !== undefined) request.logo_id = logoId

    onSubmit(request)
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

      {/* Bullets per slide --------------------------------------------- */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Label
            htmlFor="bullets"
            className="flex items-center gap-2 text-sm font-medium text-white/80"
          >
            <ListChecksIcon className="size-3.5 text-violet-300/70" />
            Bullet points per slide
          </Label>
          <span className="rounded-md border border-violet-400/20 bg-violet-500/10 px-2 py-0.5 font-mono text-sm text-violet-200 tabular-nums">
            {bulletCount}
          </span>
        </div>

        <Slider
          id="bullets"
          min={MIN_BULLETS}
          max={MAX_BULLETS}
          step={1}
          value={[bulletCount]}
          onValueChange={(values) => {
            const next = values[0]
            if (next !== undefined) setBulletCount(next)
          }}
          className="py-1"
        />

        <div className="flex justify-between text-xs text-white/25">
          <span>{MIN_BULLETS} — tight</span>
          <span>{MAX_BULLETS} — thorough</span>
        </div>

        <p className="text-xs text-white/25">
          Animated on screen in time with the narration, under each slide heading.
        </p>
      </div>

      {/* Audience / tone ----------------------------------------------- */}
      <div className="space-y-3">
        <Label htmlFor="tone" className="flex items-center gap-2 text-sm font-medium text-white/80">
          <UsersIcon className="size-3.5 text-violet-300/70" />
          Audience
        </Label>

        <Select
          value={tone}
          onValueChange={(value) => {
            // The Select is typed as `string`; narrow before it hits state.
            if (isTone(value)) setTone(value)
          }}
        >
          <SelectTrigger
            id="tone"
            className="h-auto w-full border-white/10 bg-white/[0.03] py-2.5 text-left text-white focus-visible:border-violet-400/50 focus-visible:ring-violet-500/20 [&>span]:min-w-0 [&>span]:flex-1"
          >
            <SelectValue>
              <span className="truncate font-medium">{selectedTone.label}</span>
            </SelectValue>
          </SelectTrigger>

          <SelectContent className="border-white/10 bg-neutral-900/95 backdrop-blur">
            {TONE_OPTIONS.map((option) => (
              <SelectItem
                key={option.value}
                value={option.value}
                className="items-start py-2.5 text-white focus:bg-violet-500/15"
              >
                <span className="flex flex-col gap-0.5">
                  <span className="text-sm font-medium">{option.label}</span>
                  <span className="text-[11px] text-white/40">{option.hint}</span>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <p className="text-xs text-white/25">{selectedTone.hint}</p>
      </div>

      {/* Brand theme --------------------------------------------------- */}
      <div className="space-y-4">
        <div className="flex items-baseline justify-between gap-3">
          <Label className="flex items-center gap-2 text-sm font-medium text-white/80">
            <PaletteIcon className="size-3.5 text-violet-300/70" />
            Brand theme
          </Label>
          <span className="text-xs text-white/30">
            {isCustomTheme ? 'Custom colours' : (selectedPreset?.name ?? '—')}
          </span>
        </div>

        {/* Live preview of whatever is selected. Slides are a solid ground,
            so the palette is the design — it has to be seen as a frame. */}
        <figure className="space-y-2">
          <div
            data-testid="theme-frame-preview"
            className="overflow-hidden rounded-xl border border-white/10 shadow-2xl shadow-black/40"
          >
            {/* The brand mark is part of the frame, so it is drawn here too —
                bottom-left, at the renderer's scale and opacity. */}
            <SlidePreview
              palette={activePalette}
              logo={
                previewLogoSrc === null ? null : { src: previewLogoSrc, opacity: logoOpacity }
              }
            />
          </div>
          <figcaption className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-white/30">
            <span>Preview of a hero-right slide.</span>
            <span>
              Every word uses one colour — emphasis is weight and a larger marker, never a
              second text colour.
            </span>
            <span className="font-mono text-white/40 tabular-nums">
              text {formatRatio(report.checks[0]?.ratio ?? null)}:1
            </span>
          </figcaption>
        </figure>

        <ThemePicker
          themes={themes}
          isLoading={themesLoading}
          usedFallback={usedFallbackThemes}
          value={themeId}
          customPalette={customPalette}
          onSelectPreset={handleSelectPreset}
          onSelectCustom={handleSelectCustom}
          onChangeCustom={handleChangeCustom}
          contrastFailure={contrastFailure}
        />
      </div>

      {/* Brand logo ---------------------------------------------------- */}
      <div className="space-y-4">
        <div className="flex items-baseline justify-between gap-3">
          <Label className="flex items-center gap-2 text-sm font-medium text-white/80">
            <StampIcon className="size-3.5 text-violet-300/70" />
            Brand logo
          </Label>
          <span className="text-xs text-white/30">
            {logoSelection === BUILT_IN_LOGO_ID
              ? 'Built-in mark'
              : logoSelection === logoNoneValue
                ? 'None'
                : (logos.find((logo) => logo.id === logoSelection)?.filename ??
                  'Uploaded')}
          </span>
        </div>

        <p className="text-xs leading-relaxed text-white/35">
          Composited bottom-left for the whole video, at{' '}
          {(LOGO_HEIGHT_FRACTION * 100).toFixed(1)}% of frame height —{' '}
          <span className="font-medium text-white/50">{LOGO_RENDER_HEIGHT}px at 1080p</span>. At
          that size a wordmark is unreadable, so a symbol works better than a full lockup.
        </p>

        <LogoPicker
          logos={logos}
          isLoading={logosLoading}
          available={logosAvailable}
          noneValue={logoNoneValue}
          selection={logoSelection}
          onSelect={onSelectLogo}
          onRemove={onRemoveLogo}
          progress={logoUploadProgress}
          uploadError={logoUploadError}
          onUpload={onUploadLogo}
          palette={activePalette}
          themeName={activeThemeName}
          opacity={logoOpacity}
          rejection={logoRejection}
          pending={pendingLogo}
          onPendingChange={setPendingLogo}
        />
      </div>

      {/* Speech engine ------------------------------------------------- */}
      <div className="space-y-3">
        <div className="flex items-baseline justify-between gap-3">
          <Label className="flex items-center gap-2 text-sm font-medium text-white/80">
            <AudioLinesIcon className="size-3.5 text-violet-300/70" />
            Speech engine
          </Label>
          <span className="text-xs text-white/30">{selectedEngine?.name ?? engineId}</span>
        </div>

        <EngineSelector
          engines={engines}
          isLoading={enginesLoading}
          usedFallback={usedFallbackEngines}
          value={engineId}
          onSelect={onSelectEngine}
        />
      </div>

      {/* Voice --------------------------------------------------------- */}
      <div className="space-y-3">
        <div className="flex items-baseline justify-between gap-3">
          <Label htmlFor="voice" className="text-sm font-medium text-white/80">
            Narrator voice
          </Label>
          {/* Named so it is obvious the list changed with the engine. */}
          <span className="text-xs text-white/30">
            {voices.length} {selectedEngine?.name ?? engineId} voice
            {voices.length === 1 ? '' : 's'}
          </span>
        </div>

        <VoicePicker
          value={voice}
          voices={voices}
          isLoading={voicesLoading}
          engine={selectedEngine}
          onChange={(voiceId) => {
            setVoiceChoice({ engine: engineId, id: voiceId })
          }}
        />

        {voicesEngineMismatch ? (
          <p className="flex items-start gap-1.5 text-xs text-amber-300/70">
            <InfoIcon className="mt-px size-3.5 shrink-0" />
            The server ignored the engine filter and returned another engine&rsquo;s voices —
            showing the built-in {selectedEngine?.name ?? engineId} list instead, so the voice
            sent matches the engine.
          </p>
        ) : (
          usedFallbackVoices && (
            <p className="flex items-start gap-1.5 text-xs text-amber-300/70">
              <InfoIcon className="mt-px size-3.5 shrink-0" />
              Voice list unavailable — showing built-in {selectedEngine?.name ?? engineId} voices.
            </p>
          )
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
      <div className="space-y-3">
        {/* Blocked before the request, not after: the renderer burns text into
            pixels, so an unreadable palette cannot be fixed downstream. */}
        {paletteBlocks && (
          <p
            id="submit-blocked"
            className="flex items-start gap-2 rounded-xl border border-red-400/25 bg-red-500/[0.07] p-3 text-xs leading-relaxed text-red-200/90"
          >
            <AlertTriangleIcon className="mt-px size-3.5 shrink-0" />
            <span>
              Your custom palette is below WCAG AA on{' '}
              {report.failures.length === 1 ? 'one pair' : `${report.failures.length} pairs`}
              {report.failures.length > 0 && (
                <>
                  {' '}
                  (
                  {report.failures.map((check, index) => (
                    <span key={check.rule.id}>
                      {index > 0 && ', '}
                      <span className="font-medium">{check.rule.label}</span> {check.display}:1
                      {' vs '}
                      {check.rule.min.toFixed(1)}:1
                    </span>
                  ))}
                  )
                </>
              )}
              . Adjust the colours or use <span className="font-medium">Fix contrast</span> above
              to continue.
            </span>
          </p>
        )}

        <Button
          type="submit"
          size="lg"
          disabled={isSubmitting || paletteBlocks}
          aria-describedby={paletteBlocks ? 'submit-blocked' : undefined}
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
          Script, headings, timed bullets, narration and music are generated for you.
        </p>
      </div>
    </form>
  )
}
