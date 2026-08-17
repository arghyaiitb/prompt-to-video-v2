import { useCallback, useRef, useState } from 'react'
import {
  CheckCircle2Icon,
  ChevronRightIcon,
  DownloadIcon,
  SparklesIcon,
} from 'lucide-react'

import { EngineBadge } from '@/components/EngineBadge'
import { SceneInspector } from '@/components/SceneInspector'
import { SceneSeekList } from '@/components/SceneSeekList'
import { ThemeBadge } from '@/components/ThemeBadge'
import { Button } from '@/components/ui/button'
import { toFilename } from '@/lib/format'
import { cn } from '@/lib/utils'
import {
  TONE_OPTIONS,
  type Job,
  type SpeechEngine,
  type ThemePreset,
  type Timeline,
} from '@/lib/types'

interface ResultViewProps {
  job: Job
  themes: ThemePreset[]
  engines: SpeechEngine[]
  timeline: Timeline | null
  timelineLoading: boolean
  timelinePending: boolean
  timelineError: string | null
  onRetryTimeline: () => void
  onReset: () => void
}

export function ResultView({
  job,
  themes,
  engines,
  timeline,
  timelineLoading,
  timelinePending,
  timelineError,
  onRetryTimeline,
  onReset,
}: ResultViewProps) {
  const videoUrl = job.video_url
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [showDetail, setShowDetail] = useState(false)

  const scenes = timeline?.scenes ?? []

  /** Seeks the actual media element — the scene list is a chapter control. */
  const handleSeek = useCallback((time: number) => {
    const video = videoRef.current
    if (video === null) return

    // Clamp: seeking past `duration` is ignored by some browsers, and the
    // timeline's narration length can exceed the file (xfade eats overlap).
    const limit = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : null
    video.currentTime = limit === null ? Math.max(0, time) : Math.min(Math.max(0, time), limit)
    setCurrentTime(video.currentTime)
    void video.play().catch(() => {
      // Autoplay policies can refuse; the seek still happened.
    })
  }, [])

  return (
    <div className="space-y-6">
      <header className="space-y-3">
        <span className="flex w-fit items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-200">
          <CheckCircle2Icon className="size-3" />
          Ready
        </span>
        <h2 className="text-xl leading-snug font-semibold text-white/90">
          {timeline?.title ?? job.title ?? job.topic ?? 'Your video'}
        </h2>
        {job.topic !== null && (timeline?.title ?? job.title ?? null) !== null && (
          <p className="text-sm text-white/40">{job.topic}</p>
        )}

        {/* What it was actually rendered with. Only shown for backends that
            report these fields — older ones omit them entirely. */}
        <div className="flex flex-wrap items-center gap-2">
          <ThemeBadge job={job} themes={themes} />
          {/* Engine and voice sit next to each other: the voice id only means
              anything in the context of the engine that produced it. */}
          <EngineBadge job={job} engines={engines} />
          {(job.voice ?? timeline?.voice ?? null) !== null && (
            <span
              className="rounded-full border border-white/10 bg-white/[0.03] px-2 py-1 font-mono text-xs text-white/60"
              title="Narrator voice"
            >
              {job.voice ?? timeline?.voice}
            </span>
          )}
          {job.bullets_per_slide != null && (
            <span className="rounded-full border border-white/10 bg-white/[0.03] px-2 py-1 text-xs text-white/60">
              {job.bullets_per_slide} bullets per slide
            </span>
          )}
          {job.tone != null && (
            <span className="rounded-full border border-white/10 bg-white/[0.03] px-2 py-1 text-xs text-white/60">
              {TONE_OPTIONS.find((option) => option.value === job.tone)?.label ?? job.tone}
            </span>
          )}
        </div>
      </header>

      {videoUrl !== null ? (
        <div className="overflow-hidden rounded-xl border border-white/10 bg-black shadow-2xl shadow-black/50">
          <video
            ref={videoRef}
            controls
            autoPlay
            playsInline
            src={videoUrl}
            onTimeUpdate={(event) => {
              setCurrentTime(event.currentTarget.currentTime)
            }}
            className="aspect-video w-full bg-black"
          >
            Your browser cannot play this video.
          </video>
        </div>
      ) : (
        <div className="rounded-xl border border-amber-400/25 bg-amber-500/[0.07] p-4 text-sm text-amber-100/80">
          The job finished but the server did not return a video URL.
        </div>
      )}

      {videoUrl !== null && scenes.length > 0 && (
        <SceneSeekList scenes={scenes} currentTime={currentTime} onSeek={handleSeek} />
      )}

      <div className="flex flex-col gap-3 sm:flex-row">
        {videoUrl !== null && (
          <Button
            asChild
            variant="outline"
            className="flex-1 border-white/15 bg-white/[0.03] text-white hover:bg-white/[0.08] hover:text-white"
          >
            {/* `download` is best-effort: cross-origin responses may ignore it. */}
            <a href={videoUrl} download={toFilename(job.topic, job.job_id)}>
              <DownloadIcon className="size-4" />
              Download MP4
            </a>
          </Button>
        )}

        <Button
          onClick={onReset}
          className="flex-1 bg-gradient-to-r from-violet-600 to-indigo-600 text-white hover:from-violet-500 hover:to-indigo-500"
        >
          <SparklesIcon className="size-4" />
          Create another
        </Button>
      </div>

      {/* Full breakdown stays collapsed by default: the video is the point,
          the bullet timings are for when something looked wrong. */}
      <div className="border-t border-white/[0.06] pt-4">
        <button
          type="button"
          onClick={() => {
            setShowDetail((value) => !value)
          }}
          aria-expanded={showDetail}
          className="flex w-full items-center gap-1.5 text-xs font-medium text-white/40 transition-colors hover:text-white/70"
        >
          <ChevronRightIcon
            className={cn('size-3.5 transition-transform', showDetail && 'rotate-90')}
          />
          {showDetail ? 'Hide' : 'Show'} scene detail
          {scenes.length > 0 && (
            <span className="font-mono text-[10px] text-white/25 tabular-nums">
              {scenes.length} scenes
            </span>
          )}
        </button>

        {showDetail && (
          <SceneInspector
            className="mt-4"
            timeline={timeline}
            isLoading={timelineLoading}
            isPending={timelinePending}
            error={timelineError}
            status={job.status}
            onRetry={onRetryTimeline}
          />
        )}
      </div>
    </div>
  )
}
