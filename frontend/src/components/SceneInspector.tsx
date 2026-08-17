import {
  AudioLinesIcon,
  FilmIcon,
  ImageIcon,
  ListChecksIcon,
  Loader2Icon,
  RotateCcwIcon,
  StarIcon,
  TriangleAlertIcon,
  TypeIcon,
} from 'lucide-react'

import { BulletTrack } from '@/components/BulletTrack'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { formatClock, formatTimecode, labelFor } from '@/lib/format'
import { analyzeBullets, sceneNumber, type BulletTiming } from '@/lib/timeline'
import { cn } from '@/lib/utils'
import {
  MOTION_LABELS,
  SLIDE_LAYOUT_LABELS,
  STAGE_LABELS,
  TEXT_ANIMATION_LABELS,
  TEXT_POSITION_LABELS,
  TRANSITION_LABELS,
  type JobStatus,
  type Timeline,
  type TimelineScene,
} from '@/lib/types'

interface SceneInspectorProps {
  timeline: Timeline | null
  isLoading: boolean
  /** The artifact does not exist yet — a normal early state. */
  isPending: boolean
  error: string | null
  /** Used to explain *why* nothing is here yet. */
  status: JobStatus
  onRetry: () => void
  className?: string
}

/* ------------------------------------------------------------------ *
 * Plan badges
 * ------------------------------------------------------------------ */

function PlanBadge({
  label,
  value,
  tone = 'neutral',
}: {
  label: string
  value: string
  tone?: 'neutral' | 'accent'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] leading-none',
        tone === 'accent'
          ? 'border-violet-400/25 bg-violet-500/10 text-violet-100/90'
          : 'border-white/[0.08] bg-white/[0.03] text-white/50',
      )}
    >
      <span className="tracking-wide text-white/30 uppercase">{label}</span>
      <span className="font-medium">{value}</span>
    </span>
  )
}

function PlanBadges({ scene }: { scene: TimelineScene }) {
  const plan = scene.plan
  if (plan === null) {
    return (
      <p className="text-[11px] text-white/25">No visual plan on this scene yet.</p>
    )
  }

  const entries: { label: string; value: string; tone?: 'accent' }[] = []
  const push = (label: string, value: string | null, tone?: 'accent'): void => {
    if (value !== null) entries.push(tone === undefined ? { label, value } : { label, value, tone })
  }

  push('Layout', labelFor(SLIDE_LAYOUT_LABELS, plan.layout))
  push('Motion', labelFor(MOTION_LABELS, plan.motion), 'accent')
  push('In', labelFor(TRANSITION_LABELS, plan.transition_in), 'accent')
  push('Text', labelFor(TEXT_POSITION_LABELS, plan.text_position))
  push('Heading', labelFor(TEXT_ANIMATION_LABELS, plan.heading_animation))
  push('Bullets', labelFor(TEXT_ANIMATION_LABELS, plan.bullet_animation))

  if (entries.length === 0) {
    return <p className="text-[11px] text-white/25">The visual plan is empty.</p>
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {entries.map((entry) => (
        <PlanBadge key={entry.label} label={entry.label} value={entry.value} tone={entry.tone} />
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ *
 * Bullets
 * ------------------------------------------------------------------ */

function BulletRow({ timing, minGap }: { timing: BulletTiming; minGap: number }) {
  const { bullet, isOverflow, isCrowded, gap } = timing

  const warning = isOverflow
    ? 'This point appears at or after the end of its scene — it will never be read.'
    : isCrowded && gap !== null
      ? `Only ${gap.toFixed(1)}s after the previous point (plan asks for ${minGap.toFixed(1)}s).`
      : null

  return (
    <li
      className={cn(
        'flex items-start gap-2.5 rounded-lg border px-2.5 py-2 transition-colors',
        isOverflow
          ? 'border-red-400/25 bg-red-500/[0.06]'
          : isCrowded
            ? 'border-amber-400/20 bg-amber-500/[0.05]'
            : bullet.emphasis
              ? 'border-amber-300/20 bg-amber-400/[0.04]'
              : 'border-white/[0.06] bg-white/[0.015]',
      )}
    >
      <span
        className={cn(
          'mt-px shrink-0 rounded font-mono text-[11px] tabular-nums',
          isOverflow ? 'text-red-200' : bullet.emphasis ? 'text-amber-200' : 'text-violet-200/70',
        )}
      >
        {formatTimecode(bullet.appear_at)}
      </span>

      <span
        className={cn(
          'min-w-0 flex-1 text-[13px] leading-snug',
          bullet.emphasis ? 'font-medium text-amber-100' : 'text-white/75',
        )}
      >
        {bullet.text}
      </span>

      {bullet.emphasis && (
        <StarIcon
          aria-label="Emphasised point"
          className="mt-0.5 size-3 shrink-0 fill-amber-300/80 text-amber-300/80"
        />
      )}

      {warning !== null && (
        // Wrapped in a span: lucide's props type omits SVG `title`.
        <span title={warning} className="mt-0.5 shrink-0">
          <TriangleAlertIcon
            aria-label={warning}
            className={cn('size-3.5', isOverflow ? 'text-red-300' : 'text-amber-300')}
          />
        </span>
      )}
    </li>
  )
}

function SceneBullets({ scene }: { scene: TimelineScene }) {
  const analysis = analyzeBullets(scene)

  if (scene.bullets === null) {
    return (
      <p className="rounded-lg border border-dashed border-white/[0.08] px-3 py-2.5 text-[11px] text-white/30">
        No bullets yet — this timeline was written before bullet planning ran.
      </p>
    )
  }

  if (scene.bullets.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-white/[0.08] px-3 py-2.5 text-[11px] text-white/30">
        No bullets planned for this scene.
      </p>
    )
  }

  return (
    <div className="space-y-2.5">
      <BulletTrack analysis={analysis} duration={scene.duration} />

      <ul className="space-y-1.5">
        {analysis.timings.map((timing) => (
          <BulletRow
            key={`${String(timing.index)}-${timing.bullet.text}`}
            timing={timing}
            minGap={analysis.minGap}
          />
        ))}
      </ul>

      {analysis.hasIssues && (
        <p className="flex items-start gap-1.5 text-[11px] text-amber-200/70">
          <TriangleAlertIcon className="mt-px size-3 shrink-0" />
          {analysis.overflowCount > 0 &&
            `${String(analysis.overflowCount)} ${analysis.overflowCount === 1 ? 'point lands' : 'points land'} past the end of this scene. `}
          {analysis.crowdedCount > 0 &&
            `${String(analysis.crowdedCount)} ${analysis.crowdedCount === 1 ? 'point appears' : 'points appear'} closer than the ${analysis.minGap.toFixed(1)}s minimum gap.`}
        </p>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ *
 * Scene card
 * ------------------------------------------------------------------ */

function AssetDot({ present, label, icon: Icon }: { present: boolean; label: string; icon: typeof ImageIcon }) {
  return (
    <span
      title={present ? `${label}: ready` : `${label}: pending`}
      className={cn(
        'flex size-5 items-center justify-center rounded border',
        present
          ? 'border-emerald-400/25 bg-emerald-500/10 text-emerald-300/80'
          : 'border-white/[0.07] bg-white/[0.02] text-white/20',
      )}
    >
      <Icon aria-label={`${label} ${present ? 'ready' : 'pending'}`} className="size-3" />
    </span>
  )
}

export function SceneCard({ scene }: { scene: TimelineScene }) {
  const bulletCount = scene.bullets?.length ?? 0

  return (
    <article className="space-y-3.5 rounded-xl border border-white/[0.07] bg-white/[0.015] p-4 transition-colors hover:border-white/[0.12]">
      <header className="flex items-start gap-3">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg border border-violet-400/20 bg-violet-500/10 font-mono text-[11px] font-medium text-violet-200 tabular-nums">
          {sceneNumber(scene)}
        </span>

        <div className="min-w-0 flex-1 space-y-1">
          <h4 className="text-sm leading-snug font-semibold text-white/90">
            {scene.heading ?? <span className="text-white/40">Untitled scene</span>}
          </h4>
          <p className="flex flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-[10px] text-white/30 tabular-nums">
            <span>
              {formatTimecode(scene.start)} → {formatTimecode(scene.end)}
            </span>
            {scene.duration > 0 && <span>· {formatClock(scene.duration)}</span>}
            <span>· {bulletCount === 0 ? 'no bullets' : `${String(bulletCount)} bullets`}</span>
            {scene.wordCount > 0 && <span>· {String(scene.wordCount)} words aligned</span>}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <AssetDot present={scene.hasImage} label="Still" icon={ImageIcon} />
          <AssetDot present={scene.hasAudio} label="Narration" icon={AudioLinesIcon} />
          <AssetDot present={scene.hasClip} label="Rendered clip" icon={FilmIcon} />
        </div>
      </header>

      {scene.narration !== null && (
        <p className="border-l-2 border-white/[0.08] pl-3 text-[13px] leading-relaxed text-white/55">
          {scene.narration}
        </p>
      )}

      <div className="space-y-2">
        <p className="flex items-center gap-1.5 text-[10px] tracking-wide text-white/30 uppercase">
          <ListChecksIcon className="size-3" />
          On-screen points
        </p>
        <SceneBullets scene={scene} />
      </div>

      <div className="flex items-start gap-1.5 border-t border-white/[0.05] pt-3">
        <TypeIcon className="mt-0.5 size-3 shrink-0 text-white/20" />
        <div className="min-w-0 flex-1">
          <PlanBadges scene={scene} />
        </div>
      </div>
    </article>
  )
}

/* ------------------------------------------------------------------ *
 * Inspector
 * ------------------------------------------------------------------ */

function pendingCopy(status: JobStatus): string {
  if (status === 'queued') return 'Waiting for a worker to pick the job up.'
  if (status === 'scripting') return 'The script is being written — scenes appear as soon as it lands.'
  if (status === 'failed') return 'This job failed before it wrote a scene breakdown.'
  return `Nothing to show yet (currently: ${STAGE_LABELS[status].toLowerCase()}).`
}

export function SceneInspector({
  timeline,
  isLoading,
  isPending,
  error,
  status,
  onRetry,
  className,
}: SceneInspectorProps) {
  const scenes = timeline?.scenes ?? []
  const totalBullets = scenes.reduce((sum, scene) => sum + (scene.bullets?.length ?? 0), 0)
  const flagged = scenes.filter((scene) => analyzeBullets(scene).hasIssues).length

  return (
    <section className={cn('space-y-3', className)}>
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="flex items-center gap-2 text-xs font-medium tracking-wide text-white/40 uppercase">
          <ListChecksIcon className="size-3.5" />
          Scene breakdown
        </h3>

        <div className="flex items-center gap-2">
          {scenes.length > 0 && (
            <p className="font-mono text-[10px] text-white/25 tabular-nums">
              {scenes.length} scenes
              {timeline !== null && timeline.duration > 0 && ` · ${formatClock(timeline.duration)}`}
              {` · ${String(totalBullets)} bullets`}
            </p>
          )}
          {isLoading && scenes.length > 0 && (
            <Loader2Icon className="size-3 animate-spin text-white/20" />
          )}
        </div>
      </div>

      {flagged > 0 && (
        <p className="flex items-start gap-2 rounded-xl border border-amber-400/20 bg-amber-500/[0.06] px-3 py-2 text-[11px] leading-relaxed text-amber-100/75">
          <TriangleAlertIcon className="mt-px size-3.5 shrink-0 text-amber-300/80" />
          {flagged} scene{flagged === 1 ? ' has' : 's have'} bullet timing worth a look — flagged
          cues are marked on the tracks below.
        </p>
      )}

      {error !== null && scenes.length === 0 && (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-white/[0.07] bg-white/[0.015] px-3 py-2.5">
          <p className="text-[11px] leading-relaxed text-white/40">
            Scene breakdown unavailable — {error}
          </p>
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            className="h-7 shrink-0 px-2 text-[11px] text-white/50 hover:bg-white/5 hover:text-white/80"
          >
            <RotateCcwIcon className="size-3" />
            Retry
          </Button>
        </div>
      )}

      {isLoading && scenes.length === 0 && (
        <div className="space-y-2">
          {[0, 1].map((key) => (
            <Skeleton key={key} className="h-32 w-full rounded-xl bg-white/[0.03]" />
          ))}
        </div>
      )}

      {!isLoading && error === null && isPending && (
        <p className="rounded-xl border border-dashed border-white/10 px-4 py-6 text-center text-xs leading-relaxed text-white/30">
          {pendingCopy(status)}
        </p>
      )}

      {scenes.length > 0 && (
        <div className="space-y-2.5">
          {scenes.map((scene) => (
            <SceneCard key={`${String(scene.index)}-${String(scene.id)}`} scene={scene} />
          ))}
        </div>
      )}
    </section>
  )
}
