import { formatClock, formatTimecode } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { BulletAnalysis } from '@/lib/timeline'

interface BulletTrackProps {
  analysis: BulletAnalysis
  /** Scene duration in seconds — sets the scale of the track. */
  duration: number
  className?: string
}

/**
 * A one-line map of a scene: track length is the narration duration, each tick
 * sits at its bullet's `appear_at` proportion of it.
 *
 * Reading it is the whole point — bullets bunched at the left mean the slide
 * fills up before the narrator gets there, and a tick pinned to the right edge
 * means a point that never really lands.
 */
export function BulletTrack({ analysis, duration, className }: BulletTrackProps) {
  const { timings } = analysis
  const scaled = duration > 0

  const summary = scaled
    ? timings
        .map((timing) => `${timing.bullet.text} at ${formatTimecode(timing.bullet.appear_at)}`)
        .join('; ')
    : 'Bullet cue points are not timed yet.'

  return (
    <div className={cn('space-y-1.5', className)}>
      <div
        role="img"
        aria-label={`Bullet cue points across ${formatClock(duration)} of narration. ${summary}`}
        className="relative h-7"
      >
        {/* Baseline: the full narration duration. */}
        <div className="absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 overflow-hidden rounded-full bg-white/[0.07]">
          {scaled && (
            <div className="h-full w-full bg-gradient-to-r from-violet-500/45 via-indigo-400/30 to-white/[0.06]" />
          )}
        </div>

        {/* Scene boundaries, so a tick sitting on the edge is visible. */}
        <span className="absolute top-1/2 left-0 h-3 w-px -translate-y-1/2 bg-white/15" />
        <span className="absolute top-1/2 right-0 h-3 w-px -translate-y-1/2 bg-white/15" />

        {scaled
          ? timings.map((timing) => {
              const emphasis = timing.bullet.emphasis
              const flagged = timing.isOverflow || timing.isCrowded

              return (
                <span
                  key={`${String(timing.index)}-${String(timing.bullet.appear_at)}`}
                  title={`${formatTimecode(timing.bullet.appear_at)} — ${timing.bullet.text}`}
                  style={{ left: `${String(timing.ratio * 100)}%` }}
                  className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
                >
                  <span
                    className={cn(
                      'block rounded-full transition-transform',
                      emphasis ? 'h-5 w-[3px]' : 'h-3.5 w-[3px]',
                      timing.isOverflow
                        ? 'bg-red-400 shadow-[0_0_8px] shadow-red-500/60'
                        : timing.isCrowded
                          ? 'bg-amber-300 shadow-[0_0_8px] shadow-amber-400/50'
                          : emphasis
                            ? 'bg-amber-200 shadow-[0_0_8px] shadow-amber-300/40'
                            : 'bg-violet-200 shadow-[0_0_6px] shadow-violet-400/40',
                    )}
                  />
                  {(emphasis || flagged) && (
                    <span
                      className={cn(
                        'absolute -top-1 left-1/2 size-1.5 -translate-x-1/2 rounded-full',
                        timing.isOverflow
                          ? 'bg-red-400'
                          : timing.isCrowded
                            ? 'bg-amber-300'
                            : 'bg-amber-200',
                      )}
                    />
                  )}
                </span>
              )
            })
          : /* No word timings yet: show where the points will go, greyed out. */
            timings.map((timing, position) => (
              <span
                key={`ghost-${String(timing.index)}`}
                style={{
                  left: `${String(((position + 1) / (timings.length + 1)) * 100)}%`,
                }}
                className="absolute top-1/2 h-3 w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/15"
              />
            ))}
      </div>

      <div className="flex items-center justify-between font-mono text-[10px] text-white/25 tabular-nums">
        <span>0:00</span>
        <span>{scaled ? formatClock(duration) : 'awaiting alignment'}</span>
      </div>
    </div>
  )
}
