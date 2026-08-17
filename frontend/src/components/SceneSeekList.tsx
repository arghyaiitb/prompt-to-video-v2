import { PlayIcon } from 'lucide-react'

import { formatClock, formatTimecode } from '@/lib/format'
import { analyzeBullets, sceneNumber } from '@/lib/timeline'
import { cn } from '@/lib/utils'
import type { TimelineScene } from '@/lib/types'

interface SceneSeekListProps {
  scenes: TimelineScene[]
  /** Playhead position in global seconds, for the active highlight. */
  currentTime: number
  /** Seeks the player. `time` is global seconds. */
  onSeek: (time: number) => void
}

/**
 * Chapter list under the player. Clicking a scene — or one of its bullet cues —
 * seeks the video, which makes it easy to check that a point actually appears
 * when the narrator says it.
 */
export function SceneSeekList({ scenes, currentTime, onSeek }: SceneSeekListProps) {
  if (scenes.length === 0) return null

  return (
    <section className="space-y-2">
      <h3 className="text-xs font-medium tracking-wide text-white/40 uppercase">Scenes</h3>

      <ul className="divide-y divide-white/[0.05] overflow-hidden rounded-xl border border-white/[0.07] bg-white/[0.015]">
        {scenes.map((scene) => {
          const isActive =
            currentTime >= scene.start && (currentTime < scene.end || scene.end === 0)
          const progress =
            isActive && scene.duration > 0
              ? Math.min(100, Math.max(0, ((currentTime - scene.start) / scene.duration) * 100))
              : 0
          const bullets = analyzeBullets(scene).timings

          return (
            <li key={`${String(scene.index)}-${String(scene.id)}`} className="relative">
              {isActive && (
                <span
                  aria-hidden
                  style={{ width: `${String(progress)}%` }}
                  className="absolute inset-y-0 left-0 bg-violet-500/[0.07] transition-[width] duration-500 ease-linear"
                />
              )}

              <div className="relative space-y-2 p-3">
                <button
                  type="button"
                  onClick={() => {
                    onSeek(scene.start)
                  }}
                  aria-current={isActive}
                  className="group flex w-full items-center gap-3 text-left"
                >
                  <span
                    className={cn(
                      'flex size-6 shrink-0 items-center justify-center rounded-md border font-mono text-[10px] tabular-nums transition-colors',
                      isActive
                        ? 'border-violet-400/40 bg-violet-500/20 text-violet-100'
                        : 'border-white/[0.08] bg-white/[0.03] text-white/40 group-hover:border-violet-400/30 group-hover:text-violet-200',
                    )}
                  >
                    <PlayIcon className="hidden size-2.5 fill-current group-hover:block" />
                    <span className="group-hover:hidden">{sceneNumber(scene)}</span>
                  </span>

                  <span
                    className={cn(
                      'min-w-0 flex-1 truncate text-sm font-medium transition-colors',
                      isActive ? 'text-white' : 'text-white/70 group-hover:text-white',
                    )}
                  >
                    {scene.heading ?? scene.narration ?? `Scene ${sceneNumber(scene)}`}
                  </span>

                  <span className="shrink-0 font-mono text-[10px] text-white/30 tabular-nums">
                    {formatClock(scene.start)}
                  </span>
                </button>

                {bullets.length > 0 && (
                  <div className="flex flex-wrap gap-1 pl-9">
                    {bullets.map((timing) => {
                      const globalTime = scene.start + timing.bullet.appear_at
                      const reached = currentTime >= globalTime

                      return (
                        <button
                          key={`${String(timing.index)}-${timing.bullet.text}`}
                          type="button"
                          onClick={() => {
                            onSeek(globalTime)
                          }}
                          title={`Jump to ${formatTimecode(globalTime)} — ${timing.bullet.text}`}
                          className={cn(
                            'max-w-full truncate rounded-full border px-2 py-0.5 text-[11px] transition-colors',
                            timing.bullet.emphasis
                              ? 'border-amber-300/25 text-amber-100/80 hover:bg-amber-400/10'
                              : 'border-white/[0.08] text-white/45 hover:bg-white/[0.06] hover:text-white/80',
                            reached && 'bg-white/[0.04] text-white/70',
                          )}
                        >
                          <span className="font-mono text-[10px] text-white/30 tabular-nums">
                            {formatTimecode(timing.bullet.appear_at)}
                          </span>{' '}
                          {timing.bullet.text}
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
