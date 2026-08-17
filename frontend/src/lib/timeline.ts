/**
 * Pure helpers for reading a Timeline. No React, no fetching.
 *
 * The point of the bullet analysis is to make bad timing obvious at a glance:
 * a point that fires after the narration has moved on, or two points landing
 * so close together that they read as one flash, are both render-time bugs
 * that are invisible in the raw numbers.
 */

import { DEFAULT_BULLET_MIN_GAP, type Bullet, type TimelineScene } from '@/lib/types'

export interface BulletTiming {
  bullet: Bullet
  /** Stable index into the scene's own bullet list. */
  index: number
  /** `appear_at / duration`, clamped to 0..1. Zero when timings are absent. */
  ratio: number
  /** Seconds since the previous bullet, or `null` for the first. */
  gap: number | null
  /** The bullet appears at or past the end of its own scene. */
  isOverflow: boolean
  /** Closer to the previous bullet than the plan's `bullet_min_gap`. */
  isCrowded: boolean
}

export interface BulletAnalysis {
  timings: BulletTiming[]
  /** Effective floor on bullet spacing, from the plan or the backend default. */
  minGap: number
  overflowCount: number
  crowdedCount: number
  hasIssues: boolean
}

export function analyzeBullets(scene: TimelineScene): BulletAnalysis {
  const minGap = scene.plan?.bullet_min_gap ?? DEFAULT_BULLET_MIN_GAP

  const bullets = scene.bullets ?? []
  // Sort by cue time so the track and the list agree, without mutating props.
  const ordered = bullets
    .map((bullet, index) => ({ bullet, index }))
    .sort((a, b) => a.bullet.appear_at - b.bullet.appear_at)

  // Before the aligner runs, every `appear_at` is 0 and every gap is 0, which
  // is not a timing bug — it is an absence of timing. Flag nothing yet.
  const timed = scene.duration > 0

  const timings: BulletTiming[] = ordered.map(({ bullet, index }, position) => {
    const previous = position > 0 ? ordered[position - 1] : undefined
    const gap = previous !== undefined ? bullet.appear_at - previous.bullet.appear_at : null

    return {
      bullet,
      index,
      ratio: timed ? Math.min(1, Math.max(0, bullet.appear_at / scene.duration)) : 0,
      gap,
      isOverflow: timed && bullet.appear_at >= scene.duration,
      isCrowded: timed && gap !== null && gap < minGap,
    }
  })

  const overflowCount = timings.filter((timing) => timing.isOverflow).length
  const crowdedCount = timings.filter((timing) => timing.isCrowded).length

  return {
    timings,
    minGap,
    overflowCount,
    crowdedCount,
    hasIssues: overflowCount > 0 || crowdedCount > 0,
  }
}

/** `Scene 01`-style label, padded so the list stays aligned past nine scenes. */
export function sceneNumber(scene: TimelineScene): string {
  return String(scene.index + 1).padStart(2, '0')
}

/** First non-empty line of copy to show for a scene. */
export function sceneTitle(scene: TimelineScene): string {
  return scene.heading ?? scene.narration?.slice(0, 60) ?? `Scene ${sceneNumber(scene)}`
}

/** The scene containing `time` (global seconds), or `null` outside the range. */
export function sceneAt(scenes: TimelineScene[], time: number): TimelineScene | null {
  for (const scene of scenes) {
    if (time >= scene.start && time < scene.end) return scene
  }
  // Past the last scene's end (rounding, or the final frame) — treat as last.
  const last = scenes.at(-1)
  if (last !== undefined && time >= last.end && last.end > 0) return last
  return null
}
