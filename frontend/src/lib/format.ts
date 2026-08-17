/** Small presentational formatters. */

/**
 * Parses an ISO-8601 timestamp, returning `null` if it is unusable.
 *
 * The backend builds timestamps with `datetime.now(UTC)` but SQLite drops the
 * tzinfo on the way back out, so `created_at` arrives as
 * `2026-08-17T06:48:33.221471` — no `Z`, no offset. ECMAScript reads a
 * bare date-time as *local* time, which would report a job made a minute ago
 * as hours old in any non-UTC zone. A missing designator is therefore treated
 * as UTC, matching what the backend actually meant.
 */
export function parseTimestamp(iso: string | null): number | null {
  if (iso === null) return null

  const trimmed = iso.trim()
  if (trimmed === '') return null

  // Date-only strings are already spec'd as UTC; leave them alone.
  const hasTime = trimmed.includes('T') || trimmed.includes(' ')
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(trimmed)
  const normalized = hasTime && !hasZone ? `${trimmed.replace(' ', 'T')}Z` : trimmed

  const parsed = Date.parse(normalized)
  return Number.isNaN(parsed) ? null : parsed
}

/**
 * "3 min ago" / "just now". Falls back to the raw string if unparseable.
 *
 * @param now injected so callers on a ticking clock re-render in step.
 */
export function formatRelativeTime(iso: string | null, now: number = Date.now()): string {
  if (iso === null) return 'unknown time'

  const parsed = parseTimestamp(iso)
  if (parsed === null) return iso

  const seconds = Math.round((now - parsed) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 45) return 'just now'
  if (seconds < 90) return '1 min ago'

  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${String(minutes)} min ago`

  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${String(hours)} hr${hours === 1 ? '' : 's'} ago`

  const days = Math.round(hours / 24)
  if (days < 7) return `${String(days)} day${days === 1 ? '' : 's'} ago`

  return new Date(parsed).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

/** Absolute timestamp for a tooltip, e.g. "17 Aug 2026, 12:18". */
export function formatAbsoluteTime(iso: string | null): string | null {
  const parsed = parseTimestamp(iso)
  if (parsed === null) return null
  return new Date(parsed).toLocaleString(undefined, {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * Media timecode with tenths: `4.2` -> `0:04.2`, `71.06` -> `1:11.1`.
 *
 * Used for bullet `appear_at` values, where a tenth of a second is the
 * difference between a point landing on its phrase and landing after it.
 */
export function formatTimecode(seconds: number): string {
  if (!Number.isFinite(seconds)) return '—'
  const safe = Math.max(0, seconds)
  const minutes = Math.floor(safe / 60)
  const rest = safe - minutes * 60
  // Round to tenths first so 59.98 becomes 1:00.0, not 0:60.0.
  const tenths = Math.round(rest * 10)
  const carry = Math.floor(tenths / 600)
  const displayMinutes = minutes + carry
  const displaySeconds = (tenths - carry * 600) / 10
  const padded = displaySeconds < 10 ? `0${displaySeconds.toFixed(1)}` : displaySeconds.toFixed(1)
  return `${String(displayMinutes)}:${padded}`
}

/** Whole-second clock, e.g. `0:49`. For durations rather than cue points. */
export function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds)) return '—'
  const total = Math.max(0, Math.round(seconds))
  const minutes = Math.floor(total / 60)
  const rest = total - minutes * 60
  return `${String(minutes)}:${rest < 10 ? '0' : ''}${String(rest)}`
}

/** `zoom_in` / `slideleft` -> `Zoom in` / `Slideleft`. Enum-value fallback. */
export function prettifyToken(value: string): string {
  const spaced = value.replace(/[_-]+/g, ' ').trim()
  if (spaced === '') return value
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

/** Looks up a label map, de-slugging anything the backend added since. */
export function labelFor(map: Record<string, string>, value: string | null): string | null {
  if (value === null) return null
  return map[value] ?? prettifyToken(value)
}

/** Turns a topic into a safe-ish download filename. */
export function toFilename(topic: string | null, jobId: string): string {
  const slug = (topic ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60)
  return `${slug === '' ? `video-${jobId}` : slug}.mp4`
}
