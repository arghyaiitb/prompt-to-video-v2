/** Small presentational formatters. */

/** "3 min ago" / "just now". Falls back to the raw string if unparseable. */
export function formatRelativeTime(iso: string | null): string {
  if (iso === null) return 'unknown time'

  const parsed = Date.parse(iso)
  if (Number.isNaN(parsed)) return iso

  const seconds = Math.round((Date.now() - parsed) / 1000)
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

/** Turns a topic into a safe-ish download filename. */
export function toFilename(topic: string | null, jobId: string): string {
  const slug = (topic ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60)
  return `${slug === '' ? `video-${jobId}` : slug}.mp4`
}
