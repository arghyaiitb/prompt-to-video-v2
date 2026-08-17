import { BanIcon, StampIcon } from 'lucide-react'

import { BUILT_IN_LOGO_URL } from '@/lib/logo'
import { cn } from '@/lib/utils'
import type { BrandLogo, Job } from '@/lib/types'

interface LogoBadgeProps {
  job: Job
  /** Uploaded marks, for resolving an id to a name and a thumbnail. */
  logos: BrandLogo[]
  /** The server's spelling of "no brand mark". */
  noneValue: string
  /** Compact form for the history list. */
  small?: boolean
  className?: string
}

interface Resolved {
  label: string
  title: string
  thumbnail: string | null
  none: boolean
}

/**
 * What a job was branded with.
 *
 * The three states of `logo_id` are all distinct and all handled here:
 * `undefined` means the backend does not report the field, so nothing is shown —
 * inventing "built-in mark" for jobs rendered before the choice existed would be
 * a claim we cannot support. `null` is the backend's documented "fall back to the
 * configured default". A string is either the no-logo value or an uploaded id.
 */
function resolve(job: Job, logos: BrandLogo[], noneValue: string): Resolved | null {
  const id = job.logo_id
  if (id === undefined) return null

  if (id === null || id === '') {
    return {
      label: 'Built-in mark',
      title: 'Branded with the built-in mark',
      thumbnail: BUILT_IN_LOGO_URL,
      none: false,
    }
  }
  if (id === noneValue) {
    return { label: 'No logo', title: 'Rendered without a brand mark', none: true, thumbnail: null }
  }

  const match = logos.find((logo) => logo.id === id) ?? null
  // An id with no catalogue entry still gets a badge — it is what the server says
  // was used, and the logo may simply have been deleted since.
  const label = match?.original_filename ?? `Logo ${id.slice(0, 8)}`
  return {
    label,
    title: `Branded with ${label}`,
    thumbnail: match?.url ?? `/api/logos/${encodeURIComponent(id)}`,
    none: false,
  }
}

export function LogoBadge({ job, logos, noneValue, small = false, className }: LogoBadgeProps) {
  const resolved = resolve(job, logos, noneValue)
  if (resolved === null) return null

  return (
    <span
      data-testid="logo-badge"
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03]',
        small ? 'px-1.5 py-px text-[10px]' : 'px-2 py-1 text-xs',
        className,
      )}
      title={resolved.title}
    >
      {resolved.none ? (
        <BanIcon className={cn('shrink-0 text-white/40', small ? 'size-2.5' : 'size-3')} />
      ) : resolved.thumbnail !== null ? (
        <img
          src={resolved.thumbnail}
          alt=""
          className={cn('shrink-0 object-contain', small ? 'size-2.5' : 'size-3')}
          // A deleted logo's URL 404s; the icon fallback keeps the badge honest
          // rather than showing a broken image.
          onError={(event) => {
            event.currentTarget.style.display = 'none'
          }}
        />
      ) : (
        <StampIcon className={cn('shrink-0 text-white/40', small ? 'size-2.5' : 'size-3')} />
      )}
      <span className={cn('max-w-32 truncate', small ? 'text-white/45' : 'text-white/60')}>
        {resolved.label}
      </span>
    </span>
  )
}
