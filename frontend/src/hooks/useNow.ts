import { useEffect, useState } from 'react'

/**
 * A ticking wall clock, so "3 min ago" labels stay honest without a reload.
 *
 * Returns `Date.now()`, refreshed every `intervalMs`. Cheap by design: one
 * timer per caller and a single number in state, which is why the interval
 * defaults to a coarse 20s — relative labels are minute-granular anyway.
 */
export function useNow(intervalMs = 20_000): number {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const timer = setInterval(() => {
      setNow(Date.now())
    }, intervalMs)
    return () => {
      clearInterval(timer)
    }
  }, [intervalMs])

  return now
}
