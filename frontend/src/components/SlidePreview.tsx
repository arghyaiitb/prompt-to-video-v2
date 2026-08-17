import { ImageIcon } from 'lucide-react'

import { cn } from '@/lib/utils'
import type { Palette } from '@/lib/contrast'

/**
 * A 16:9 mock of the `hero_right` slide — the workhorse training layout.
 *
 * Every measurement below is the backend's own, from the geometry constants in
 * `app/render/text_overlay.py`, expressed as container-query units so the mock
 * scales with the box instead of being redrawn per size. `1cqw` is 1% of the
 * frame width, which is exactly how the renderer thinks: its constants are
 * ratios of a 1920px reference width.
 *
 *   SOLID_MARGIN_X_RATIO  0.0583 of width
 *   SOLID_MARGIN_Y_RATIO  0.1222 of height
 *   SOLID_GUTTER_RATIO    0.0417 of width
 *   TEXT_COLUMN_SHARE     0.47 of the usable width
 *   HEADING_W_RATIO       0.0328  BULLET_W_RATIO 0.0188
 *   RULE_WIDTH_RATIO      0.18 of the column   LINE_SPACING 1.22
 *   MARKER_RATIO          0.30 of the bullet size (x1.25 when emphasised)
 *
 * The point of the mock is that the palette is judged as a video frame rather
 * than as five colour chips: a ratio that passes on paper can still look wrong
 * once it is a heading on a solid ground next to an image panel.
 */

/** 16:9, so one percent of height is 0.5625 of one percent of width. */
const H = 0.5625

const MARGIN_X = 5.83
const GUTTER = 4.17
const USABLE = 100 - 2 * MARGIN_X - GUTTER
const TEXT_W = USABLE * 0.47
const IMAGE_W = USABLE - TEXT_W
const IMAGE_X = MARGIN_X + TEXT_W + GUTTER
const MARGIN_Y = 12.22 * H
const COLUMN_H = (100 - 2 * 12.22) * H

const HEADING_SIZE = 3.28
const BULLET_SIZE = 1.88
const LINE_SPACING = 1.22

/** `max(2, round(3 * scale))` — a hair over a pixel at any preview size. */
const RULE_HEIGHT = 0.15625
const RULE_WIDTH = TEXT_W * 0.18
const RULE_GAP = BULLET_SIZE * 0.55
const HEADING_GAP = BULLET_SIZE * 1.15
const BULLET_GAP = BULLET_SIZE * 0.62
const MARKER = BULLET_SIZE * 0.3
const MARKER_EMPHASIS = MARKER * 1.25
/** Shared by every bullet so emphasised and plain text start on the same x. */
const INDENT = MARKER_EMPHASIS + BULLET_SIZE * 0.55
/** `Theme.image_radius` is 24px at 1920. */
const IMAGE_RADIUS = (24 / 1920) * 100

const cqw = (value: number): string => `${value.toFixed(4)}cqw`

export interface PreviewBullet {
  text: string
  emphasis: boolean
}

const SAMPLE_HEADING = 'Report it, then stop touching it'

/** Four bullets: the default `bullets_per_slide`, one of them emphasised. */
const SAMPLE_BULLETS: PreviewBullet[] = [
  { text: 'Forward the message to phishing@ and wait', emphasis: false },
  { text: 'Never click a link to "check if it is real"', emphasis: true },
  { text: 'Tell your manager if you already replied', emphasis: false },
  { text: 'The security team will confirm within a day', emphasis: false },
]

interface SlidePreviewProps {
  palette: Palette
  heading?: string
  bullets?: PreviewBullet[]
  /** Card-sized: fewer bullets, no panel label. Used by the preset swatches. */
  compact?: boolean
  className?: string
}

export function SlidePreview({
  palette,
  heading = SAMPLE_HEADING,
  bullets = SAMPLE_BULLETS,
  compact = false,
  className,
}: SlidePreviewProps) {
  const shown = compact ? bullets.slice(0, 2) : bullets.slice(0, 4)

  return (
    <div
      aria-hidden
      className={cn('relative aspect-video w-full overflow-hidden', className)}
      style={{ containerType: 'inline-size', backgroundColor: palette.bg }}
    >
      {/* Image panel — `surface` stands in for the hero still, which is what
          the renderer fills the region with before the image lands. */}
      <div
        className="absolute flex items-center justify-center"
        style={{
          left: cqw(IMAGE_X),
          top: cqw(MARGIN_Y),
          width: cqw(IMAGE_W),
          height: cqw(COLUMN_H),
          backgroundColor: palette.surface,
          borderRadius: `max(3px, ${cqw(IMAGE_RADIUS)})`,
        }}
      >
        {compact ? (
          <ImageIcon
            style={{ width: cqw(BULLET_SIZE * 2), height: cqw(BULLET_SIZE * 2), color: palette.muted }}
          />
        ) : (
          <div className="flex flex-col items-center" style={{ gap: cqw(BULLET_SIZE * 0.6) }}>
            <ImageIcon
              style={{ width: cqw(HEADING_SIZE), height: cqw(HEADING_SIZE), color: palette.muted }}
            />
            {/* Sits on `surface`, so it is what `text_on_surface` measures. */}
            <span
              style={{
                color: palette.text,
                fontSize: cqw(BULLET_SIZE * 0.85),
                letterSpacing: '0.04em',
              }}
            >
              HERO IMAGE
            </span>
          </div>
        )}
      </div>

      {/* Text column — vertically centred, left aligned, exactly as
          `slide_geometry` resolves `hero_right`. */}
      <div
        className="absolute flex flex-col justify-center"
        style={{
          left: cqw(MARGIN_X),
          top: cqw(MARGIN_Y),
          width: cqw(TEXT_W),
          height: cqw(COLUMN_H),
        }}
      >
        {/* Kicker — the secondary tier `muted` is gated for. */}
        <span
          style={{
            color: palette.muted,
            fontSize: cqw(BULLET_SIZE * 0.8),
            letterSpacing: '0.09em',
            marginBottom: cqw(BULLET_SIZE * 0.5),
          }}
        >
          MODULE 2 · REPORTING
        </span>

        <h4
          className="line-clamp-2 font-semibold"
          style={{
            color: palette.text,
            fontSize: cqw(HEADING_SIZE),
            lineHeight: LINE_SPACING,
            margin: 0,
          }}
        >
          {heading}
        </h4>

        {/* Accent rule under the heading. A graphic, not text — which is why
            `accent_on_bg` is gated at 3.0 rather than 4.5. */}
        <div
          style={{
            backgroundColor: palette.accent,
            width: cqw(RULE_WIDTH),
            height: `max(2px, ${cqw(RULE_HEIGHT)})`,
            marginTop: cqw(RULE_GAP),
            borderRadius: '9999px',
          }}
        />

        <ul
          className="list-none"
          style={{ margin: 0, padding: 0, marginTop: cqw(HEADING_GAP) }}
        >
          {shown.map((bullet, index) => (
            <li
              key={bullet.text}
              className="flex"
              style={{
                marginTop: index === 0 ? 0 : cqw(BULLET_GAP),
                gap: cqw(INDENT - (bullet.emphasis ? MARKER_EMPHASIS : MARKER)),
              }}
            >
              {/* Marker is always `accent` and grows 25% when emphasised —
                  that plus the heavier weight is the whole emphasis signal. */}
              <span
                className="shrink-0"
                style={{
                  backgroundColor: palette.accent,
                  width: cqw(bullet.emphasis ? MARKER_EMPHASIS : MARKER),
                  height: cqw(bullet.emphasis ? MARKER_EMPHASIS : MARKER),
                  borderRadius: '9999px',
                  marginTop: cqw(BULLET_SIZE * (LINE_SPACING - 0.62)),
                }}
              />
              {/* Uniform colour: every bullet uses `text`, emphasised or not. */}
              <span
                className="line-clamp-2"
                style={{
                  color: palette.text,
                  fontSize: cqw(BULLET_SIZE),
                  lineHeight: LINE_SPACING,
                  fontWeight: bullet.emphasis ? 600 : 400,
                }}
              >
                {bullet.text}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
