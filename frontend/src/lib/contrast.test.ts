/**
 * Unit tests for `contrast.ts`.
 *
 * No test runner is installed in this project, and `contrast.ts` is pure and
 * import-free precisely so it can be checked without one:
 *
 *     pnpm run test:contrast
 *
 * which compiles this file plus its subject to CommonJS in a temp directory and
 * runs it under plain `node`. A failed expectation throws, so a non-zero exit
 * code is the pass/fail signal.
 *
 * Expected values are not hand-computed. Every number below was taken from the
 * backend itself — `python -c "from app.core.themes import contrast_table"` —
 * because the backend gate is the authority and this module only earns its
 * keep by agreeing with it. If a number here disagrees with the backend, the
 * maths in `contrast.ts` is wrong; the expectation is not the thing to adjust.
 */

import {
  CONTRAST_RULES,
  contrastRatio,
  evaluatePalette,
  isHexColour,
  isLightBackground,
  normalizeHex,
  parseHex,
  relativeLuminance,
  RECOMMENDED_THRESHOLDS,
  REQUIRED_THRESHOLDS,
  suggestFix,
  WCAG_AAA_TEXT,
  WCAG_AA_TEXT,
  WCAG_LARGE_OBJECT,
  type Palette,
} from './contrast'

let failures = 0
let assertions = 0

function fail(what: string, detail: string): void {
  failures += 1
  console.error(`  FAIL  ${what}\n        ${detail}`)
}

function ok(what: string): void {
  assertions += 1
  console.log(`  pass  ${what}`)
}

function expectEqual(what: string, actual: unknown, expected: unknown): void {
  if (actual === expected) ok(`${what} = ${String(expected)}`)
  else fail(what, `expected ${String(expected)}, got ${String(actual)}`)
}

/** Ratios are floats: compare to the precision the backend reports. */
function expectClose(what: string, actual: number | null, expected: number, dp = 4): void {
  const tolerance = 0.5 * 10 ** -dp
  if (actual !== null && Math.abs(actual - expected) < tolerance) {
    ok(`${what} = ${actual.toFixed(dp)}`)
  } else {
    fail(what, `expected ${expected.toFixed(dp)} +/- ${tolerance.toString()}, got ${String(actual)}`)
  }
}

function expectTrue(what: string, value: boolean, detail = ''): void {
  if (value) ok(what)
  else fail(what, detail === '' ? 'expected true, got false' : `expected true, got false: ${detail}`)
}

function palette(parts: Palette): Palette {
  return parts
}

/* ------------------------------------------------------------------ *
 * 1. The WCAG anchors
 * ------------------------------------------------------------------ */

console.log('\nWCAG anchor values')
expectClose('#000000 on #FFFFFF', contrastRatio('#000000', '#FFFFFF'), 21, 6)
expectClose('#FFFFFF on #FFFFFF', contrastRatio('#FFFFFF', '#FFFFFF'), 1, 6)
expectClose('#000000 on #000000', contrastRatio('#000000', '#000000'), 1, 6)
// Order must not matter: the ratio puts the lighter colour on top either way.
expectClose('#FFFFFF on #000000 (reversed)', contrastRatio('#FFFFFF', '#000000'), 21, 6)
expectClose('luminance of black', relativeLuminance({ r: 0, g: 0, b: 0 }), 0, 6)
expectClose('luminance of white', relativeLuminance({ r: 255, g: 255, b: 255 }), 1, 6)
// Mid grey is the classic check that the 2.4 gamma curve is applied, not a
// naive channel/255 average: #777777 is 0.1845, not 0.4667.
expectClose('luminance of #777777', relativeLuminance({ r: 119, g: 119, b: 119 }), 0.184475, 6)
// The linear branch below 0.04045 — #0A is 10/255 = 0.0392, so c/12.92.
expectClose('luminance of #0A0A0A', relativeLuminance({ r: 10, g: 10, b: 10 }), 0.0030353, 6)

/* ------------------------------------------------------------------ *
 * 2. The default palette, against the numbers in the brief
 * ------------------------------------------------------------------ */

console.log('\nDefault `midnight` palette (backend: 17.89 text / 7.30 muted / 9.17 accent)')
const midnight = palette({
  bg: '#0B1220',
  surface: '#131F35',
  text: '#F8FAFC',
  muted: '#94A3B8',
  accent: '#F5A524',
})
expectClose('text_on_bg', contrastRatio(midnight.text, midnight.bg), 17.895)
expectClose('text_on_surface', contrastRatio(midnight.text, midnight.surface), 15.741)
expectClose('muted_on_bg', contrastRatio(midnight.muted, midnight.bg), 7.3022)
expectClose('accent_on_bg', contrastRatio(midnight.accent, midnight.bg), 9.1743)
// Two decimals is what the UI shows and what the backend prints.
expectEqual('text_on_bg to 2dp', (contrastRatio(midnight.text, midnight.bg) ?? 0).toFixed(2), '17.89')
expectEqual('muted_on_bg to 2dp', (contrastRatio(midnight.muted, midnight.bg) ?? 0).toFixed(2), '7.30')
expectEqual('accent_on_bg to 2dp', (contrastRatio(midnight.accent, midnight.bg) ?? 0).toFixed(2), '9.17')

/* ------------------------------------------------------------------ *
 * 3. Every shipped preset clears the gate
 *
 * Ratios copied from `app.core.themes.contrast_table()`. These also guard the
 * fallback catalogue in `api.ts`, which is this same set of palettes.
 * ------------------------------------------------------------------ */

console.log('\nPresets vs backend contrast_table()')
const PRESET_EXPECTATIONS: {
  id: string
  isLight: boolean
  palette: Palette
  ratios: [number, number, number, number]
}[] = [
  {
    id: 'midnight',
    isLight: false,
    palette: midnight,
    ratios: [17.895, 15.741, 7.3022, 9.1743],
  },
  {
    id: 'graphite',
    isLight: false,
    palette: {
      bg: '#15171C',
      surface: '#212530',
      text: '#F4F6F8',
      muted: '#A2A9B6',
      accent: '#38BDF8',
    },
    ratios: [16.5507, 14.1288, 7.5878, 8.3697],
  },
  {
    id: 'halo',
    isLight: false,
    palette: {
      bg: '#140C24',
      surface: '#20153A',
      text: '#F6F2FF',
      muted: '#B7A9DA',
      accent: '#863BFF',
    },
    ratios: [17.2082, 15.5015, 8.757, 3.6859],
  },
  {
    id: 'forest',
    isLight: false,
    palette: {
      bg: '#0C2118',
      surface: '#143024',
      text: '#F1F7F3',
      muted: '#96B8A6',
      accent: '#F0B429',
    },
    ratios: [15.5161, 13.0843, 7.7856, 9.0387],
  },
  {
    id: 'daylight',
    isLight: true,
    palette: {
      bg: '#F7F8FA',
      surface: '#FFFFFF',
      text: '#111827',
      muted: '#55607A',
      accent: '#1D4ED8',
    },
    ratios: [16.6941, 17.7397, 5.9161, 6.3066],
  },
  {
    id: 'boardroom',
    isLight: true,
    palette: {
      bg: '#E8EDF4',
      surface: '#F8FAFD',
      text: '#0F2440',
      muted: '#49597A',
      accent: '#0E6BA8',
    },
    ratios: [13.2555, 14.9135, 5.9638, 4.8414],
  },
  {
    id: 'paper',
    isLight: true,
    palette: {
      bg: '#F6F1E7',
      surface: '#FFFDF8',
      text: '#211C15',
      muted: '#5D5346',
      accent: '#A8442A',
    },
    ratios: [15.0247, 16.6379, 6.6834, 5.2917],
  },
  {
    id: 'lilac',
    isLight: true,
    palette: {
      bg: '#F5F1FF',
      surface: '#FFFFFF',
      text: '#1B1230',
      muted: '#574A78',
      accent: '#863BFF',
    },
    ratios: [16.0814, 17.8601, 7.1193, 4.6308],
  },
]

for (const preset of PRESET_EXPECTATIONS) {
  const report = evaluatePalette(preset.palette)
  expectTrue(`${preset.id} passes the gate`, report.isValid)
  // Our own presets are held to the stricter recommendation, not just AA:
  // we choose these, so they should be excellent. A preset that only warns
  // is a bug in the registry.
  expectTrue(`${preset.id} clears the recommendation too`, report.isRecommended)
  expectEqual(`${preset.id} is_light`, report.isLight, preset.isLight)
  CONTRAST_RULES.forEach((rule, index) => {
    expectClose(`${preset.id} ${rule.id}`, report.checks[index]?.ratio ?? null, preset.ratios[index] ?? 0)
  })
}

/* ------------------------------------------------------------------ *
 * 4. Thresholds mirror the backend's THRESHOLDS
 * ------------------------------------------------------------------ */

console.log('\nThresholds — two tiers')
expectEqual('AA text', WCAG_AA_TEXT, 4.5)
expectEqual('AAA text', WCAG_AAA_TEXT, 7)
expectEqual('non-text / large object', WCAG_LARGE_OBJECT, 3)

// The blocking floor is WCAG AA, mirroring the backend's REQUIRED_THRESHOLDS.
// Gating a customer's own brand colours at AAA rejects legitimate palettes.
expectEqual('required text_on_bg', REQUIRED_THRESHOLDS.text_on_bg, 4.5)
expectEqual('required text_on_surface', REQUIRED_THRESHOLDS.text_on_surface, 4.5)
expectEqual('required muted_on_bg', REQUIRED_THRESHOLDS.muted_on_bg, 4.5)
// The accent bar is 3.0 because accent paints the heading rule and bullet
// markers, which are graphics under SC 1.4.11 — not a relaxed text rule.
expectEqual('required accent_on_bg', REQUIRED_THRESHOLDS.accent_on_bg, 3)

// The recommendation (and the bar our own presets are held to) is AAA for text.
expectEqual('recommended text_on_bg', RECOMMENDED_THRESHOLDS.text_on_bg, 7)
expectEqual('recommended text_on_surface', RECOMMENDED_THRESHOLDS.text_on_surface, 7)
expectEqual('recommended muted_on_bg', RECOMMENDED_THRESHOLDS.muted_on_bg, 4.5)
expectEqual('recommended accent_on_bg', RECOMMENDED_THRESHOLDS.accent_on_bg, 3)

expectEqual('rule 0 blocks at AA', CONTRAST_RULES[0]?.min, 4.5)
expectEqual('rule 0 recommends AAA', CONTRAST_RULES[0]?.recommended, 7)
expectEqual('rule 3 blocks at 3.0', CONTRAST_RULES[3]?.min, 3)

/* ------------------------------------------------------------------ *
 * 5. Failure detection and reporting
 * ------------------------------------------------------------------ */

console.log('\nFailing palettes')
const unreadable = palette({
  bg: '#FFFFFF',
  surface: '#FFFFFF',
  text: '#DDDDDD',
  muted: '#EEEEEE',
  accent: '#FFFF00',
})
const unreadableReport = evaluatePalette(unreadable)
expectEqual('all four pairs fail', unreadableReport.failures.length, 4)
expectTrue('palette is rejected', !unreadableReport.isValid)
expectTrue('failure names its pair', unreadableReport.failures[0]?.message?.includes('text_on_bg') === true)
// The requirement quoted is the AA floor, not the recommendation.
expectTrue(
  'failure names the AA requirement',
  unreadableReport.failures[0]?.message?.includes('needs >= 4.50:1') === true,
  unreadableReport.failures[0]?.message ?? 'null',
)
expectEqual('nothing merely warns when everything fails', unreadableReport.warnings.length, 0)
expectTrue('white bg reads as light', unreadableReport.isLight)

// An accent between 3.0 and 4.5 must pass: this is the case where treating
// accent as text would wrongly block a legitimate palette. `lilac` ships at
// 4.63 and `halo` at 3.69, both below the 4.5 text bar.
const graphicAccent = palette({ ...midnight, accent: '#8C6112' })
const graphicReport = evaluatePalette(graphicAccent)
expectClose('accent at 3.x on bg', graphicReport.checks[3]?.ratio ?? null, 3.418, 3)
expectTrue('accent below 4.5 still passes its 3.0 bar', graphicReport.isValid)

/* ---- the AA-to-AAA band warns, and must NOT block ----
 *
 * This is the case the two-tier gate exists for. Gating a user's palette at
 * AAA refuses ordinary corporate colours, so anything above AA is accepted
 * with a warning. Blocking here would be stricter than the server, which is
 * the wrong direction for a client-side mirror.
 */
const aaOnly = palette({ ...midnight, text: '#8A93A5', muted: '#8A93A5' })
const aaReport = evaluatePalette(aaOnly)
expectClose('AA-band text ratio', aaReport.checks[0]?.ratio ?? null, 6.0605, 3)
expectTrue('AA-but-not-AAA text is NOT a failure', aaReport.isValid)
expectTrue('it is a warning', aaReport.warnings.length > 0)
expectTrue('and is flagged on the pair', aaReport.checks[0]?.warns === true)
expectTrue('so the palette is valid but not recommended', !aaReport.isRecommended)
expectTrue('while still meeting AA', aaReport.checks[0]?.meetsAA === true)
expectTrue('and not AAA', aaReport.checks[0]?.meetsAAA === false)
expectTrue(
  'warning wording matches the server',
  aaReport.checks[0]?.warning ===
    'text_on_bg is 6.06:1, which passes WCAG AA but is under the 7.0:1 we recommend for video — thin type may smear on a projector or at low bitrate',
  aaReport.checks[0]?.warning ?? 'null',
)

// Slate on white: the exact palette that motivated the change. 4.76:1 is
// perfectly accessible and a completely normal corporate grey.
const slate = palette({
  bg: '#FFFFFF',
  surface: '#FFFFFF',
  text: '#64748B',
  muted: '#64748B',
  accent: '#94A3B8',
})
const slateReport = evaluatePalette(slate)
expectClose('slate #64748B on white', slateReport.checks[0]?.ratio ?? null, 4.7588)
expectTrue('slate text is accepted', slateReport.checks[0]?.passes === true)
expectTrue('slate text warns', slateReport.checks[0]?.warns === true)
expectTrue('slate muted passes outright at AA', slateReport.checks[2]?.passes === true)
expectTrue('slate muted does not warn (its bar is AA)', slateReport.checks[2]?.warns === false)
// Its accent is the one real failure: 2.56:1, under the 3.0 graphic bar.
expectClose('slate accent', slateReport.checks[3]?.ratio ?? null, 2.564)
expectEqual('one blocking failure', slateReport.failures.length, 1)
expectEqual('failing pair is the accent', slateReport.failures[0]?.rule.id, 'accent_on_bg')
expectEqual('two warnings', slateReport.warnings.length, 2)
expectTrue('palette blocked only by the accent', !slateReport.isValid)

// Raising just the accent makes the same palette submittable, warnings intact.
const slateFixedAccent = evaluatePalette({ ...slate, accent: '#6B7A99' })
expectTrue('accent lifted above 3.0 unblocks it', slateFixedAccent.isValid)
expectTrue('the AA-band warnings remain', slateFixedAccent.warnings.length === 2)

// Malformed colours short-circuit the ratios rather than reporting a bogus
// number, exactly as `validate_theme` bails before measuring.
const broken = palette({ ...midnight, text: 'red' })
const brokenReport = evaluatePalette(broken)
expectEqual('malformed colour is listed', brokenReport.malformed.join(','), 'text')
expectTrue('malformed palette is rejected', !brokenReport.isValid)
expectEqual('its ratio is unknown, not zero', brokenReport.checks[0]?.ratio, null)
expectEqual('and displays as a dash', brokenReport.checks[0]?.display, '—')

/* ------------------------------------------------------------------ *
 * 6. Hex handling
 * ------------------------------------------------------------------ */

console.log('\nHex normalisation')
expectEqual('#0b1220 uppercased', normalizeHex('#0b1220'), '#0B1220')
expectEqual('bare 6-digit', normalizeHex('0B1220'), '#0B1220')
expectEqual('whitespace trimmed', normalizeHex('  #0B1220  '), '#0B1220')
expectEqual('3-digit expanded', normalizeHex('#0f8'), '#00FF88')
expectEqual('bare 3-digit expanded', normalizeHex('fff'), '#FFFFFF')
expectEqual('not a colour', normalizeHex('red'), null)
expectEqual('too short', normalizeHex('#12'), null)
expectEqual('too long', normalizeHex('#1234567'), null)
expectEqual('non-hex digits', normalizeHex('#gggggg'), null)
expectEqual('empty', normalizeHex(''), null)
// `isHexColour` is the stricter backend-shaped check: 6 digits only.
expectTrue('isHexColour accepts #RRGGBB', isHexColour('#0B1220'))
expectTrue('isHexColour rejects shorthand', !isHexColour('#0f8'))
const parsed = parseHex('#0B1220')
expectEqual('parseHex red', parsed?.r, 11)
expectEqual('parseHex green', parsed?.g, 18)
expectEqual('parseHex blue', parsed?.b, 32)

console.log('\nPolarity')
expectTrue('#0B1220 is dark', !isLightBackground('#0B1220'))
expectTrue('#F7F8FA is light', isLightBackground('#F7F8FA'))
// The boundary is luminance > 0.5, not lightness: #808080 is 0.2159 and dark.
expectTrue('#808080 is dark by luminance', !isLightBackground('#808080'))
// #BBBBBB sits just under the line at 0.4969; #CCCCCC is 0.6038.
expectTrue('#BBBBBB is dark', !isLightBackground('#BBBBBB'))
expectTrue('#CCCCCC is light', isLightBackground('#CCCCCC'))

/* ------------------------------------------------------------------ *
 * 7. suggestFix parity with the backend
 *
 * Expected palettes are the literal output of `app.core.themes.suggest_fix`
 * for the same inputs. Matching to the byte matters: the button applies this
 * locally, and the user must not see one palette from the client and a
 * different one from a later 422.
 * ------------------------------------------------------------------ */

console.log('\nsuggestFix vs backend suggest_fix')
const FIX_CASES: { name: string; input: Palette; expected: Palette }[] = [
  {
    name: 'washed light (foregrounds move, bg held)',
    input: unreadable,
    expected: {
      bg: '#FFFFFF',
      surface: '#FFFFFF',
      text: '#595959',
      muted: '#767676',
      accent: '#9A9A00',
    },
  },
  {
    name: 'mid-tone ground (bg has to give)',
    input: { bg: '#808080', surface: '#8A8A8A', text: '#909090', muted: '#999999', accent: '#8F8F8F' },
    expected: {
      bg: '#595959',
      surface: '#595959',
      text: '#FFFFFF',
      muted: '#D0D0D0',
      accent: '#AAAAAA',
    },
  },
  {
    name: 'dark with weak muted and accent',
    input: { bg: '#0B1220', surface: '#131F35', text: '#F8FAFC', muted: '#334155', accent: '#1E293B' },
    expected: {
      bg: '#0B1220',
      surface: '#131F35',
      text: '#F8FAFC',
      muted: '#647EA2',
      accent: '#47628C',
    },
  },
  {
    name: 'brand violet used as text (hue preserved)',
    input: { bg: '#140C24', surface: '#20153A', text: '#863BFF', muted: '#5B4B80', accent: '#2A1D45' },
    expected: {
      bg: '#140C24',
      surface: '#140D24',
      text: '#B486FF',
      muted: '#8472AD',
      accent: '#6D4CB4',
    },
  },
]

for (const testCase of FIX_CASES) {
  const fixed = suggestFix(testCase.input)
  for (const key of ['bg', 'surface', 'text', 'muted', 'accent'] as const) {
    expectEqual(`${testCase.name}: ${key}`, fixed[key], testCase.expected[key])
  }
  expectTrue(`${testCase.name}: result clears the gate`, evaluatePalette(fixed).isValid)
}

// A passing palette is returned untouched — the button must be a no-op there.
const untouched = suggestFix(midnight)
expectEqual('passing palette keeps its text', untouched.text, midnight.text)
expectEqual('passing palette keeps its bg', untouched.bg, midnight.bg)
// Malformed input cannot be repaired; it comes back as-is for the editor to flag.
expectEqual('malformed input passes through', suggestFix(broken).text, 'red')

/* ------------------------------------------------------------------ *

 * ------------------------------------------------------------------ */

console.log(`\n${String(assertions)} passed, ${String(failures)} failed`)
if (failures > 0) {
  throw new Error(`${String(failures)} contrast assertion(s) failed`)
}
