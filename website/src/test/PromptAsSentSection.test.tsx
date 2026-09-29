/**
 * "Prompt as sent" — the developer view of a turn's exact text under the
 * Context Breakdown panel.
 *
 *  - adjacent same-label spans merge into one row; a gap never merges across.
 *  - a turn is matched to the newest prompt recorded before its usage row and
 *    after the previous turn's row, so a failed turn is skipped, not credited.
 *  - the panel omits the section until the prompt trace has loaded, marks the
 *    turns that still have text with a dot, and says plainly when the selected
 *    turn has none.
 *  - a segment row opens to the raw slice of the prompt it names.
 */
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react'

import { ContextBreakdownPanel, type ContextTrace, type ContextTurn } from '../pages/ContextBreakdownPanel'

vi.mock('../utils/clipboard', () => ({ copyToClipboard: vi.fn(async () => false) }))
import {
  promptForTurn,
  segmentNames,
  segmentsFor,
  spansToUtf16,
  type PromptRecord,
  type PromptTrace,
} from '../pages/PromptAsSentSection'

afterEach(cleanup)

const TEXT =
  '[CRITICAL RULES -- always follow these]\nrule\n[END CRITICAL RULES]\n\n' +
  '[CURRENT USER REQUEST -- respond to this]\nhello there\n\n(If presenting choices, end with x.)'

const HEADER = TEXT.indexOf('[CURRENT USER REQUEST')
const USER = TEXT.indexOf('hello there')

const record = (over: Partial<PromptRecord> = {}): PromptRecord => ({
  ts: '2026-08-04T00:00:00Z',
  chars: TEXT.length,
  assembled_chars: TEXT.length,
  text: TEXT,
  truncated: false,
  redacted: false,
  spans: [
    // One span per block, as the backend serves them (it coalesces a block's
    // body with the blank line after its closer before serving).
    { start: 0, end: HEADER, label: 'critical_rules' },
    { start: HEADER, end: USER, label: 'request_header' },
    { start: USER, end: USER + 'hello there'.length, label: 'your_message' },
    { start: USER + 'hello there'.length, end: TEXT.length, label: 'reply_format_rules' },
  ],
  ...over,
})

const turn = (over: Partial<ContextTurn> = {}): ContextTurn => ({
  ts: '2026-08-04T00:00:30Z',
  phase: 'per_turn',
  blocks: { request_header: 42, your_message: 11, critical_rules: 63, reply_format_rules: 40 },
  total_chars: 156,
  context_used: 2000,
  context_window: 200000,
  model: 'auto',
  ...over,
})

const trace = (turns: ContextTurn[]): ContextTrace => ({
  slot: 'chat-1',
  turns,
  totals: {},
  injected_chars: 0,
  user_chars: 0,
  peak_context_used: 0,
  context_window: 0,
  window_days: 14,
})

const prompts = (turns: PromptRecord[], over: Partial<PromptTrace> = {}): PromptTrace => ({
  slot: 'chat-1',
  turns,
  dropped: 0,
  evicted: false,
  ...over,
})

describe('segmentsFor', () => {
  it('is one segment per served span, sized in code points', () => {
    const segs = segmentsFor(TEXT, record().spans)
    expect(segs.map(s => s.label)).toEqual([
      'critical_rules',
      'request_header',
      'your_message',
      'reply_format_rules',
    ])
    expect(segs[0]).toEqual({ label: 'critical_rules', start: 0, end: HEADER, chars: HEADER })
  })

  it('does not merge: the backend already coalesced, so two same-label spans are two rows', () => {
    // A repeated block (the reply-format rules sit at both ends) is two spans
    // with text between them; a merge here would be a second spelling of the
    // server's rule, and the ordinal the view adds needs them kept apart.
    const segs = segmentsFor('aaaaabbaa', [
      { start: 0, end: 5, label: 'a' },
      { start: 5, end: 7, label: 'b' },
      { start: 7, end: 9, label: 'a' },
    ])
    expect(segs).toHaveLength(3)
  })

  it('drops an empty span', () => {
    expect(segmentsFor('ab', [{ start: 0, end: 0, label: 'x' }, { start: 0, end: 2, label: 'y' }])).toHaveLength(1)
  })
})

describe('promptForTurn', () => {
  const p1 = record({ ts: '2026-08-04T00:00:00Z', text: 'one' })
  const p2 = record({ ts: '2026-08-04T00:01:00Z', text: 'two' })
  const p3 = record({ ts: '2026-08-04T00:02:00Z', text: 'three' })

  it('picks the newest prompt stamped before the turn row', () => {
    expect(promptForTurn([p1, p2, p3], '2026-08-04T00:01:30Z', undefined)?.text).toBe('two')
  })

  it('never reaches back past the previous turn row', () => {
    // p2's turn produced no usage row (it failed); the next row must not adopt it.
    expect(promptForTurn([p1, p2], '2026-08-04T00:01:30Z', '2026-08-04T00:01:10Z')).toBeNull()
  })

  it('is null when nothing was recorded before the row', () => {
    expect(promptForTurn([p3], '2026-08-04T00:00:30Z', undefined)).toBeNull()
  })

  it('compares instants, so a local-offset row still finds its UTC prompt', () => {
    // 00:01:30Z written as a -07:00 local stamp sorts BEFORE every UTC string.
    expect(promptForTurn([p1, p2, p3], '2026-08-03T17:01:30-07:00', undefined)?.text).toBe('two')
  })
})

describe('spansToUtf16', () => {
  it('shifts every offset after a non-BMP character by its extra UTF-16 unit', () => {
    // "a😀b" is 3 code points but 4 UTF-16 units; the backend counts code points.
    const text = 'a😀b|tail'
    const spans = spansToUtf16(text, [
      { start: 0, end: 3, label: 'x' },
      { start: 3, end: 8, label: 'y' },
    ])
    expect(text.slice(spans[0].start, spans[0].end)).toBe('a😀b')
    expect(text.slice(spans[1].start, spans[1].end)).toBe('|tail')
  })

  it('sizes a segment in code points even after its offsets are re-addressed for slicing', () => {
    const text = 'a😀b|tail'
    const segs = segmentsFor(text, [
      { start: 0, end: 3, label: 'x' },
      { start: 3, end: 8, label: 'y' },
    ])
    expect(segs[0].chars).toBe(3)
    expect(text.slice(segs[0].start, segs[0].end)).toBe('a😀b')
    expect(segs[1].chars).toBe(5)
  })

  it('is the identity on BMP-only text and clamps an offset past the end', () => {
    expect(spansToUtf16('abc', [{ start: 0, end: 3, label: 'x' }])).toEqual([{ start: 0, end: 3, label: 'x' }])
    expect(spansToUtf16('abc', [{ start: 1, end: 9, label: 'x' }])).toEqual([{ start: 1, end: 3, label: 'x' }])
  })
})

describe('segmentNames', () => {
  const name = (label: string) => label
  it('numbers a label that appears more than once and leaves single ones alone', () => {
    const segs = [
      { label: 'a', start: 0, end: 1, chars: 1 },
      { label: 'b', start: 1, end: 2, chars: 1 },
      { label: 'a', start: 2, end: 3, chars: 1 },
    ]
    expect(segmentNames(segs, name)).toEqual(['a · 1 of 2', 'b', 'a · 2 of 2'])
  })
})

describe('ContextBreakdownPanel with a prompt trace', () => {
  it('omits the section while the prompt trace is not loaded', () => {
    render(<ContextBreakdownPanel trace={trace([turn()])} />)
    expect(screen.queryByTestId('prompt-as-sent')).toBeNull()
  })

  it('shows the selected turn text by segment and marks the turn with a dot', () => {
    const { container } = render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([record()])} />)
    const section = screen.getByTestId('prompt-as-sent')
    expect(within(section).getByText('Prompt as sent')).toBeTruthy()
    // The user's text was carved by the backend, so it has a row of its own and
    // the header row does not claim it.
    expect(within(section).getByText('The same pieces as above, in the order they were sent.')).toBeTruthy()
    expect(within(section).queryByText(/includes your/)).toBeNull()
    const rows = section.querySelectorAll('[data-prompt-segment]')
    expect(Array.from(rows).map(r => r.getAttribute('data-prompt-segment'))).toEqual([
      'critical_rules',
      'request_header',
      'your_message',
      'reply_format_rules',
    ])
    expect(container.querySelectorAll('[data-prompt-dot]').length).toBe(1)
    expect(screen.getByTestId('prompt-kept-legend')).toBeTruthy()
    expect(screen.getByTestId('prompt-segment-bar').getAttribute('title')).toBe('The pieces below, to scale')
  })

  it('says where the message sits when the backend could not carve it', () => {
    // A record scanned without a user span: the header's span runs through
    // the user's text, so the header row says it includes the message.
    const uncarved = record({
      spans: [
        { start: 0, end: HEADER, label: 'critical_rules' },
        { start: HEADER, end: USER + 'hello there'.length, label: 'request_header' },
        { start: USER + 'hello there'.length, end: TEXT.length, label: 'reply_format_rules' },
      ],
    })
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([uncarved])} />)
    const section = screen.getByTestId('prompt-as-sent')
    // The row's NAME says it carries the message, so its larger total is
    // explained before any aside; the aside then carries the message's own
    // count (the turn fixture says 11), so the reader is not left to subtract
    // the summary total from the row's.
    expect(within(section).getByText(/^Message wrapper \(built-in\) \+ your message · 11 of its \d+ characters are your message$/)).toBeTruthy()
    // The helper is the same one sentence either way: where the message sits is
    // said on the row suffix and the summary aside, not a third time here.
    expect(within(section).getByText('The same pieces as above, in the order they were sent.')).toBeTruthy()
    expect(section.querySelector('[data-prompt-segment="your_message"]')).toBeNull()
    // The summary's own "Your message" total says where its count sits below,
    // naming the number, so the two lists do not read as counting one thing
    // two ways.
    const userRow = document.querySelector('[data-category-row="message"]')!
    expect(userRow.textContent).toMatch(/its 11 characters sit in the message wrapper below, counted once there/)
    // The chevron stays beside the name; the aside follows it.
    const chevron = userRow.querySelector('svg')!
    const aside = within(userRow as HTMLElement).getByText(/sit in the message wrapper below/)
    expect(chevron.compareDocumentPosition(aside) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('does not annotate the summary row when the message has its own prompt row', () => {
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([record()])} />)
    expect(document.querySelector('[data-category-row="message"]')!.textContent).not.toMatch(/sit in the message wrapper/)
  })

  it('repeats the size in the footer only when it is not the heading number', () => {
    render(<ContextBreakdownPanel trace={trace([turn({ total_chars: TEXT.length })])} prompts={prompts([record()])} />)
    expect(screen.getByTestId('prompt-matched-at').textContent).not.toMatch(new RegExp(`^${TEXT.length} characters`))
    cleanup()
    // A substituted receipt: sent differs from the heading, so the count is worth a line.
    render(
      <ContextBreakdownPanel
        trace={trace([turn({ total_chars: TEXT.length + 400 })])}
        prompts={prompts([record({ assembled_chars: TEXT.length + 400 })])}
      />,
    )
    expect(screen.getByTestId('prompt-matched-at').textContent).toContain(`${TEXT.length} characters`)
  })

  it('opens a segment to the raw slice of the prompt', () => {
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([record()])} />)
    const row = screen.getByTestId('prompt-as-sent').querySelector('[data-prompt-segment="your_message"] button')
    expect(row).not.toBeNull()
    fireEvent.click(row!)
    expect(screen.getByText('hello there')).toBeTruthy()
  })

  it('says how many earlier turns fell out, and when the whole session was evicted', () => {
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([record()], { dropped: 3 })} />)
    expect(screen.getByTestId('prompt-dropped').textContent).toMatch(/^Newest turns only: prompt text for 3 earlier turns .* trimmed/)
    cleanup()
    // The 13th turn drops exactly one: a plural key, so it does not say "1 earlier turns".
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([record()], { dropped: 1 })} />)
    expect(screen.getByTestId('prompt-dropped').textContent).toMatch(/for 1 earlier turn of this session/)
    cleanup()
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([], { evicted: true })} />)
    // Two verbs for two mechanisms: the per-session cap "trimmed" older turns; the
    // global budget "removed" the whole session.
    expect(within(screen.getByTestId('prompt-as-sent')).getByText(/^Back with this session's next turn: .*removed so other sessions could keep theirs/)).toBeTruthy()
    expect(within(screen.getByTestId('prompt-as-sent')).queryByText(/trimmed/)).toBeNull()
  })

  it('says when only a prefix of the prompt is kept, counting code points', () => {
    // An emoji in the kept prefix: one code point, two UTF-16 units. The note
    // must agree with the rows above it, which count code points.
    const emojiText = TEXT.replace('hello there', 'hello 🙂here')
    const emojiSpans = record().spans.map(sp => ({ ...sp }))
    render(
      <ContextBreakdownPanel
        trace={trace([turn({ total_chars: 9_000_000 })])}
        prompts={prompts([
          record({ truncated: true, chars: 9_000_000, assembled_chars: 9_000_000, text: emojiText, spans: emojiSpans }),
        ])}
      />,
    )
    const truncated = screen.getByTestId('prompt-truncated')
    const codePoints = Array.from(emojiText).length
    expect(codePoints).toBe(emojiText.length - 1)
    expect(truncated.textContent).toContain(String(codePoints))
    expect(truncated.textContent).not.toContain(String(emojiText.length))
    // Bridges the rows (kept part) to the totals above (whole prompt).
    expect(truncated.textContent).toContain('The pieces below cover only that kept part.')
    expect(screen.queryByTestId('prompt-mismatch')).toBeNull()
    // Said before the rows, whose sizes sum to the kept text and not the heading.
    const firstRow = screen.getByTestId('prompt-as-sent').querySelector('[data-prompt-segment]')!
    expect(truncated.compareDocumentPosition(firstRow) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('warns when the matched text is not the size the turn was', () => {
    // The usage row and the record come from one string when the join is right.
    render(<ContextBreakdownPanel trace={trace([turn({ total_chars: 156 })])} prompts={prompts([record()])} />)
    const mismatch = screen.getByTestId('prompt-mismatch').textContent ?? ''
    expect(mismatch).toContain('156')
    // Opens with the number to distrust (the text's own prompt size), each
    // number saying what it is in plain words (no "prepared"), then the one
    // action; no rationale sentence in between.
    expect(mismatch).toMatch(/^Likely a wrong match: this text belongs to a .*-character prompt, but this turn's breakdown counted 156/)
    expect(mismatch).not.toMatch(/prepared/i)
    // The warning names both numbers; the footer does not add a third one.
    // Under the warning the footer keeps only what still holds — the send time
    // and where the text lives — not the count, the by-time match or the
    // "exact text this turn sent" claim the warning says may be wrong.
    const footer = screen.getByTestId('prompt-matched-at').textContent ?? ''
    expect(footer).not.toMatch(/characters sent/)
    expect(footer).not.toMatch(/Matched to this turn/)
    expect(footer).not.toMatch(/exact text this turn sent/)
    expect(footer).toMatch(/^Sent .*\. Held in memory only; restarting .* clears it\.$/)
    expect(mismatch).toMatch(/Pick a neighbouring turn to find its text\.$/)
    // An alert of the ErrorNotice shape, placed ABOVE the segment bar: the two
    // grey notes are footnotes, and the warning must be met before the rows.
    const alert = screen.getByTestId('prompt-mismatch')
    expect(alert.getAttribute('role')).toBe('alert')
    const bar = screen.getByTestId('prompt-segment-bar')
    expect(alert.compareDocumentPosition(bar) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('compares against the ASSEMBLED size and explains a shorter wire text instead of warning', () => {
    // A member session in its acknowledged steady state: the essentials
    // envelope is dropped between the size measurement and the transport write,
    // so the usage row (assembled) and the record (sent) legitimately differ.
    const assembled = TEXT.length + 400
    render(
      <ContextBreakdownPanel
        trace={trace([turn({ total_chars: assembled })])}
        prompts={prompts([record({ assembled_chars: assembled })])}
      />,
    )
    expect(screen.queryByTestId('prompt-mismatch')).toBeNull()
    const line = screen.getByTestId('prompt-assembled-vs-sent').textContent ?? ''
    expect(line).toContain(String(assembled))
    expect(line).toContain(String(TEXT.length))
  })

  it('says when the served text had a secret masked', () => {
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([record({ redacted: true })])} />)
    expect(screen.getByTestId('prompt-redacted').textContent).toMatch(/masked/)
    cleanup()
    render(<ContextBreakdownPanel trace={trace([turn()])} prompts={prompts([record()])} />)
    expect(screen.queryByTestId('prompt-redacted')).toBeNull()
  })

  it('says plainly when the selected turn has no text kept', () => {
    const t1 = turn({ ts: '2026-08-04T00:00:30Z' })
    const t2 = turn({ ts: '2026-08-04T00:05:30Z' })
    // Only the SECOND turn's prompt is still in the ring.
    const kept = record({ ts: '2026-08-04T00:05:00Z' })
    const { container } = render(<ContextBreakdownPanel trace={trace([t1, t2])} prompts={prompts([kept])} />)
    expect(container.querySelectorAll('[data-prompt-dot]').length).toBe(1)
    // Newest is selected by default and has text; pick turn 1.
    fireEvent.click(container.querySelector('button[data-turn="1"]')!)
    // Each text-gone note leads with its consequence; this one is not coming back.
    expect(within(screen.getByTestId('prompt-as-sent')).getByText(/^Not coming back: no prompt text is kept for this turn/)).toBeTruthy()
  })
  it('drops a copy failure when the selected turn changes', async () => {
    // The clipboard mock refuses, so Copy all raises the notice. Selecting
    // another turn re-renders the section in place with a different record;
    // the notice belongs to the turn it happened on and must not follow.
    const t1 = turn({ ts: '2026-08-04T00:00:30Z' })
    const t2 = turn({ ts: '2026-08-04T00:05:30Z' })
    const r1 = record({ ts: '2026-08-04T00:00:00Z' })
    const r2 = record({ ts: '2026-08-04T00:05:00Z', text: TEXT.replace('hello there', 'second turn') })
    render(<ContextBreakdownPanel trace={trace([t1, t2])} prompts={prompts([r1, r2])} />)
    const section = screen.getByTestId('prompt-as-sent')
    fireEvent.click(within(section).getByRole('button', { name: /Copy all/ }))
    expect(await within(section).findByText(/Couldn't copy/)).toBeTruthy()
    fireEvent.click(document.querySelector('button[data-turn="1"]')!)
    expect(within(screen.getByTestId('prompt-as-sent')).queryByText(/Couldn't copy/)).toBeNull()
  })

})
