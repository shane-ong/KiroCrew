import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import InfoTip, { HOVER_LEAVE_GRACE_MS } from '../components/InfoTip'

const TIP_W = 300 // matches tipW in InfoTip's pos()

const setInnerWidth = (w: number) =>
  Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: w })
const setInnerHeight = (h: number) =>
  Object.defineProperty(window, 'innerHeight', { writable: true, configurable: true, value: h })

const stubRect = (el: HTMLElement, rect: { left: number; top: number; right: number; bottom: number }) => {
  el.getBoundingClientRect = () =>
    ({
      ...rect,
      width: rect.right - rect.left,
      height: rect.bottom - rect.top,
      x: rect.left,
      y: rect.top,
      toJSON: () => ({}),
    }) as DOMRect
}

const trigger = () => screen.getByRole('button', { name: 'More information' })
const hover = (el: HTMLElement) => fireEvent.pointerEnter(el, { pointerType: 'mouse' })
const unhover = (el: HTMLElement) => fireEvent.pointerLeave(el, { pointerType: 'mouse' })
// The hover close is deferred by a short grace; settle it.
const settle = () => act(() => { vi.advanceTimersByTime(HOVER_LEAVE_GRACE_MS + 1) })

describe('InfoTip', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); setInnerWidth(1024); setInnerHeight(768) })

  it('is named by a short phrase, and describes itself with the tip text', () => {
    // The glyph carries no text, so the NAME is a short generic phrase and the
    // tip prose is the DESCRIPTION: a name is read on every visit and is the
    // handle other controls are queried by, so a paragraph-length one both
    // talks over the user and collides with real actions named inside it.
    // Closed, the text sits on `title` -- the native fallback, and nothing of it
    // is body copy; open, the visible tooltip is referenced instead and `title`
    // is dropped so the browser cannot show a second tooltip over ours.
    render(<InfoTip text="What this binding does" />)
    const btn = trigger()
    expect(btn).toHaveAttribute('aria-expanded', 'false')
    expect(btn).toHaveAttribute('title', 'What this binding does')
    expect(btn).not.toHaveAttribute('aria-describedby')
    expect(screen.queryByText('What this binding does')).not.toBeInTheDocument()

    fireEvent.click(btn)
    expect(btn).toHaveAttribute('aria-expanded', 'true')
    expect(btn).not.toHaveAttribute('title')
    const tip = screen.getByRole('tooltip')
    expect(tip).toHaveTextContent('What this binding does')
    expect(btn.getAttribute('aria-describedby')).toBe(tip.id)
  })

  it('shows on mouse hover and hides when the pointer leaves', () => {
    render(<InfoTip text="Hover help" />)
    hover(trigger())
    expect(screen.getByRole('tooltip')).toHaveTextContent('Hover help')
    unhover(trigger())
    // Still there for the grace -- the pointer may be on its way to the bubble.
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    settle()
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('keeps a focused tip closed after the second tap and its pointer leave', () => {
    render(<InfoTip text="Touch toggle help" />)
    const btn = trigger()
    fireEvent.focus(btn)
    fireEvent.click(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.pointerLeave(btn, { pointerType: 'touch' })
    settle()
    fireEvent.click(btn)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    fireEvent.pointerLeave(btn, { pointerType: 'touch' })
    settle()
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    fireEvent.blur(btn)
    fireEvent.focus(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
  })

  it('a hover-only tip never takes the pointer; a pinned one does', () => {
    // The bubble is a fixed overlay beside the glyph, over the next row. On
    // hover alone it is a glance: a click aimed at the row under it must reach
    // the row, not be swallowed, and the pointer moving on must not keep it
    // alive. Pinning is what makes it a target for reading or copying.
    render(<InfoTip text="Glance help" />)
    const btn = trigger()
    hover(btn)
    expect(screen.getByRole('tooltip').className).toContain('pointer-events-none')
    fireEvent.click(btn)
    expect(screen.getByRole('tooltip').className).not.toContain('pointer-events-none')
  })

  it('stays open while the pointer moves from the glyph onto a pinned tip', () => {
    // Reading or copying a long tip means pinning it, then crossing the gap
    // between glyph and bubble; the trigger's leave fires on the way, so the
    // pinned bubble counts as hovered until the pointer leaves it too, and
    // unpinning it then lets the leave close it.
    render(<InfoTip text="Long help to copy" />)
    const btn = trigger()
    hover(btn)
    fireEvent.click(btn)
    const tip = screen.getByRole('tooltip')
    unhover(btn)
    fireEvent.pointerEnter(tip, { pointerType: 'mouse' })
    settle()
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.pointerLeave(tip, { pointerType: 'mouse' })
    settle()
    expect(screen.getByRole('tooltip')).toBeInTheDocument() // pinned: leaving does not close
    fireEvent.mouseDown(document.body) // unpin
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('does not treat a touch pointer entering as hover', () => {
    // A tap fires a synthetic enter before its click. Treating it as hover
    // would leave the tip open after the click that was meant to toggle it.
    render(<InfoTip text="Touch help" />)
    fireEvent.pointerEnter(trigger(), { pointerType: 'touch' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    fireEvent.click(trigger())
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.click(trigger())
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('shows on keyboard focus, hides on Escape, and shows again on the next focus', () => {
    render(<InfoTip text="Focus help" />)
    const btn = trigger()
    fireEvent.focus(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.keyDown(btn, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    // Escape dismissed only this showing: moving away and back reveals it again.
    fireEvent.blur(btn)
    fireEvent.focus(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.blur(btn)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()

    fireEvent.focus(btn)
    hover(btn)
    fireEvent.keyDown(btn, { key: 'Escape' })
    unhover(btn)
    settle()
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    hover(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
  })

  it('a click pins the tip so leaving no longer hides it', () => {
    render(<InfoTip text="Pinned help" />)
    const btn = trigger()
    hover(btn)
    fireEvent.click(btn)
    unhover(btn)
    settle()
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
  })

  it('a second click closes a pinned tip even while the trigger stays focused', () => {
    // A tap focuses the trigger as it clicks. Focus alone would keep the tip
    // open after the unpin, so a touch reader could never close it.
    render(<InfoTip text="Toggle help" />)
    const btn = trigger()
    fireEvent.focus(btn)
    fireEvent.click(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.click(btn)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    // Leaving and returning reveals it again: the dismissal was for this visit only.
    fireEvent.blur(btn)
    fireEvent.focus(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
  })

  it('unpins on an outside click', () => {
    render(<InfoTip text="Tip content" />)
    fireEvent.click(trigger())
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.mouseDown(document.body)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('unpins when focus moves to another control', () => {
    // A tip pinned with Enter outlives the glyph's blur on purpose (clicking
    // into the bubble to copy blurs the glyph too), so Tab alone leaves it
    // open; focus landing outside glyph and bubble is what closes it.
    render(
      <>
        <InfoTip text="Keyboard pinned help" />
        <button type="button">Next control</button>
      </>,
    )
    const btn = trigger()
    fireEvent.focus(btn)
    fireEvent.click(btn) // Enter on a focused button fires click
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.blur(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument() // pinned survives blur
    fireEvent.focus(screen.getByRole('button', { name: 'Next control' }))
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('a pinned tip closes on Escape after the glyph has lost focus', () => {
    render(<InfoTip text="Escape anywhere help" />)
    const btn = trigger()
    fireEvent.focus(btn)
    fireEvent.click(btn)
    fireEvent.blur(btn)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('does not submit an enclosing form', () => {
    render(<InfoTip text="In a form" />)
    expect(trigger()).toHaveAttribute('type', 'button')
  })

  it('keeps auto placement on-screen on a narrow viewport', () => {
    // Phone-width regression: right-side placement overflows, and the left-flip
    // (r.left - tipW - 6) goes far negative for a button near the left edge.
    // Unclamped, the tip renders mostly past the left viewport edge.
    setInnerWidth(390)
    render(<InfoTip text="Narrow viewport tip" />)
    const btn = trigger()
    stubRect(btn, { left: 100, top: 200, right: 116, bottom: 216 })
    fireEvent.click(btn)
    const tip = screen.getByRole('tooltip')
    const left = parseFloat(tip.style.left)
    expect(left).toBeGreaterThanOrEqual(8)
    expect(left + TIP_W).toBeLessThanOrEqual(390) // fully on-screen
  })

  it('still flips left of the button when the flipped position fits', () => {
    // The clamp must not defeat the flip: a button near the RIGHT edge flips
    // left and the flipped value already fits, so it is used as-is.
    setInnerWidth(390)
    render(<InfoTip text="Right edge tip" />)
    const btn = trigger()
    stubRect(btn, { left: 350, top: 200, right: 366, bottom: 216 })
    fireEvent.click(btn)
    const tip = screen.getByRole('tooltip')
    // flipped: 350 - 300 - 6 = 44; inside [8, 390-300-8=82], so unchanged.
    expect(parseFloat(tip.style.left)).toBe(44)
  })

  it('anchors a tip for a glyph low in the viewport to its bottom edge and caps its height', () => {
    // The bubble's height is unknown when it is placed and can be a whole
    // paragraph, so the side is chosen by room: a glyph in the lower half gets
    // a bubble that grows upward from its bottom edge, with maxHeight capped to
    // the space above, and anything longer scrolls inside the bubble. A fixed
    // element that ran past the bottom edge could never be scrolled to.
    setInnerHeight(600)
    render(<InfoTip text="A long tip" />)
    const btn = trigger()
    stubRect(btn, { left: 100, top: 480, right: 116, bottom: 496 })
    fireEvent.click(btn)
    const tip = screen.getByRole('tooltip')
    expect(tip.style.top).toBe('')
    expect(parseFloat(tip.style.bottom)).toBe(600 - 496)
    expect(parseFloat(tip.style.maxHeight)).toBeLessThanOrEqual(496 - 8)
    expect(tip.className).toContain('overflow-y-auto')
    expect(tip.className).toMatch(/max-h-\[/)
  })

  it('keeps a glyph high in the viewport top-aligned with the full cap', () => {
    setInnerHeight(600)
    render(<InfoTip text="Short tip" />)
    const btn = trigger()
    stubRect(btn, { left: 100, top: 100, right: 116, bottom: 116 })
    fireEvent.click(btn)
    const tip = screen.getByRole('tooltip')
    expect(parseFloat(tip.style.top)).toBe(100)
    expect(parseFloat(tip.style.maxHeight)).toBe(Math.min(480, Math.round(600 * 0.6)))
  })
})
