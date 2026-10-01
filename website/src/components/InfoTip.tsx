import { useState, useRef, useEffect, useId, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { Info } from 'lucide-react'

import { i18nT } from '../i18n/t'

/**
 * Small info glyph that reveals a help tip.
 *
 * One pattern for every surface, so a reader learns it once:
 *   - mouse:    hover shows the tip, leaving the glyph hides it;
 *   - keyboard: focus shows it, Escape hides it, blur hides it;
 *   - touch:    a tap pins it, another tap (or a tap elsewhere) unpins it.
 * A click pins on any input. A hover-only tip is a glance that never takes the
 * pointer, so a click aimed at the row under it lands on the row; pinning is
 * what makes the bubble interactive, for reading or copying a long tip, and a
 * pinned tip closes on a pointer down, focus or Escape outside it. While the
 * tip is shown it is the glyph's
 * accessible DESCRIPTION (`aria-describedby`); while hidden, the same text sits
 * on `title`, the native fallback that is also what assistive technology falls
 * back to for a description -- so the help is attached whether or not the tip
 * is open, and nothing of it is in the body copy until it is. The `title` is
 * dropped the moment the tip shows, so the browser's own tooltip never doubles
 * it. Portal-rendered to escape overflow clipping.
 */
/** How long the tip survives after the pointer leaves the glyph or the bubble. */
export const HOVER_LEAVE_GRACE_MS = 150

export default function InfoTip({ text, placement = 'auto' }: {
  text: string
  placement?: 'auto' | 'top'
}) {
  const [pinned, setPinned] = useState(false)
  const [hovered, setHovered] = useState(false)
  const [focused, setFocused] = useState(false)
  // Set by Escape or unpinning: suppresses hover/focus reveal until the next
  // mouse enter or blur, never a timer, so a focused tip stays dismissed.
  const [dismissed, setDismissed] = useState(false)
  const btnRef = useRef<HTMLButtonElement>(null)
  const tipRef = useRef<HTMLDivElement>(null)
  // Hover ends after a short grace rather than on the leave event itself, so a
  // pointer that skims off the glyph and back does not flicker the tip, and a
  // pinned bubble stays hovered across the 6px gap between glyph and bubble.
  // Entering either element cancels the pending close.
  const leaveTimer = useRef<number | null>(null)
  const cancelLeave = () => {
    if (leaveTimer.current !== null) { window.clearTimeout(leaveTimer.current); leaveTimer.current = null }
  }
  const scheduleLeave = () => {
    cancelLeave()
    leaveTimer.current = window.setTimeout(() => {
      leaveTimer.current = null
      setHovered(false)
    }, HOVER_LEAVE_GRACE_MS)
  }
  useEffect(() => cancelLeave, [])
  const tipId = useId()
  const open = pinned || (!dismissed && (hovered || focused))

  // A pinned tip outlives the trigger's focus by design (a reader clicks into
  // the bubble to copy text, which blurs the glyph), so blur cannot unpin it.
  // The dismissals must then live on the document while pinned: a pointer
  // down outside, focus moving outside (Tab onward), or Escape from wherever
  // focus has landed -- the trigger's own Escape handler is unreachable once
  // the glyph is no longer focused.
  useEffect(() => {
    if (!pinned) return
    const outside = (target: EventTarget | null) =>
      !btnRef.current?.contains(target as Node) && !tipRef.current?.contains(target as Node)
    const onMouseDown = (e: MouseEvent) => { if (outside(e.target)) setPinned(false) }
    const onFocusIn = (e: FocusEvent) => { if (outside(e.target)) setPinned(false) }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      setPinned(false)
      setDismissed(true)
    }
    document.addEventListener('mousedown', onMouseDown)
    document.addEventListener('focusin', onFocusIn)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onMouseDown)
      document.removeEventListener('focusin', onFocusIn)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [pinned])

  /**
   * Where the bubble goes, decided from the glyph's rect alone -- no second
   * paint, no measurement. The bubble's own height is unknown at this point
   * and can be large (a paragraph moved out of a row's body copy), so the
   * side is chosen by ROOM rather than by a nominal height: anchor the bubble
   * to whichever vertical side of the glyph has more space and cap its
   * `maxHeight` to that space, and anything longer scrolls inside it. A
   * `position: fixed` element that ran past the viewport edge could never be
   * scrolled to, which is what this prevents.
   */
  const pos = (): CSSProperties => {
    if (!btnRef.current) return { top: 0, left: 0 }
    const r = btnRef.current.getBoundingClientRect()
    const tipW = 300
    const vh = window.innerHeight
    // The bubble's ceiling: 60% of the viewport, at most 480px, and never
    // more than the space on the chosen side less the 8px margin.
    const cap = Math.min(480, Math.round(vh * 0.6))
    if (placement === 'top') {
      // Centered above the button, bottom-anchored so height doesn't matter.
      let left = r.left + r.width / 2 - tipW / 2
      left = Math.max(8, Math.min(left, window.innerWidth - tipW - 8))
      const above = r.top - 6 - 8
      const below = vh - r.bottom - 6 - 8
      if (above >= Math.min(cap, below)) return { bottom: vh - r.top + 6, left, maxHeight: Math.min(cap, above) }
      return { top: r.bottom + 6, left, maxHeight: Math.min(cap, below) } // not enough room above → fall below
    }
    let left = r.right + 6
    if (left + tipW > window.innerWidth) left = r.left - tipW - 6
    // The left-flip can land past the left edge when the button sits within
    // tipW of it (any narrow viewport); clamp to the same 8px margins the
    // 'top' branch uses so the tip always stays readable on-screen.
    left = Math.max(8, Math.min(left, window.innerWidth - tipW - 8))
    // Beside the glyph: top-aligned with it when the space below reaches the
    // cap, otherwise bottom-aligned with it so a tall bubble grows upward.
    const below = vh - r.top - 8
    if (below >= cap) return { top: r.top, left, maxHeight: cap }
    const above = r.bottom - 8
    if (above > below) return { bottom: Math.max(8, vh - r.bottom), left, maxHeight: Math.min(cap, above) }
    return { top: r.top, left, maxHeight: Math.max(0, below) }
  }

  return (
    <>
      <button
        ref={btnRef}
        /* A <button> defaults to type="submit": inside a <form> (the New crewmate
           dialog wraps its fields in one) a bare toggle would submit the owner form
           on click. This is a help toggle, never a submitter. */
        type="button"
        /* The glyph alone would be announced as "info" or nothing -- a control with
           no discoverable purpose. The NAME is a short generic verb phrase and the
           tip text is attached as the DESCRIPTION instead, which matters for two
           reasons: a name is read whenever the button is reached, so a
           paragraph-length one is unusable; and an accessible name is the handle
           every other control is found by, so naming a help toggle after its own
           prose makes it collide with real actions whose labels happen to appear
           in that prose. */
        aria-label={i18nT('components.infoTip.more_information')}
        aria-expanded={open}
        aria-describedby={open ? tipId : undefined}
        title={open ? undefined : text}
        /* Unpinning also dismisses the hover/focus reveal: the click that
           unpins leaves the trigger focused (and, on a mouse, hovered), so
           without this the tip would stay open and a second tap could never
           close it. The dismissal lifts on the next mouse enter or blur, never
           on the pointer-leave timer. */
        onClick={(e) => { e.stopPropagation(); setDismissed(pinned); setPinned(!pinned) }}
        /* Hover is a mouse affordance only. A finger has no hover, and the
           synthetic enter a tap fires would race the click that pins. */
        onPointerEnter={(e) => { if (e.pointerType === 'mouse') { cancelLeave(); setHovered(true); setDismissed(false) } }}
        onPointerLeave={scheduleLeave}
        onFocus={() => setFocused(true)}
        onBlur={() => { setFocused(false); setDismissed(false) }}
        onKeyDown={(e) => {
          if (e.key !== 'Escape' || !open) return
          e.stopPropagation()
          setPinned(false)
          setDismissed(true)
        }}
        className={`w-4 h-4 rounded-full transition-colors leading-none cursor-pointer flex items-center justify-center shrink-0 ${open ? 'text-text' : 'text-muted hover:text-text'}`}
      >
        <Info size={14} aria-hidden="true" />
      </button>
      {open && createPortal(
        // Portal clicks bubble through React into clickable owners (SettingsToggle),
        // whose physical-ancestry guard cannot see the tip. Swallowing adds no interaction.
        // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions
        <div
          ref={tipRef}
          id={tipId}
          role="tooltip"
          onClick={(e) => e.stopPropagation()}
          /* Once pinned, the bubble is part of the hover target: entering it
             cancels the close the glyph's leave scheduled, leaving it schedules
             one. Hover-only, these never fire -- the bubble is pointer-transparent. */
          onPointerEnter={(e) => { if (e.pointerType === 'mouse') cancelLeave() }}
          onPointerLeave={scheduleLeave}
          /* A hover-only bubble is a glance, never a pointer target: it sits
             over the neighbouring row and a click aimed there must land there,
             not be swallowed by the overlay, and the pointer moving on must not
             keep it alive. Pinning (click or tap) is what makes the bubble
             interactive, for reading or copying a long tip. */
          className={`fixed z-[9999] rounded-lg border border-border p-2.5 text-[12px] text-muted leading-relaxed max-w-[300px] max-h-[min(60vh,480px)] overflow-y-auto whitespace-normal ${pinned ? '' : 'pointer-events-none'}`}
          style={{ ...pos(), backgroundColor: 'var(--card)', boxShadow: '0 4px 24px rgba(0,0,0,0.5)' }}
        >
          {text}
        </div>,
        document.body
      )}
    </>
  )
}
