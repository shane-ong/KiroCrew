import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { SettingsToggle } from '../components/settings'

describe('InfoTip in a settings row', () => {
  it('keeps tooltip clicks separate from row activation', () => {
    const onChange = vi.fn()
    const { container } = render(
      <SettingsToggle label="Example setting" hint="Help text" checked={false} onChange={onChange} />
    )

    fireEvent.click(screen.getByRole('button', { name: 'More information' }))
    expect(onChange).not.toHaveBeenCalled()
    const tip = screen.getByRole('tooltip')
    expect(tip).toHaveTextContent('Help text')
    expect(tip.parentElement).toBe(document.body)
    fireEvent.mouseDown(tip)
    fireEvent.click(tip)
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByRole('tooltip')).toBe(tip)

    fireEvent.click(container.firstElementChild!)
    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith(true)
  })
})
