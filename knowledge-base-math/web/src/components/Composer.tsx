import { useState } from 'react'

export function Composer({
  busy,
  onSend,
  onStop,
}: {
  busy: boolean
  onSend: (message: string) => void
  onStop: () => void
}) {
  const [text, setText] = useState('')

  function submit() {
    const message = text.trim()
    if (!message || busy) return
    setText('')
    onSend(message)
  }

  return (
    <form
      className="flex items-end gap-2"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          // Enter sends, Shift+Enter writes a newline — a multi-line question is
          // common when it carries an equation.
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
          }
        }}
        rows={2}
        placeholder="e.g. How do I solve a quadratic equation?"
        className="min-h-[3rem] flex-1 resize-y rounded-xl border border-edge bg-surface-2 px-3.5 py-2.5
          text-[0.95rem] outline-none placeholder:text-ink-faint focus:border-accent"
      />
      {busy ? (
        <button
          type="button"
          onClick={onStop}
          className="rounded-xl border border-edge px-4 py-2.5 text-sm font-medium text-ink-dim hover:text-ink"
        >
          Stop
        </button>
      ) : (
        <button
          type="submit"
          disabled={!text.trim()}
          className="rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-surface
            disabled:cursor-not-allowed disabled:opacity-40"
        >
          Send
        </button>
      )}
    </form>
  )
}
