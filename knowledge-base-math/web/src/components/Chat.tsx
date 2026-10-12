import { useEffect, useRef, useState } from 'react'

import { buildHistory } from '../lib/history'
import { streamChat } from '../lib/sse'
import { appendSegment, type Turn } from '../lib/types'
import { Composer } from './Composer'
import { Message } from './Message'
import { SourcesPanel } from './SourcesPanel'

export function Chat({ user }: { user: string }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const abort = useRef<AbortController | null>(null)
  const scroller = useRef<HTMLDivElement>(null)

  // Follow the stream. Token frames arrive many times a second, so this runs often and
  // must stay cheap.
  useEffect(() => {
    const el = scroller.current
    if (el) el.scrollTop = el.scrollHeight
  }, [turns, status])

  useEffect(() => () => abort.current?.abort(), [])

  /** Patch the assistant turn currently being streamed into. */
  const patch = (fn: (t: Turn) => Turn) =>
    setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? fn(t) : t)))

  async function send(message: string) {
    if (!message || busy) return

    // History is built from the turns as they stand BEFORE this question is appended:
    // the current message travels as `message`, never inside `history`.
    const history = buildHistory(turns)
    setTurns((ts) => [
      ...ts,
      { role: 'user', segments: [{ text: message, server: false }], clean: message },
      { role: 'assistant', segments: [], clean: '' },
    ])
    setBusy(true)
    setStatus('Searching your documents…')

    const ac = new AbortController()
    abort.current = ac

    try {
      for await (const ev of streamChat({ user, message, history }, ac.signal)) {
        switch (ev.type) {
          case 'sources':
            patch((t) => ({ ...t, sources: ev.sources, mode: ev.mode }))
            // Retrieval is done; the rest of the wait is decoding, and there is no
            // heartbeat to confirm the connection is alive, so say what is happening.
            setStatus('Thinking…')
            break
          case 'token':
            setStatus('')
            patch((t) => ({
              ...t,
              segments: appendSegment(t.segments, ev.text, ev.server_marker),
              // The clean half of the split: server-written text is shown but never
              // sent back as something the model said.
              clean: ev.server_marker ? t.clean : t.clean + ev.text,
            }))
            break
          case 'done':
            patch((t) => ({
              ...t,
              done: ev,
              mode: ev.mode,
              // `done.sources` is the complete list, including anything a mid-answer
              // search found after the `sources` frame had already been sent.
              sources: ev.sources.length ? ev.sources : t.sources,
            }))
            break
          case 'error':
            throw new Error(ev.message)
        }
      }
    } catch (e) {
      const aborted = ac.signal.aborted
      const text = aborted
        ? '_Stopped._'
        : `**Error:** ${e instanceof Error ? e.message : String(e)}`
      // A failed turn is marked rather than deleted: the student should see what went
      // wrong, while buildHistory drops it along with its question.
      patch((t) => ({
        ...t,
        segments: t.segments.length ? [...t.segments, { text: `\n\n${text}`, server: true }] : [{ text, server: true }],
        failed: !aborted || !t.clean.trim(),
      }))
    } finally {
      setBusy(false)
      setStatus('')
      abort.current = null
    }
  }

  const last = turns[turns.length - 1]
  const latest = [...turns].reverse().find((t) => t.role === 'assistant' && t.sources !== undefined)

  return (
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <section className="flex min-h-0 flex-col gap-3">
        <div ref={scroller} className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
          {turns.length === 0 && (
            <p className="mt-10 text-center text-sm text-ink-faint">
              Ask a question about your textbooks.
            </p>
          )}
          {turns.map((t, i) => (
            <Message key={i} turn={t} streaming={busy && i === turns.length - 1} />
          ))}
          {status && <p className="text-sm text-ink-faint italic">{status}</p>}
        </div>

        <Composer busy={busy} onSend={send} onStop={() => abort.current?.abort()} />
      </section>

      <SourcesPanel
        sources={latest?.sources}
        mode={latest?.mode}
        pending={busy && last?.sources === undefined}
      />
    </div>
  )
}
