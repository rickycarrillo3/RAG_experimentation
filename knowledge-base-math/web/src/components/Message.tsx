import { useState } from 'react'

import { apiFetch } from '../lib/sse'
import type { Turn } from '../lib/types'
import { AnswerMeta } from './AnswerMeta'
import { Markdown } from './Markdown'

export function Message({ turn, streaming }: { turn: Turn; streaming: boolean }) {
  if (turn.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent-dim px-4 py-2.5 text-[0.95rem] whitespace-pre-wrap">
          {turn.clean}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <div
        className={`max-w-[92%] rounded-2xl rounded-bl-sm px-4 py-3 text-[0.95rem] leading-relaxed ${
          turn.failed ? 'bg-surface-2 text-warn' : 'bg-surface-2'
        }`}
      >
        {turn.segments.map((seg, i) =>
          // Server-written runs — the Sources footer, the truncation notice, the agent's
          // search and tool markers — are styled as the server's voice rather than the
          // model's. They are already excluded from `clean`; this is the visual half of
          // the same distinction.
          seg.server ? (
            <div key={i} className="mt-2 text-[0.85rem] text-ink-faint italic">
              <Markdown>{seg.text}</Markdown>
            </div>
          ) : (
            <Markdown key={i}>{seg.text}</Markdown>
          ),
        )}
        {streaming && <span className="ml-0.5 inline-block w-2 animate-pulse">▍</span>}
      </div>
      {turn.done && (
        <div className="flex flex-wrap items-start gap-3">
          <AnswerMeta done={turn.done} />
          <Feedback eventId={turn.done.event_id} />
        </div>
      )}
    </div>
  )
}

/**
 * Thumbs feed the telemetry log (kbm/telemetry.py) — where gold-set questions and the
 * embedding fine-tune pairs eventually come from. One vote per answer.
 */
function Feedback({ eventId }: { eventId: string }) {
  const [state, setState] = useState<'' | 'up' | 'down' | 'error'>('')

  async function vote(rating: 'up' | 'down') {
    setState(rating)
    try {
      const res = await apiFetch('/feedback', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ event_id: eventId, rating }),
      })
      if (!res.ok) throw new Error(String(res.status))
    } catch {
      setState('error')
    }
  }

  if (state === 'error') return <span className="text-[0.78rem] text-warn">couldn't record that</span>
  return (
    <span className="flex gap-1 text-[0.85rem]">
      {(['up', 'down'] as const).map((r) => (
        <button
          key={r}
          type="button"
          disabled={state !== ''}
          onClick={() => vote(r)}
          aria-label={r === 'up' ? 'Helpful' : 'Not helpful'}
          className={`rounded px-1.5 py-0.5 hover:bg-surface-2 disabled:cursor-default ${
            state && state !== r ? 'opacity-30' : ''
          }`}
        >
          {r === 'up' ? '👍' : '👎'}
        </button>
      ))}
    </span>
  )
}
