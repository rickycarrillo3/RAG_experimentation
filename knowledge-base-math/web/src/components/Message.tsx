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
      {turn.done && <AnswerMeta done={turn.done} />}
    </div>
  )
}
