import type { DoneEvent, Timings } from '../lib/types'
import { ModeBadge } from './ModeBadge'

/**
 * What the server said about the answer it just produced.
 *
 * All of this was already on the `done` frame and the Gradio client read exactly one
 * field of it — `event_id` (app.py:333). `mode` is the provenance decision the
 * calibrated relevance floor made up front; the counters say whether the answer was
 * calculated or recalled, and whether the model went looking again mid-answer.
 */
export function AnswerMeta({ done }: { done: DoneEvent }) {
  const chips: string[] = []
  if (done.protocol !== 'none') chips.push(done.protocol)
  if (done.tool_calls) chips.push(`${done.tool_calls} calc${done.tool_calls > 1 ? 's' : ''}`)
  if (done.tool_errors) chips.push(`${done.tool_errors} failed`)
  if (done.searches) chips.push(`${done.searches} re-search`)
  if (done.late_sources) chips.push(`${done.late_sources} late`)
  if (done.continuations) chips.push(`${done.continuations} cont.`)
  if (done.memories_recalled) chips.push(`${done.memories_recalled} memor${done.memories_recalled > 1 ? 'ies' : 'y'}`)

  return (
    <details className="group ml-1 text-[0.78rem] text-ink-faint">
      <summary className="flex cursor-pointer list-none items-center gap-2 select-none">
        <ModeBadge mode={done.mode} />
        {done.truncated && <span className="text-warn">cut off</span>}
        {chips.map((c) => (
          <span key={c} className="rounded bg-surface-2 px-1.5 py-0.5">
            {c}
          </span>
        ))}
        <span className="opacity-50 group-open:hidden">timings</span>
      </summary>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md bg-surface-2 p-3 font-mono">
        {stageRows(done.timings).map(([label, value]) => (
          <div key={label} className="col-span-2 grid grid-cols-subgrid">
            <dt className="text-ink-dim">{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
        <div className="col-span-2 grid grid-cols-subgrid">
          <dt className="text-ink-dim">event</dt>
          <dd className="truncate">{done.event_id}</dd>
        </div>
      </dl>
    </details>
  )
}

/** Only the stages that actually ran: every Timings field is optional for a reason. */
function stageRows(t: Timings): [string, string][] {
  const ms = (v: number | null | undefined) => (v == null ? null : `${Math.round(v)} ms`)
  const rows: [string, string | null][] = [
    ['bm25', ms(t.bm25_ms)],
    ['dense', ms(t.dense_ms)],
    ['rrf', ms(t.rrf_ms)],
    ['rerank', ms(t.rerank_ms)],
    ['ttft', ms(t.ttft_ms)],
    ['generate', ms(t.generate_ms)],
    ['sandbox', ms(t.tool_ms)],
    ['re-search', ms(t.search_ms)],
  ]
  return rows.filter((r): r is [string, string] => r[1] !== null)
}
