import type { Mode, Source } from '../lib/types'
import { ModeBadge } from './ModeBadge'

/**
 * The retrieved chunks, from the `sources` SSE frame.
 *
 * This panel is new capability rather than a port: the frame is emitted *before*
 * generation starts precisely so a UI can fill the seconds the model spends decoding
 * (generation is ~95% of query time), and the Gradio client ignored it — so `score`,
 * `chunk_id` and `preview` have never been visible, despite being computed every query.
 *
 * `score` is the cross-encoder's sigmoid output on 0–1, the same scale the relevance
 * floor is calibrated on, so the bar is directly comparable to the grounded/general
 * decision shown above it.
 */
export function SourcesPanel({
  sources,
  mode,
  pending,
}: {
  sources: Source[] | undefined
  mode: Mode | undefined
  pending: boolean
}) {
  return (
    <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto rounded-xl bg-surface-2/60 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-[0.8rem] font-semibold tracking-wide text-ink-dim uppercase">
          Retrieved
        </h2>
        {mode && <ModeBadge mode={mode} />}
      </div>

      {!sources && (
        <p className="text-sm text-ink-faint">
          {pending ? 'Searching…' : 'Ask something and the chunks it retrieved appear here.'}
        </p>
      )}

      {sources?.length === 0 && (
        <p className="text-sm text-ink-faint">
          Nothing scored above the relevance floor, so the answer is general knowledge.
        </p>
      )}

      {sources?.map((s) => (
        <article key={s.chunk_id} className="rounded-lg bg-surface-3/70 p-3">
          <header className="flex items-baseline justify-between gap-2">
            <span className="truncate text-sm font-medium" title={s.source}>
              {s.source}
            </span>
            <span className="font-mono text-[0.72rem] text-ink-dim">{s.score.toFixed(3)}</span>
          </header>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-surface">
            <div
              className="h-full rounded-full bg-accent"
              style={{ width: `${Math.max(0, Math.min(1, s.score)) * 100}%` }}
            />
          </div>
          <p className="mt-2 line-clamp-4 font-mono text-[0.72rem] leading-snug text-ink-dim">
            {s.preview}
          </p>
          <p className="mt-1.5 truncate font-mono text-[0.68rem] text-ink-faint" title={s.chunk_id}>
            {s.chunk_id}
          </p>
        </article>
      ))}
    </aside>
  )
}
