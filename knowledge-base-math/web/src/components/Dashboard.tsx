import { useCallback, useEffect, useMemo, useState } from 'react'

import { API_BASE } from '../lib/sse'

/**
 * Operator dashboard over kbm/telemetry.py's JSONL log (GET /telemetry/events).
 *
 * Rows are raw dicts, not generated types: the log is append-only and older rows lack
 * newer fields (`passes`, `tokens`, `at_ms`), so every field here is optional and a
 * missing one renders as "—" rather than as a zero that would drag averages down.
 */

type Pass = {
  at_ms: number
  ms: number
  prompt_tokens?: number | null
  output_tokens?: number | null
  prefill_ms?: number | null
  decode_ms?: number | null
  load_ms?: number | null
  done_reason?: string | null
  tool_calls?: number
}
type ToolEntry = {
  n: number
  tool: string
  ok: boolean
  ms: number
  at_ms?: number
  code?: string
  output?: string
  reason?: string
  query?: string
}
type Source = { source?: string; chunk_id?: string; score?: number; preview?: string }
type QueryRow = {
  kind: 'query'
  event_id: string
  ts: string
  question: string
  retrieval_query?: string | null
  mode: string
  model: string
  protocol?: string
  sources?: Source[]
  late_sources?: Source[]
  timings?: Record<string, number | null | undefined>
  truncated?: boolean
  continuations?: number
  tool_calls?: number
  tool_errors?: number
  tool_counts?: Record<string, number>
  tool_log?: ToolEntry[]
  search_queries?: string[]
  memories_recalled?: number
  passes?: Pass[]
  tokens?: { prompt?: number | null; output?: number | null; mtpt_ms?: number | null; tok_s?: number | null }
  error?: string | null
}
type FeedbackRow = { kind: 'feedback'; event_id: string; rating: 'up' | 'down' }

// Validated (dataviz validate_palette.js, dark, surface #16191f): all checks pass.
// Every bar also carries a text label, so identity is never colour alone.
const KIND_COLOR = {
  retrieval: '#3987e5',
  generation: '#d95926',
  python: '#199e70',
  search: '#c98500',
} as const
const FLOOR = 0.15 // KBM_RELEVANCE_FLOOR default; see api/settings.py

const RETRIEVAL_STAGES = ['bm25_ms', 'dense_ms', 'rrf_ms', 'rerank_ms'] as const

export function Dashboard() {
  const [rows, setRows] = useState<QueryRow[]>([])
  const [ratings, setRatings] = useState<Record<string, 'up' | 'down'>>({})
  const [err, setErr] = useState<string | null>(null)
  const [loadedAt, setLoadedAt] = useState<Date | null>(null)
  const [filter, setFilter] = useState({ model: '', protocol: '', mode: '' })
  const [selected, setSelected] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/telemetry/events?limit=2000`)
      if (!res.ok) throw new Error(`API returned ${res.status}`)
      const events: (QueryRow | FeedbackRow)[] = await res.json()
      const r: Record<string, 'up' | 'down'> = {}
      for (const e of events) if (e.kind === 'feedback') r[e.event_id] = e.rating // last wins
      setRatings(r)
      setRows(events.filter((e): e is QueryRow => e.kind === 'query').reverse())
      setErr(null)
      setLoadedAt(new Date())
    } catch (e) {
      setErr(String(e))
    }
  }, [])

  useEffect(() => {
    load()
    const id = setInterval(load, 15_000)
    return () => clearInterval(id)
  }, [load])

  const view = useMemo(
    () =>
      rows.filter(
        (r) =>
          (!filter.model || r.model === filter.model) &&
          (!filter.protocol || (r.protocol ?? 'none') === filter.protocol) &&
          (!filter.mode || r.mode === filter.mode),
      ),
    [rows, filter],
  )
  const sel = view.find((r) => r.event_id === selected) ?? null

  return (
    <div className="flex flex-col gap-5 pb-10">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        {(['model', 'protocol', 'mode'] as const).map((k) => (
          <label key={k} className="flex items-center gap-1.5 text-ink-dim">
            {k}
            <select
              value={filter[k]}
              onChange={(e) => setFilter({ ...filter, [k]: e.target.value })}
              className="rounded-md border border-edge bg-surface-2 px-2 py-1 text-ink"
            >
              <option value="">all</option>
              {unique(rows.map((r) => (k === 'protocol' ? (r.protocol ?? 'none') : r[k]))).map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
        ))}
        <button
          onClick={load}
          className="rounded-md border border-edge bg-surface-2 px-3 py-1 hover:border-accent"
        >
          Refresh
        </button>
        <span className="text-[0.75rem] text-ink-faint">
          {err ? <span className="text-warn">{err}</span> : loadedAt && `updated ${loadedAt.toLocaleTimeString()} · auto every 15 s`}
        </span>
      </div>

      {view.length === 0 ? (
        <p className="mt-10 text-center text-sm text-ink-dim">No queries logged yet.</p>
      ) : (
        <>
          <Kpis rows={view} ratings={ratings} />
          <div className="grid gap-4 md:grid-cols-2">
            <SeriesChart title="Time to first token" unit="ms" rows={view} value={(r) => r.timings?.ttft_ms} />
            <SeriesChart title="Mean time per output token" unit="ms/tok" rows={view} value={(r) => r.tokens?.mtpt_ms} />
          </div>
          <ToolPanel rows={view} />
          <RequestTable rows={view} ratings={ratings} selected={selected} onSelect={setSelected} />
          {sel && <Trace row={sel} />}
        </>
      )}
    </div>
  )
}

// ── KPIs ─────────────────────────────────────────────────────────────────────

function Kpis({ rows, ratings }: { rows: QueryRow[]; ratings: Record<string, string> }) {
  const ttft = nums(rows.map((r) => r.timings?.ttft_ms))
  const mtpt = nums(rows.map((r) => r.tokens?.mtpt_ms))
  const tps = nums(rows.map((r) => r.tokens?.tok_s))
  const pin = nums(rows.map((r) => r.tokens?.prompt))
  const pout = nums(rows.map((r) => r.tokens?.output))
  const up = rows.filter((r) => ratings[r.event_id] === 'up').length
  const down = rows.filter((r) => ratings[r.event_id] === 'down').length
  const pct = (n: number) => `${Math.round((100 * n) / rows.length)}%`

  const tiles: [string, string, string?][] = [
    ['queries', String(rows.length)],
    ['errors', pct(rows.filter((r) => r.error).length)],
    ['TTFT p50 / p95', `${fmt(quantile(ttft, 0.5))} / ${fmt(quantile(ttft, 0.95))}`, 'ms'],
    ['MTPT p50', fmt(quantile(mtpt, 0.5), 1), `ms/tok · ${fmt(quantile(tps, 0.5), 1)} tok/s`],
    ['tokens in / out (mean)', `${fmt(mean(pin))} / ${fmt(mean(pout))}`],
    ['used a tool', pct(rows.filter((r) => (r.tool_calls ?? 0) > 0 || (r.tool_log?.length ?? 0) > 0).length)],
    ['grounded', pct(rows.filter((r) => r.mode === 'grounded').length)],
    ['truncated', pct(rows.filter((r) => r.truncated).length)],
    ['feedback', `👍 ${up} · 👎 ${down}`],
  ]
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
      {tiles.map(([label, value, unit]) => (
        <div key={label} className="rounded-lg bg-surface-2 px-3 py-2">
          <div className="text-[0.72rem] text-ink-faint">{label}</div>
          <div className="font-mono text-lg">
            {value}
            {unit && <span className="ml-1 text-[0.7rem] text-ink-faint">{unit}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Time series (one series, one axis) ───────────────────────────────────────

function SeriesChart({
  title,
  unit,
  rows,
  value,
}: {
  title: string
  unit: string
  rows: QueryRow[]
  value: (r: QueryRow) => number | null | undefined
}) {
  const [hover, setHover] = useState<number | null>(null)
  const pts = rows
    .map((r) => ({ t: Date.parse(r.ts), v: value(r), r }))
    .filter((p): p is { t: number; v: number; r: QueryRow } => typeof p.v === 'number')
    .sort((a, b) => a.t - b.t)

  const W = 560, H = 180, L = 44, R = 10, T = 10, B = 22
  if (pts.length === 0) {
    return (
      <figure className="rounded-lg bg-surface-2 p-3">
        <figcaption className="text-sm text-ink-dim">{title}</figcaption>
        <p className="py-10 text-center text-[0.8rem] text-ink-faint">No data yet (older rows predate this field).</p>
      </figure>
    )
  }
  const t0 = pts[0].t, t1 = pts[pts.length - 1].t || t0 + 1
  const vmax = niceMax(Math.max(...pts.map((p) => p.v)))
  // Index-spaced, not time-spaced: queries arrive in bursts with days between, and a
  // true time axis would crush every session into a sliver.
  const x = (i: number) => L + (pts.length === 1 ? (W - L - R) / 2 : (i / (pts.length - 1)) * (W - L - R))
  const y = (v: number) => T + (1 - v / vmax) * (H - T - B)
  const h = hover == null ? null : pts[hover]

  return (
    <figure className="relative rounded-lg bg-surface-2 p-3">
      <figcaption className="mb-1 text-sm text-ink-dim">
        {title} <span className="text-[0.72rem] text-ink-faint">({unit}, by query)</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" onMouseLeave={() => setHover(null)}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={L} x2={W - R} y1={y(vmax * f)} y2={y(vmax * f)} stroke="var(--color-edge)" strokeWidth={1} />
            <text x={L - 6} y={y(vmax * f) + 4} textAnchor="end" fontSize={11} fill="var(--color-ink-faint)">
              {fmt(vmax * f)}
            </text>
          </g>
        ))}
        <text x={L} y={H - 6} fontSize={11} fill="var(--color-ink-faint)">{new Date(t0).toLocaleDateString()}</text>
        <text x={W - R} y={H - 6} fontSize={11} textAnchor="end" fill="var(--color-ink-faint)">
          {new Date(t1).toLocaleDateString()}
        </text>
        <polyline
          points={pts.map((p, i) => `${x(i)},${y(p.v)}`).join(' ')}
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth={2}
          strokeLinejoin="round"
        />
        {h && <line x1={x(hover!)} x2={x(hover!)} y1={T} y2={H - B} stroke="var(--color-ink-faint)" strokeWidth={1} />}
        {pts.map((p, i) => (
          <g key={p.r.event_id} onMouseEnter={() => setHover(i)}>
            <circle cx={x(i)} cy={y(p.v)} r={hover === i ? 5 : 3} fill="var(--color-accent)" stroke="var(--color-surface-2)" strokeWidth={2} />
            {/* hit target bigger than the mark */}
            <rect x={x(i) - Math.max(4, (W - L - R) / pts.length / 2)} y={T} width={Math.max(8, (W - L - R) / pts.length)} height={H - T - B} fill="transparent" />
          </g>
        ))}
      </svg>
      {h && (
        <div className="pointer-events-none absolute top-8 right-3 max-w-[60%] rounded-md border border-edge bg-surface-3 px-2 py-1 text-[0.75rem]">
          <div className="font-mono">
            {fmt(h.v, 1)} {unit}
          </div>
          <div className="text-ink-faint">{new Date(h.t).toLocaleString()}</div>
          <div className="truncate text-ink-dim">{h.r.question}</div>
        </div>
      )}
    </figure>
  )
}

// ── Tools ────────────────────────────────────────────────────────────────────

function ToolPanel({ rows }: { rows: QueryRow[] }) {
  const stats: Record<string, { calls: number; errors: number; ms: number[] }> = {}
  for (const r of rows)
    for (const t of r.tool_log ?? []) {
      const s = (stats[t.tool] ??= { calls: 0, errors: 0, ms: [] })
      s.calls++
      if (!t.ok) s.errors++
      s.ms.push(t.ms)
    }
  const names = Object.keys(stats).sort()
  if (names.length === 0) return null
  return (
    <section className="rounded-lg bg-surface-2 p-3">
      <h2 className="mb-2 text-sm text-ink-dim">Tool calls</h2>
      <table className="w-full text-left font-mono text-[0.8rem]">
        <thead className="text-ink-faint">
          <tr>
            <th className="font-normal">tool</th>
            <th className="font-normal">calls</th>
            <th className="font-normal">error rate</th>
            <th className="font-normal">p50 ms</th>
            <th className="font-normal">p95 ms</th>
          </tr>
        </thead>
        <tbody>
          {names.map((n) => (
            <tr key={n}>
              <td>{n}</td>
              <td>{stats[n].calls}</td>
              <td className={stats[n].errors ? 'text-warn' : ''}>{Math.round((100 * stats[n].errors) / stats[n].calls)}%</td>
              <td>{fmt(quantile(stats[n].ms, 0.5))}</td>
              <td>{fmt(quantile(stats[n].ms, 0.95))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

// ── Requests ─────────────────────────────────────────────────────────────────

function RequestTable({
  rows,
  ratings,
  selected,
  onSelect,
}: {
  rows: QueryRow[]
  ratings: Record<string, string>
  selected: string | null
  onSelect: (id: string | null) => void
}) {
  return (
    <section className="overflow-x-auto rounded-lg bg-surface-2 p-3">
      <h2 className="mb-2 text-sm text-ink-dim">Requests (newest first; click for the trace)</h2>
      <table className="w-full min-w-[56rem] text-left text-[0.8rem]">
        <thead className="text-ink-faint">
          <tr>
            {['time', 'question', 'model · protocol', 'mode', 'ttft', 'total', 'tok in → out', 'mtpt', 'tools', '👍'].map((h) => (
              <th key={h} className="pr-3 pb-1 font-normal">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono">
          {rows.slice(0, 200).map((r) => (
            <tr
              key={r.event_id}
              onClick={() => onSelect(selected === r.event_id ? null : r.event_id)}
              className={`cursor-pointer border-t border-edge hover:bg-surface-3 ${selected === r.event_id ? 'bg-surface-3' : ''}`}
            >
              <td className="pr-3 whitespace-nowrap text-ink-faint">{new Date(r.ts).toLocaleString()}</td>
              <td className="max-w-[18rem] truncate pr-3 font-sans" title={r.question}>
                {r.error && <span className="mr-1 text-warn">⚠ error</span>}
                {r.question}
              </td>
              <td className="pr-3 whitespace-nowrap">{shortModel(r.model)} · {r.protocol ?? 'none'}</td>
              <td className="pr-3">{r.mode}</td>
              <td className="pr-3">{fmt(r.timings?.ttft_ms)}</td>
              <td className="pr-3">{fmt(totalMs(r))}</td>
              <td className="pr-3 whitespace-nowrap">{fmt(r.tokens?.prompt)} → {fmt(r.tokens?.output)}</td>
              <td className="pr-3">{fmt(r.tokens?.mtpt_ms, 1)}</td>
              <td className="pr-3">{toolSummary(r)}</td>
              <td>{ratings[r.event_id] === 'up' ? '👍' : ratings[r.event_id] === 'down' ? '👎' : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

// ── Trace ────────────────────────────────────────────────────────────────────

type Span = { label: string; kind: keyof typeof KIND_COLOR; start: number; dur: number; detail?: string }

function Trace({ row }: { row: QueryRow }) {
  const spans: Span[] = []
  // Retrieval stages carry durations but no offsets (and accumulate across mid-answer
  // searches), so they are laid end to end from 0 — an approximation, labelled as one.
  let cursor = 0
  for (const k of RETRIEVAL_STAGES) {
    const d = row.timings?.[k]
    if (d) {
      spans.push({ label: k.replace('_ms', ''), kind: 'retrieval', start: cursor, dur: d })
      cursor += d
    }
  }
  ;(row.passes ?? []).forEach((p, i) =>
    spans.push({
      label: `gen #${i + 1}`,
      kind: 'generation',
      start: p.at_ms,
      dur: p.ms,
      detail: `${fmt(p.prompt_tokens)} prompt tok (prefill ${fmt(p.prefill_ms)} ms) → ${fmt(p.output_tokens)} out (decode ${fmt(p.decode_ms)} ms)${p.load_ms && p.load_ms > 50 ? ` · model load ${fmt(p.load_ms)} ms` : ''} · ${p.done_reason ?? '?'}${p.tool_calls ? ` · ${p.tool_calls} call(s)` : ''}`,
    }),
  )
  for (const t of row.tool_log ?? [])
    if (t.at_ms != null)
      spans.push({
        label: `${t.tool}${t.ok ? '' : ' ✗'}`,
        kind: t.tool === 'run_python' ? 'python' : 'search',
        start: t.at_ms,
        dur: t.ms,
      })
  const end = Math.max(totalMs(row) ?? 0, ...spans.map((s) => s.start + s.dur), 1)
  const ttft = row.timings?.ttft_ms

  return (
    <section className="flex flex-col gap-4 rounded-lg border border-edge bg-surface-2 p-4">
      <div>
        <h2 className="text-sm text-ink-dim">Trace · {new Date(row.ts).toLocaleString()} · <span className="font-mono text-[0.75rem]">{row.event_id}</span></h2>
        <p className="mt-1">{row.question}</p>
        {row.retrieval_query && (
          <p className="mt-1 text-[0.8rem] text-ink-dim">searched as: <span className="font-mono">{row.retrieval_query}</span></p>
        )}
        {row.error && <p className="mt-1 font-mono text-[0.8rem] text-warn">{row.error}</p>}
      </div>

      <div className="flex flex-wrap gap-3 text-[0.75rem] text-ink-dim">
        {Object.entries(KIND_COLOR).map(([k, c]) => (
          <span key={k} className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: c }} /> {k}
          </span>
        ))}
        {ttft != null && <span>┊ first token at {fmt(ttft)} ms</span>}
      </div>

      {spans.length === 0 ? (
        <p className="text-[0.8rem] text-ink-faint">No span data on this row (logged before tracing was added).</p>
      ) : (
        <div className="relative flex flex-col gap-1">
          {ttft != null && (
            <div
              className="absolute top-0 bottom-0 border-l border-dashed border-ink-faint"
              style={{ left: `calc(9rem + (100% - 9rem) * ${ttft / end})` }}
              aria-hidden
            />
          )}
          {spans.map((s, i) => (
            <div key={i} className="grid grid-cols-[9rem_1fr] items-center gap-0 text-[0.75rem]">
              <span className="truncate pr-2 font-mono text-ink-dim" title={s.detail}>{s.label}</span>
              <div className="relative h-5">
                <div
                  className="absolute h-full rounded"
                  style={{ left: `${(100 * s.start) / end}%`, width: `max(2px, ${(100 * s.dur) / end}%)`, background: KIND_COLOR[s.kind] }}
                  title={`${s.label}: ${fmt(s.dur)} ms at +${fmt(s.start)} ms${s.detail ? `\n${s.detail}` : ''}`}
                />
                <span
                  className="absolute top-0.5 font-mono whitespace-nowrap text-ink"
                  style={s.start + s.dur > end * 0.75 ? { right: `${100 - (100 * s.start) / end}%`, paddingRight: 4 } : { left: `calc(${(100 * (s.start + s.dur)) / end}% + 4px)` }}
                >
                  {fmt(s.dur)} ms
                </span>
              </div>
            </div>
          ))}
          <div className="grid grid-cols-[9rem_1fr] text-[0.7rem] text-ink-faint">
            <span />
            <div className="flex justify-between font-mono"><span>0</span><span>{fmt(end)} ms</span></div>
          </div>
          {spans.some((s) => s.kind === 'retrieval') && (
            <p className="text-[0.7rem] text-ink-faint">Retrieval stages are shown end to end from 0: the log has their durations, not their start times.</p>
          )}
        </div>
      )}

      {(row.passes?.length ?? 0) > 0 && (
        <table className="w-full text-left font-mono text-[0.75rem]">
          <thead className="text-ink-faint">
            <tr>{['pass', 'start', 'wall', 'prompt tok', 'prefill', 'out tok', 'decode', 'ms/tok', 'end'].map((h) => <th key={h} className="pr-3 font-normal">{h}</th>)}</tr>
          </thead>
          <tbody>
            {row.passes!.map((p, i) => (
              <tr key={i}>
                <td className="pr-3">#{i + 1}</td>
                <td className="pr-3">+{fmt(p.at_ms)}</td>
                <td className="pr-3">{fmt(p.ms)}</td>
                <td className="pr-3">{fmt(p.prompt_tokens)}</td>
                <td className="pr-3">{fmt(p.prefill_ms)}</td>
                <td className="pr-3">{fmt(p.output_tokens)}</td>
                <td className="pr-3">{fmt(p.decode_ms)}</td>
                <td className="pr-3">{p.decode_ms && p.output_tokens ? fmt(p.decode_ms / p.output_tokens, 1) : '—'}</td>
                <td className="pr-3">{p.done_reason ?? '—'}{p.tool_calls ? ` +${p.tool_calls} call` : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {(row.tool_log?.length ?? 0) > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-[0.8rem] text-ink-dim">Tool calls</h3>
          {row.tool_log!.map((t) => (
            <details key={t.n} className="rounded-md bg-surface-3 p-2 text-[0.75rem]">
              <summary className="cursor-pointer font-mono">
                #{t.n} {t.tool} · {fmt(t.ms)} ms · {t.ok ? 'ok' : <span className="text-warn">failed{t.reason ? ` (${t.reason})` : ''}</span>}
                {t.query && <span className="text-ink-dim"> · “{t.query}”</span>}
              </summary>
              {t.code && <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-ink-dim">{t.code}</pre>}
              {t.output && <pre className="mt-2 overflow-x-auto border-t border-edge pt-2 whitespace-pre-wrap">{t.output}</pre>}
            </details>
          ))}
        </div>
      )}

      {(row.search_queries?.length ?? 0) > 0 && (
        <div className="text-[0.8rem]">
          <h3 className="text-ink-dim">Model-written searches</h3>
          <ul className="list-disc pl-5 font-mono">{row.search_queries!.map((q, i) => <li key={i}>{q}</li>)}</ul>
        </div>
      )}

      <SourceList title="Retrieved (all, unfiltered)" sources={row.sources} />
      <SourceList title="Found mid-answer" sources={row.late_sources} />
    </section>
  )
}

function SourceList({ title, sources }: { title: string; sources?: Source[] }) {
  if (!sources?.length) return null
  return (
    <div className="text-[0.75rem]">
      <h3 className="mb-1 text-[0.8rem] text-ink-dim">{title} <span className="text-ink-faint">(floor {FLOOR})</span></h3>
      <table className="w-full text-left">
        <tbody>
          {sources.map((s, i) => (
            <tr key={i} className={`border-t border-edge ${(s.score ?? 0) < FLOOR ? 'text-ink-faint' : ''}`}>
              <td className="w-16 pr-3 font-mono">{s.score != null ? s.score.toFixed(3) : '—'}</td>
              <td className="pr-3 font-mono whitespace-nowrap">{s.chunk_id ?? s.source}</td>
              <td className="max-w-[30rem] truncate" title={s.preview}>{s.preview}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── helpers ──────────────────────────────────────────────────────────────────

function totalMs(r: QueryRow): number | null {
  const last = r.passes?.at(-1)
  if (last) return last.at_ms + last.ms
  const t = r.timings ?? {}
  const parts = [...RETRIEVAL_STAGES, 'generate_ms'].map((k) => t[k]).filter((v): v is number => typeof v === 'number')
  return parts.length ? parts.reduce((a, b) => a + b, 0) : null
}

function toolSummary(r: QueryRow): string {
  const c = r.tool_counts ?? {}
  const parts = Object.entries(c).map(([k, n]) => `${k.replace('_documents', '').replace('run_', '')}×${n}`)
  if (!parts.length && r.tool_calls) parts.push(`python×${r.tool_calls}`)
  return parts.join(' ') + (r.tool_errors ? ` (${r.tool_errors}✗)` : '')
}

const shortModel = (m: string) => m.replace(/^hf\.co\/[^/]+\//, '').replace(/-GGUF/, '')
const unique = (xs: string[]) => [...new Set(xs)].sort()
const nums = (xs: (number | null | undefined)[]) => xs.filter((x): x is number => typeof x === 'number')
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)

function quantile(xs: number[], q: number): number | null {
  if (!xs.length) return null
  const s = [...xs].sort((a, b) => a - b)
  const i = (s.length - 1) * q
  const lo = Math.floor(i)
  return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo)
}

function fmt(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

function niceMax(v: number): number {
  if (v <= 0) return 1
  const p = 10 ** Math.floor(Math.log10(v))
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v)!
}
