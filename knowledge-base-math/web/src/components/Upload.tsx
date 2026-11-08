import { useEffect, useRef, useState } from 'react'

import { apiFetch } from '../lib/sse'
import type { Job, UserStatus } from '../lib/types'

/** Marker on a large textbook is minutes; past this we stop watching (the job is NOT cancelled). */
const POLL_TIMEOUT_MS = 45 * 60 * 1000
const POLL_MS = 3000

type State =
  | { kind: 'idle' }
  | { kind: 'working'; name: string; text: string }
  | { kind: 'done' | 'failed' | 'lost'; name: string; text: string; diagnostic?: string | null }

/**
 * Choosing a file IS the request to ingest it — no separate button, same as the Gradio
 * box this replaces. POST /upload returns a job handle and GET /jobs/{id} is polled,
 * because Marker runs for minutes.
 *
 * `detail` is the sentence for the family; `diagnostic` is the technical cause and only
 * arrives when the server sets KBM_SHOW_DIAGNOSTICS (api/routes.py:get_job).
 */
export function Upload({ user }: { user: string }) {
  const [state, setState] = useState<State>({ kind: 'idle' })
  const [docs, setDocs] = useState<UserStatus | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const alive = useRef(true)

  const refreshDocs = () =>
    apiFetch(`/users/${encodeURIComponent(user)}/status`)
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => alive.current && setDocs(s))
      .catch(() => {})

  useEffect(() => {
    alive.current = true
    refreshDocs()
    return () => {
      alive.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user])

  async function ingest(file: File) {
    const name = file.name
    // Clear the input now so choosing the same file again re-fires `change` — that is
    // the retry path after a failure, as it was in Gradio.
    if (input.current) input.current.value = ''

    setState({ kind: 'working', name, text: `Uploading ${name}…` })
    const form = new FormData()
    form.append('user', user)
    form.append('file', file)

    let jobId: string
    try {
      const res = await apiFetch('/upload', { method: 'POST', body: form })
      if (!res.ok) {
        // A FastAPI error body: show its message if there is one, never the envelope.
        const why = await res
          .json()
          .then((b) => (typeof b?.detail === 'string' ? b.detail : null))
          .catch(() => null)
        setState({ kind: 'failed', name, text: `Could not upload ${name} — ${why ?? 'the server rejected it'}.` })
        return
      }
      jobId = ((await res.json()) as Job).job_id
    } catch (e) {
      setState({ kind: 'failed', name, text: e instanceof Error ? e.message : 'Could not reach the server.' })
      return
    }

    // After this point the job exists on the server: a transport failure means "lost
    // contact", not "failed" — the import may well still be running.
    const started = Date.now()
    while (alive.current) {
      await new Promise((r) => setTimeout(r, POLL_MS))
      const mins = ((Date.now() - started) / 60000).toFixed(1)
      let job: Job
      try {
        const res = await apiFetch(`/jobs/${jobId}`)
        if (!res.ok) throw new Error(String(res.status))
        job = (await res.json()) as Job
      } catch {
        setState({
          kind: 'lost',
          name,
          text: 'Lost contact with the server while importing. It may still be running — check the document list in a minute.',
        })
        return
      }
      if (job.status === 'done' || job.status === 'failed') {
        setState({ kind: job.status, name, text: job.detail, diagnostic: job.diagnostic })
        refreshDocs()
        return
      }
      if (Date.now() - started > POLL_TIMEOUT_MS) {
        setState({ kind: 'lost', name, text: `Still working after ${mins} minutes. It has not been cancelled and should finish on its own.` })
        return
      }
      setState({ kind: 'working', name, text: `${job.status === 'queued' ? 'Queued' : 'Importing'}… (${mins} min)` })
    }
  }

  const busy = state.kind === 'working'

  return (
    <section className="flex flex-col gap-2 rounded-xl border border-edge p-3 text-sm">
      <h2 className="text-[0.8rem] font-semibold tracking-wide text-ink-dim uppercase">Your documents</h2>

      {docs?.sources.length ? (
        <ul className="flex flex-col gap-0.5 text-[0.85rem]">
          {docs.sources.map((s) => (
            <li key={s} className="truncate" title={s}>
              {s}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[0.85rem] text-ink-faint">None yet. Add a PDF textbook or worksheet.</p>
      )}

      <label
        className={`mt-1 cursor-pointer rounded-lg border border-dashed border-edge px-3 py-2 text-center text-ink-dim
          hover:border-accent hover:text-ink ${busy ? 'pointer-events-none opacity-50' : ''}`}
      >
        {busy ? 'Importing…' : 'Add a PDF'}
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          className="sr-only"
          disabled={busy}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) ingest(f)
          }}
        />
      </label>

      {state.kind !== 'idle' && (
        <div role="status" className={state.kind === 'failed' ? 'text-warn' : 'text-ink-dim'}>
          <p>
            {state.kind === 'done' ? '✅ ' : state.kind === 'failed' ? '❌ ' : ''}
            {state.text}
            {state.kind === 'failed' && ' Please try uploading the file again.'}
          </p>
          {'diagnostic' in state && state.diagnostic && (
            <p className="mt-1 font-mono text-[0.72rem] break-words text-ink-faint">{state.diagnostic}</p>
          )}
        </div>
      )}
    </section>
  )
}
