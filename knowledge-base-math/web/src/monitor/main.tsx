import { StrictMode, useCallback, useState } from 'react'
import { createRoot } from 'react-dom/client'

import '../index.css'
import { Dashboard } from './Dashboard'

/**
 * The monitoring page: an operator tool, separate from the family app. It shares only
 * the stylesheet. Its credential is the admin token (KBM_ADMIN_TOKEN), never the family
 * password, and it is kept under its own localStorage key.
 */
const KEY = 'kbm.admin_token'

const read = () => {
  try {
    return localStorage.getItem(KEY) ?? ''
  } catch {
    return ''
  }
}

function Monitor() {
  const [token, setToken] = useState(read)
  const [draft, setDraft] = useState('')
  // Start on the dashboard: on a laptop with no tokens set, the API is open and no
  // prompt is needed. A 401/403 brings the prompt up.
  const [prompt, setPrompt] = useState<string | null>(null)

  const onAuthFail = useCallback((why: string) => setPrompt(why), [])

  function save() {
    const t = draft.trim()
    try {
      localStorage.setItem(KEY, t)
    } catch {
      /* private window: this page load only */
    }
    setToken(t)
    setDraft('')
    setPrompt(null)
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4 p-4">
      <header>
        <h1 className="text-lg font-semibold">Monitoring</h1>
        <p className="text-[0.8rem] text-ink-faint">Traces, tokens, latency and tool calls from the telemetry log.</p>
      </header>
      {prompt != null ? (
        <form
          className="mx-auto mt-16 flex w-80 flex-col gap-2 text-sm"
          onSubmit={(e) => {
            e.preventDefault()
            save()
          }}
        >
          <p className="text-warn">{prompt}</p>
          <label htmlFor="tok" className="text-ink-dim">
            Admin token (KBM_ADMIN_TOKEN)
          </label>
          <input
            id="tok"
            type="password"
            autoComplete="current-password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            className="rounded-lg border border-edge bg-surface-2 px-3 py-1.5 outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={!draft.trim()}
            className="rounded-lg bg-accent px-3 py-1.5 font-semibold text-surface disabled:opacity-40"
          >
            Continue
          </button>
        </form>
      ) : (
        <Dashboard key={token} token={token} onAuthFail={onAuthFail} />
      )}
    </div>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Monitor />
  </StrictMode>,
)
