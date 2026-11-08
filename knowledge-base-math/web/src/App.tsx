import { useEffect, useState } from 'react'

import { Chat } from './components/Chat'
import { Dashboard } from './components/Dashboard'
import { health, setToken } from './lib/sse'
import type { Health } from './lib/types'

/**
 * The username is the index selector, not an identity — isolation here is per-user
 * directory naming, and this box is the whole of it. Kept in localStorage so a family
 * member types it once per device.
 */
const USER_KEY = 'kbm.user'

// Mirrors api/deps.normalize_user (Chroma's collection-name rule); the API enforces it.
const USER_RE = /^[a-z0-9_-]*[a-z0-9]$/
const USER_RULE =
  'Names can only use letters, numbers, - and _ (no spaces or accents), and must end with a letter or number.'

const storedUser = () => {
  try {
    const u = localStorage.getItem(USER_KEY) ?? ''
    // A name saved before the rule existed would only earn a 400 on every request.
    return USER_RE.test(u) ? u : ''
  } catch {
    return ''
  }
}

export default function App() {
  const [user, setUser] = useState(storedUser)
  const [draft, setDraft] = useState(user)
  const [nameError, setNameError] = useState('')
  const [status, setStatus] = useState<Health | null | 'auth' | 'loading'>('loading')
  const [password, setPassword] = useState('')
  // Hash, not a router: one extra page doesn't earn a dependency.
  const [hash, setHash] = useState(location.hash)
  const dashboard = hash === '#/dashboard'

  useEffect(() => {
    health().then(setStatus)
    const onHash = () => setHash(location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  function submitPassword() {
    setToken(password.trim())
    setPassword('')
    setStatus('loading')
    health().then(setStatus)
  }

  function commitUser(next: string) {
    const clean = next.trim().toLowerCase()
    if (clean && !USER_RE.test(clean)) {
      setNameError(USER_RULE)
      return
    }
    setNameError('')
    setUser(clean)
    setDraft(clean)
    try {
      localStorage.setItem(USER_KEY, clean)
    } catch {
      /* private window: remembered for this page load only */
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-4 p-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Math Tutor</h1>
          <p className="text-[0.8rem] text-ink-faint">
            Your personal math knowledge base.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <a
            href={dashboard ? '#/' : '#/dashboard'}
            className="text-[0.78rem] text-ink-dim underline-offset-2 hover:text-accent hover:underline"
          >
            {dashboard ? '← chat' : 'monitoring'}
          </a>
          <ServerStatus status={status} />
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => commitUser(draft)}
            onKeyDown={(e) => e.key === 'Enter' && commitUser(draft)}
            placeholder="your name"
            aria-label="Your name"
            aria-invalid={!!nameError}
            aria-describedby={nameError ? 'name-error' : undefined}
            className="w-36 rounded-lg border border-edge bg-surface-2 px-3 py-1.5 text-sm
              outline-none placeholder:text-ink-faint focus:border-accent"
          />
        </div>
      </header>

      {nameError && (
        <p id="name-error" role="alert" className="text-right text-[0.8rem] text-warn">
          {nameError}
        </p>
      )}

      {status === 'auth' ? (
        // The server has KBM_API_TOKEN set and this device has not sent it (or sent an
        // old one). Nothing else on the page can work until it does.
        <form
          className="mx-auto mt-16 flex w-72 flex-col gap-2 text-sm"
          onSubmit={(e) => {
            e.preventDefault()
            submitPassword()
          }}
        >
          <label htmlFor="pw" className="text-ink-dim">
            Family password
          </label>
          <input
            id="pw"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="rounded-lg border border-edge bg-surface-2 px-3 py-1.5 outline-none focus:border-accent"
          />
          <button type="submit" disabled={!password.trim()} className="rounded-lg bg-accent px-3 py-1.5 font-semibold text-surface disabled:opacity-40">
            Continue
          </button>
        </form>
      ) : dashboard ? (
        <Dashboard />
      ) : user ? (
        // Keyed on the user: switching name switches to a different index, and carrying
        // the previous conversation across that boundary would be misleading.
        <Chat key={user} user={user} />
      ) : (
        <p className="mt-16 text-center text-sm text-ink-dim">
          Enter your name above to load your documents.
        </p>
      )}
    </div>
  )
}

function ServerStatus({ status }: { status: Health | null | 'auth' | 'loading' }) {
  if (status === 'loading') return <span className="text-[0.78rem] text-ink-faint">checking…</span>
  if (status === 'auth') return <span className="text-[0.78rem] text-warn">password needed</span>
  if (status === null) {
    return (
      <span className="text-[0.78rem] text-warn" title="Nothing answered /healthz.">
        server not responding
      </span>
    )
  }
  return (
    <span
      className="font-mono text-[0.72rem] text-ink-faint"
      title={`protocol: ${status.protocol} · retrieval ${status.retrieval_ready ? 'ready' : 'not ready'}`}
    >
      {status.model}
      {!status.model_loaded && <span className="ml-1 text-warn">(loading)</span>}
    </span>
  )
}
