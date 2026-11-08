import { useEffect, useState } from 'react'

import { Chat } from './components/Chat'
import { health } from './lib/sse'
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

export default function App() {
  const [user, setUser] = useState(() => localStorage.getItem(USER_KEY) ?? '')
  const [draft, setDraft] = useState(user)
  const [nameError, setNameError] = useState('')
  const [status, setStatus] = useState<Health | null | 'loading'>('loading')

  useEffect(() => {
    health().then(setStatus)
  }, [])

  function commitUser(next: string) {
    const clean = next.trim().toLowerCase()
    if (clean && !USER_RE.test(clean)) {
      setNameError(USER_RULE)
      return
    }
    setNameError('')
    setUser(clean)
    setDraft(clean)
    localStorage.setItem(USER_KEY, clean)
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

      {user ? (
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

function ServerStatus({ status }: { status: Health | null | 'loading' }) {
  if (status === 'loading') return <span className="text-[0.78rem] text-ink-faint">checking…</span>
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
