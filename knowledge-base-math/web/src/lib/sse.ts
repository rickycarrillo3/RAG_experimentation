/**
 * The POST /chat client: Server-Sent Events over `fetch`.
 *
 * `EventSource` cannot be used here, and not by preference — it is GET-only and cannot
 * set request headers, while /chat is a POST and the API's auth is an `Authorization`
 * bearer header. So the stream is read off `Response.body` and framed by hand.
 *
 * The wire format is `api/routes.py:_sse`:
 *
 *     event: <name>\ndata: <single-line JSON>\n\n
 *
 * No `id:`, no `retry:`, and **no heartbeat** — a silent gap before the first token is
 * normal (generation is ~95% of query time), so nothing here may treat quiet as dead.
 * The `event:` line is parsed for completeness but not trusted: every payload carries
 * its own `type` discriminator, and that is what callers switch on.
 */
import type { ChatEvent, ChatRequest } from './types'

/**
 * Vite proxies this to the API, stripping the prefix (see vite.config.ts); in
 * production api/main.py serves this app and mounts the router under /api itself.
 */
export const API_BASE = '/api'

export class ApiError extends Error {}

/** Thrown on a 401, so the app can ask for the family password instead of erroring. */
export class AuthError extends ApiError {}

/**
 * The family password IS the API's bearer token (KBM_API_TOKEN). Kept in localStorage
 * so it is typed once per device. It is one shared secret, not per-person auth — the
 * username box still selects an index, not an identity.
 *
 * In dev the Vite proxy sets the header from its own env and overwrites this one, so
 * an empty token here is normal there.
 */
const TOKEN_KEY = 'kbm.token'

export const getToken = () => {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

export const setToken = (token: string) => {
  try {
    localStorage.setItem(TOKEN_KEY, token)
  } catch {
    /* private window: the token lasts this page load only */
  }
}

/** fetch() against the API, with the bearer header and a typed 401. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('authorization', `Bearer ${token}`)
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers })
  if (res.status === 401) throw new AuthError('The family password is missing or wrong.')
  return res
}

/**
 * Stream one answer. Yields `sources`, then `token`*, then `done`.
 *
 * An `error` frame is terminal: the server sends no `done` after one, so it is yielded
 * and the generator returns. Callers get exactly one of `done` or `error`, or an
 * `ApiError` thrown for a transport-level failure.
 */
export async function* streamChat(
  req: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await apiFetch('/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!res.ok) {
    // Surface the status and whatever the server said: a 422 and a 500 need different
    // fixes and "request failed" distinguishes neither. (401 is AuthError, above.)
    const detail = await res.text().catch(() => '')
    throw new ApiError(`API returned ${res.status}: ${detail.slice(0, 500)}`)
  }
  if (!res.body) throw new ApiError('API returned no body to stream.')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break

      // `stream: true` matters: a multi-byte character can straddle two chunks, and
      // decoding each chunk independently would corrupt it. Math answers are full of
      // them (−, ×, √, θ).
      buffer += decoder.decode(value, { stream: true })

      // Frames are separated by a blank line. The trailing remainder is an incomplete
      // frame and stays in the buffer until the rest of it arrives.
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''

      for (const frame of frames) {
        const event = parseFrame(frame)
        if (!event) continue
        yield event
        if (event.type === 'error' || event.type === 'done') return
      }
    }
  } finally {
    // Releasing the lock cancels an abandoned stream, so a component that unmounts
    // mid-answer does not leave the generation running with nobody reading it.
    reader.releaseLock()
    if (!signal?.aborted) await res.body.cancel().catch(() => {})
  }
}

function parseFrame(frame: string): ChatEvent | null {
  for (const rawLine of frame.split('\n')) {
    const line = rawLine.replace(/\r$/, '')
    if (!line.startsWith('data:')) continue
    try {
      return JSON.parse(line.slice('data:'.length).trim()) as ChatEvent
    } catch {
      // A frame we cannot parse is not worth killing an answer over; the `done` frame
      // is what matters and it will arrive on its own line.
      return null
    }
  }
  return null
}

/**
 * GET /healthz. Returns null when the API cannot be reached at all, and 'auth' when it
 * answered 401 — the two need different responses (wait vs. ask for the password).
 */
export async function health(): Promise<import('./types').Health | null | 'auth'> {
  try {
    const res = await apiFetch('/healthz')
    if (!res.ok) throw new ApiError(String(res.status))
    return (await res.json()) as import('./types').Health
  } catch (e) {
    return e instanceof AuthError ? 'auth' : null
  }
}
