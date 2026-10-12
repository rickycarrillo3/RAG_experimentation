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

/** Vite proxies this to the API, stripping the prefix (see vite.config.ts). */
export const API_BASE = '/api'

export class ApiError extends Error {}

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
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!res.ok) {
    // Mirrors app.py's handling: surface the status and whatever the server said, since
    // a 401 and a 422 need different fixes and "request failed" distinguishes neither.
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

/** GET /healthz. Returns null when the API cannot be reached at all. */
export async function health(): Promise<import('./types').Health | null> {
  try {
    const res = await fetch(`${API_BASE}/healthz`)
    if (!res.ok) throw new ApiError(String(res.status))
    return (await res.json()) as import('./types').Health
  } catch {
    return null
  }
}
