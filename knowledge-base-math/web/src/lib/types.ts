/**
 * Narrow aliases over the generated schema, plus the client-side turn model.
 *
 * `api.d.ts` is generated from the API's own `/openapi.json` (`npm run gen:api`), which
 * FastAPI generates from `api/schemas.py` — so a renamed wire field is a compile error
 * here rather than a blank bubble at runtime. Nothing in this app should type a wire
 * field by hand; add an alias instead.
 */
import type { components } from './api'

type S = components['schemas']

export type Mode = S['Mode']
export type Role = S['Role']
export type Message = S['Message']
export type Source = S['Source']
export type Timings = S['Timings']

export type SourcesEvent = S['SourcesEvent']
export type TokenEvent = S['TokenEvent']
export type DoneEvent = S['DoneEvent']
export type ErrorEvent = S['ErrorEvent']

/** Everything POST /chat can put on the wire, discriminated on `type`. */
export type ChatEvent = SourcesEvent | TokenEvent | DoneEvent | ErrorEvent

export type ChatRequest = S['ChatRequest']
export type Health = S['Health']

/**
 * A run of answer text with one authorship.
 *
 * `server` mirrors `TokenEvent.server_marker`: true for text the server wrote rather
 * than the model — the `Sources:` footer, the truncation notice, and in agent mode the
 * search and tool-output markers. Kept as separate runs rather than one string so the
 * server's voice can be styled as the server's, and because in agent mode those markers
 * arrive mid-answer rather than only at the end.
 */
export interface Segment {
  text: string
  server: boolean
}

/**
 * One turn, held twice.
 *
 * `segments` is what the student sees. `clean` is what goes back in the next request's
 * `history`, and excludes every server-written run: the model must never be handed a
 * transcript in which it appears to have written the server's words. A turn whose
 * `clean` comes out empty is dropped from history entirely — see buildHistory.
 */
export interface Turn {
  role: Role
  segments: Segment[]
  clean: string
  /** Set on the assistant turn once its `done` frame arrives. */
  done?: DoneEvent
  /** From the `sources` frame, so they can be shown while the answer is still decoding. */
  sources?: Source[]
  mode?: Mode
  /** A client-side failure bubble. Never sent back as history. */
  failed?: boolean
}

/** Append text to the last run of matching authorship, or start a new run. */
export function appendSegment(segments: Segment[], text: string, server: boolean): Segment[] {
  const last = segments[segments.length - 1]
  if (last && last.server === server) {
    return [...segments.slice(0, -1), { ...last, text: last.text + text }]
  }
  return [...segments, { text, server }]
}

export const segmentText = (segments: Segment[]) => segments.map((s) => s.text).join('')
