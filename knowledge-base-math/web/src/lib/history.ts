import type { Message, Turn } from './types'

/**
 * Build the `history` the next request carries.
 *
 * Two rules, both copied from the Gradio client's clean-history handling (app.py:342):
 *
 *  - assistant content is the turn's `clean` text, never what was displayed, so the
 *    server's own footer never comes back as something the model said;
 *  - a turn whose assistant half produced no model text at all — an error, or an empty
 *    generation — is dropped **with its user question**, rather than recorded as an
 *    empty assistant turn the model then has to make sense of.
 *
 * The current question is not part of history; the caller sends it as `message`.
 */
export function buildHistory(turns: Turn[]): Message[] {
  const out: Message[] = []
  let pendingUser: Message | null = null

  for (const turn of turns) {
    if (turn.role === 'user') {
      pendingUser = { role: 'user', content: turn.clean }
      continue
    }
    if (turn.failed || !turn.clean.trim()) {
      pendingUser = null
      continue
    }
    if (pendingUser) out.push(pendingUser)
    out.push({ role: 'assistant', content: turn.clean })
    pendingUser = null
  }
  return out
}
