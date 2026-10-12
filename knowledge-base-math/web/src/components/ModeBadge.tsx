import type { Mode } from '../lib/types'

/**
 * Provenance, as a label.
 *
 * `grounded` means retrieval produced a chunk scoring at or above the calibrated
 * relevance floor and the answer was written against the family's own documents;
 * `general` means it came from the model's own knowledge. The server decides this — it
 * is not a model claim — and without it a grounded answer and a confident hallucination
 * look identical.
 */
export function ModeBadge({ mode }: { mode: Mode }) {
  const grounded = mode === 'grounded'
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[0.72rem] font-medium tracking-wide uppercase ${
        grounded ? 'bg-accent-dim text-ink' : 'bg-surface-3 text-ink-dim'
      }`}
      title={
        grounded
          ? 'Answered from your documents, which are named at the end of the answer.'
          : "Answered from the model's own knowledge — nothing in your documents scored above the relevance floor."
      }
    >
      {mode}
    </span>
  )
}
