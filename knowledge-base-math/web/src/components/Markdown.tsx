import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'
import ReactMarkdown from 'react-markdown'

/**
 * Answer text, rendered.
 *
 * KaTeX is not decoration here: the extraction pipeline produces Markdown+LaTeX and the
 * generator answers in it, so a tutor that showed `\frac{-b \pm \sqrt{b^2-4ac}}{2a}` as
 * source would be a regression on the Gradio UI, which had `latex_delimiters` set.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div
      className="[&_code]:font-mono [&_code]:text-[0.92em] [&_h1]:mt-3 [&_h1]:text-base
        [&_h1]:font-semibold [&_h2]:mt-3 [&_h2]:text-base [&_h2]:font-semibold [&_h3]:mt-3
        [&_h3]:font-semibold [&_li]:my-0.5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5
        [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_pre]:my-2
        [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-surface [&_pre]:p-3
        [&_strong]:font-semibold [&_table]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5"
    >
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {children}
      </ReactMarkdown>
    </div>
  )
}
