import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The API host. Read from the real environment, NOT from a VITE_-prefixed variable:
// anything Vite would inline is public, and the whole point of proxying is that the
// browser never holds a credential (see the proxyReq hook below).
const API_URL = process.env.KBM_API_URL ?? 'http://127.0.0.1:8000'
const API_TOKEN = process.env.KBM_API_TOKEN ?? ''

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // The browser talks to one origin, so CORS never applies in development and the
      // bearer token stays in this Node process. `/api` is stripped on the way out —
      // the API serves `/chat`, not `/api/chat`.
      '/api': {
        target: API_URL,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // SSE: no buffering, no response transforms, and no idle timeout. /chat sends
        // no heartbeat (api/routes.py), so a silent minute before the first token is
        // normal and must not be read as a dead connection.
        timeout: 0,
        proxyTimeout: 0,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            if (API_TOKEN) proxyReq.setHeader('authorization', `Bearer ${API_TOKEN}`)
          })
        },
      },
    },
  },
})
