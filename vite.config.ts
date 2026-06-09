import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Cross-origin isolation so ONNX Runtime Web can use threaded (SharedArrayBuffer)
// wasm. COEP credentialless lets the cross-origin model fetch from the HF CDN
// through while staying isolated.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: {
    host: '0.0.0.0',
    headers: isolation,
    allowedHosts: ['akashmac', '.tail09ed28.ts.net'],
  },
  preview: {
    host: '0.0.0.0',
    headers: isolation,
    allowedHosts: ['akashmac', '.tail09ed28.ts.net'],
  },
})
