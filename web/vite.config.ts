import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // three.js is split into the lazily loaded Model chunk (~585 kB); the rest stays small.
  build: { chunkSizeWarningLimit: 650 },
  server: {
    host: true,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
});
