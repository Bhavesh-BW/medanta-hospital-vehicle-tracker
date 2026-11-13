import { defineConfig } from 'vite';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Service key: env VISION_SERVICE_KEY, else the shared key file kept next to the backend.
const keyFile = fileURLToPath(new URL('../backend/.service-key', import.meta.url));
const KEY = process.env.VISION_SERVICE_KEY || (existsSync(keyFile) ? readFileSync(keyFile, 'utf8').trim() : '');
const TARGET = process.env.VEHICLE_API || 'http://124.123.89.162:31016';

export default defineConfig({
  esbuild: { jsx: 'automatic', jsxImportSource: 'react' },
  server: {
    fs: { deny: ['.env', '.env.*', '**/.service-key', '**/.git/**'] }, proxy: {
      '/api': {
        target: TARGET,
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
        configure(proxy) {
          proxy.on('proxyReq', request => request.setHeader('X-Local-Vision-Key', KEY));
        },
      },
    }
  }
});
