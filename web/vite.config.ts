import { defineConfig } from 'vite';

declare const process: {
  env: {
    BUNNYLAND_RL_BASE?: string;
    BUNNYLAND_API_PROXY?: string;
  };
};

const apiProxyTarget = process.env.BUNNYLAND_API_PROXY || 'http://127.0.0.1:8876';

export default defineConfig({
  base: process.env.BUNNYLAND_RL_BASE || '/rl/',
  server: {
    proxy: {
      '/api': {
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, '') || '/',
        target: apiProxyTarget,
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        admin: 'admin.html',
      },
    },
    outDir: 'dist',
    sourcemap: true,
  },
});
