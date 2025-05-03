import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Improved ngrok configuration
    // Disable HMR when running behind a proxy like ngrok
    hmr: false,
    host: '0.0.0.0',
    // Allow any host to access the development server
    cors: true,
    // Add allowedHosts configuration to accept ngrok domains
    allowedHosts: ['localhost', '127.0.0.1', '.ngrok-free.app', '18cf-2001-9e8-85d5-1d00-35ac-6944-9820-ea0c.ngrok-free.app', 'all'],
    // Proxy API requests to the backend during development
    proxy: {
      '/api': {
        target: 'http://api:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
}) 