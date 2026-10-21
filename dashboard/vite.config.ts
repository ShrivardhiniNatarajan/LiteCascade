import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built straight into the Python package so `python -m litecascade.web.app` serves it.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/litecascade/web/dashboard_dist", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
