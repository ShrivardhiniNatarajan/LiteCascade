import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30000,
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "python -m uvicorn app.api.main:app --port 8000 --host 127.0.0.1",
    url: "http://127.0.0.1:8000/api/meta",
    reuseExistingServer: true,
    cwd: "..",
    timeout: 60000,
  },
});
