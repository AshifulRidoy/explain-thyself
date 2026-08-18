import { defineConfig, devices } from "@playwright/test";

/**
 * E2E runs against a live stack: web :3000, engine :8000, Postgres up
 * (`pnpm dev` brings all three up). Tests that need the engine skip
 * themselves when /health is unreachable, so fixture/replay coverage
 * still runs anywhere.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  // one worker: live tests drive a single GPT-2 on the engine — parallel
  // viewports streaming at once contend for the model and flake
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    {
      // spec §34 mobile target: 390px logical width. Chromium keeps the
      // suite on one browser download.
      name: "mobile",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
});
