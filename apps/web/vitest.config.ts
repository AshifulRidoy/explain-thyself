import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Playwright owns e2e/** (run via `pnpm e2e`); vitest owns unit tests only
    exclude: ["e2e/**", "node_modules/**"],
  },
});
