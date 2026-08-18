import { test as base } from "@playwright/test";

/** Skip tests that need the trace engine when it isn't up. */
export const test = base.extend({});
export { expect } from "@playwright/test";

export async function engineUp(): Promise<boolean> {
  try {
    const res = await fetch("http://localhost:8000/health", {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok;
  } catch {
    return false;
  }
}
