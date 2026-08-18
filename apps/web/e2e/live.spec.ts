import { engineUp, expect, test } from "./helpers";

/**
 * The full live loop (spec §45): browser → engine over SSE → per-token
 * signals → persisted → replayable. Skips itself when the engine is down
 * so fixture/replay suites stay runnable anywhere.
 */
test.describe("live inference", () => {
  test.beforeEach(async () => {
    test.skip(!(await engineUp()), "trace engine not running on :8000");
  });

  test("prompt streams tokens live, completes, and links to its replay", async ({
    page,
  }) => {
    await page.goto("/explore?prompt=Why%20is%20the%20sky%20blue%3F");

    // SSE envelope arrives, tokens accumulate
    await expect(page.locator('button[title^="p "]').first()).toBeVisible({
      timeout: 120_000, // first hit may load the model
    });

    // terminal state + the persistence payoff: a replay link
    await expect(page.getByText("Complete")).toBeVisible({ timeout: 120_000 });
    const replay = page.getByRole("link", { name: /open replay/ });
    await expect(replay).toBeVisible();

    // follow it: the saved trace replays from Postgres
    await replay.click();
    await expect(page).toHaveURL(/\/trace\/tr_/);
    await expect(
      page.getByText("Saved trace — replaying recorded events"),
    ).toBeVisible();
    await expect(page.locator('button[title^="p "]').first()).toBeVisible({
      timeout: 30_000,
    });
  });

  test("live trace shows layer activity for selected tokens (STANDARD)", async ({
    page,
  }) => {
    await page.goto("/explore?prompt=The%20capital%20of%20France%20is");
    await expect(page.locator('button[title^="p "]').first()).toBeVisible({
      timeout: 120_000,
    });
    await expect(page.getByText("Complete")).toBeVisible({ timeout: 120_000 });

    await page.locator('button[title^="p "]').nth(1).click();
    const inspector = page.getByTestId("inspector");
    await expect(inspector).toBeVisible();
    // the layer panel lists L12…L01 rows when LAYER_ACTIVITY paired in
    await expect(inspector.getByText("L01", { exact: true })).toBeVisible({
      timeout: 10_000,
    });
  });
});
