import { expect, test } from "./helpers";

/** Saved traces: the DB is the source of truth; replay must work even
 *  with the engine down (events live in Postgres). Requires Postgres. */
test.describe("saved trace replay", () => {
  test("traces list → replay page plays the recorded events", async ({
    page,
  }) => {
    await page.goto("/traces");
    const rows = page.locator("ol a");
    const count = await rows.count();
    test.skip(count === 0, "no saved traces in Postgres yet");

    await rows.first().click();
    await expect(page).toHaveURL(/\/trace\/tr_/);
    await expect(
      page.getByText("Saved trace — replaying recorded events"),
    ).toBeVisible();

    // recorded events replay through the same instrument
    await expect(page.locator('button[title^="p "]').first()).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.getByText("Complete")).toBeVisible({ timeout: 60_000 });

    // and the header carries the replay mode
    await expect(page.getByText("Replay", { exact: true })).toBeVisible();
  });

  test("unknown trace id is a clean 404", async ({ page }) => {
    const res = await page.goto("/trace/tr_nope0000");
    expect(res?.status()).toBe(404);
  });
});
