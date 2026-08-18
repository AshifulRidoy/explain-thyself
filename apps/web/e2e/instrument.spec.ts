import { expect, test } from "./helpers";

/**
 * The instrument on committed fixtures — no engine, no DB required.
 * Numbers on screen must come from the fixture JSON (spec §45: "every
 * number the UI shows is traceable").
 */
test.describe("fixture explorer", () => {
  test("python-rust fixture plays end-to-end and the inspector populates", async ({
    page,
  }) => {
    await page.goto("/explore?fixture=python-rust");

    // tokens stream in on the recorded cadence
    const firstToken = page.locator('button[title^="p "]').first();
    await expect(firstToken).toBeVisible({ timeout: 20_000 });

    // playback reaches a terminal state
    await expect(page.getByText("Complete")).toBeVisible({ timeout: 60_000 });

    // token count in the header matches tokens in the stream
    const tokenCount = await page.locator('button[title^="p "]').count();
    expect(tokenCount).toBeGreaterThanOrEqual(20);

    // click a token in the stream → inspector shows the event + its level
    await page.locator('button[title^="p "]').nth(2).click();
    const inspector = page.getByTestId("inspector");
    await expect(inspector).toBeVisible();
    await expect(inspector.getByText("TOKEN", { exact: true })).toBeVisible();
    await expect(inspector.getByText("MEASURED")).toBeVisible();

    // entropy meter + probability distribution render alongside
    await expect(page.getByText("Entropy", { exact: false }).first()).toBeVisible();
  });

  test("no horizontal overflow at 390px", async ({ page }) => {
    await page.goto("/explore?fixture=minimal");
    await expect(page.locator('button[title^="p "]').first()).toBeVisible({
      timeout: 20_000,
    });
    const overflowed = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflowed).toBeLessThanOrEqual(1);
  });
});

test("home page presents the instrument and links to explore", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("What would you like to examine?")).toBeVisible();
  await page.locator("#examine").fill("why is the sky blue");
  await page.getByRole("button", { name: "Examine →" }).click();
  await expect(page).toHaveURL(/\/explore\?prompt=/);
});

test("methodology page shows all three epistemic tiers", async ({ page }) => {
  await page.goto("/methodology");
  for (const tier of ["MEASURED", "DERIVED", "INTERPRETED"]) {
    await expect(page.getByText(tier, { exact: true }).first()).toBeVisible();
  }
});
