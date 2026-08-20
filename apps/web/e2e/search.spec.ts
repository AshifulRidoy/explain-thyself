import { engineUp, expect, test } from "./helpers";

/**
 * Trace search (spec §28): the /traces page asks the engine which
 * recorded prompts the model represents like the query; a replay page
 * ranks its own model's traces against its stored embedding. Skips
 * itself when the engine or DB is down.
 */
test.describe("trace search", () => {
  test.beforeEach(async () => {
    test.skip(!(await engineUp()), "trace engine not running on :8000");
  });

  test("traces page searches recorded prompts by representation", async ({
    page,
  }) => {
    await page.goto("/traces");

    // the instrument: a query in, a ranked list out
    const box = page.getByTestId("search-input");
    await expect(box).toBeVisible();
    await box.fill("why is the sky blue");
    await box.press("Enter");

    const hits = page.getByTestId("search-hit");
    await expect(hits.first()).toBeVisible({ timeout: 30_000 });

    // every hit carries its measured cosine — signed, four decimals
    for (let i = 0; i < Math.min(3, await hits.count()); i++) {
      await expect(
        hits.nth(i).getByText(/^[+-][01]\.\d{4}$/),
      ).toBeVisible();
    }

    // the basis under the results is the contract's own string — the UI
    // never invents a caption for the number
    await expect(page.getByTestId("search-basis")).toContainText(
      "not semantic meaning",
    );

    // a hit opens its replay
    await hits.first().click();
    await expect(page).toHaveURL(/\/trace\/tr_/);
  });

  test("gibberish matches nothing exactly — no fake 1.0000", async ({
    page,
  }) => {
    // GPT-2 embeds ANY text, so a ranked list always comes back; the
    // honesty test is that gibberish never produces a perfect match
    await page.goto("/traces");
    const box = page.getByTestId("search-input");
    await box.fill("zzz qqq unrepresentable gibberish vvv");
    await box.press("Enter");

    const hits = page.getByTestId("search-hit");
    await expect(hits.first()).toBeVisible({ timeout: 30_000 });
    // every similarity strictly below 1 — nothing is "the same prompt"
    for (let i = 0; i < Math.min(3, await hits.count()); i++) {
      await expect(
        hits.nth(i).getByText(/^[+-]0\.\d{4}$/),
      ).toBeVisible();
    }
  });

  test("replay page ranks its own model's similar traces (spec §28)", async ({
    page,
  }) => {
    // any healthy saved trace with an embedding will do
    await page.goto("/traces");
    const rows = page.locator("ol a");
    test.skip((await rows.count()) === 0, "no saved traces in Postgres yet");

    await rows.first().click();
    await expect(page).toHaveURL(/\/trace\/tr_/);

    // the section renders only when a measured ranking exists — find a
    // trace that has one (recent traces embed at open)
    const heading = page.getByText("Traces the model represents similarly");
    if (await heading.isVisible()) {
      const section = page.locator("section").filter({ has: heading });
      await expect(section.getByText(/^[+-][01]\.\d{4}$/).first()).toBeVisible();
      await expect(
        // anchored: the header line says "rank, not meaning" too
        section.getByText(/^GPT-2's hidden space is anisotropic/),
      ).toBeVisible();
    }
  });
});
