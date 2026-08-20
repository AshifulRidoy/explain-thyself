import { engineUp, expect, test } from "./helpers";

/**
 * Model comparison (spec Phase 7, V2 cut): the same prompt through two
 * registered models. The panel runs the anchor trace's prompt through
 * the second model (its own recorded trace), shows both answers side by
 * side with the measured token agreement, and restores stored
 * comparisons on the replay page. Skips itself when the engine is down.
 */
test.describe("model comparison", () => {
  test.beforeEach(async () => {
    test.skip(!(await engineUp()), "trace engine not running on :8000");
  });

  test("same prompt through gpt2-small and distilgpt2, measured", async ({
    page,
  }) => {
    test.setTimeout(240_000); // model B may need loading on first run
    await page.goto("/explore?prompt=Why%20is%20the%20sky%20blue%3F");
    await expect(page.getByText("Complete", { exact: true })).toBeVisible({
      timeout: 120_000,
    });

    const panel = page.getByTestId("compare-panel");
    await expect(panel).toBeVisible();

    // the picker offers the other same-tokenizer model — never the fake
    // (the engine refuses mismatched tokenizers; the UI does not offer them)
    const select = panel.getByTestId("compare-model-select");
    await expect(select).toBeVisible();
    const options = select.locator("option");
    await expect(options).toHaveCount(1);
    await expect(options.first()).toHaveText("distilgpt2");

    // run it — model B's answer streams in as progress frames land
    await panel.getByTestId("compare-run").click();
    const result = panel.getByTestId("compare-result");
    await expect(result).toBeVisible({ timeout: 180_000 });

    // both answers, each with its own token count
    await expect(result.getByTestId("compare-output-a")).toHaveText(
      /^gpt2-small · \d+ tokens$/,
    );
    await expect(result.getByTestId("compare-output-b")).toHaveText(
      /^distilgpt2 · \d+ tokens$/,
    );

    // the measured numbers: agreement, compared positions, divergence
    await expect(result.getByTestId("compare-agreement")).toHaveText(/^\d+\.\d%$/);
    await expect(
      result.getByText(/\d+\/\d+ compared positions agree/),
    ).toBeVisible();
    await expect(
      result.getByText(/first change at token|identical over the compared range/),
    ).toBeVisible();

    // the basis ships verbatim — the UI never captions the number itself
    await expect(result.getByTestId("compare-basis")).toContainText(
      "not internal similarity",
    );

    // model B's answer is itself a recorded trace — open its replay
    const link = result.getByTestId("compare-replay-link");
    await expect(link).toHaveAttribute("href", /\/trace\/tr_/);
    await link.click();
    await expect(page).toHaveURL(/\/trace\/tr_/);
  });

  test("replay page restores stored comparisons without re-running", async ({
    page,
  }) => {
    test.setTimeout(240_000);
    // any saved trace with a stored comparison will do; make one first
    await page.goto("/explore?prompt=Why%20is%20the%20sky%20blue%3F");
    await expect(page.getByText("Complete", { exact: true })).toBeVisible({
      timeout: 120_000,
    });
    const panel = page.getByTestId("compare-panel");
    await panel.getByTestId("compare-run").click();
    await expect(panel.getByTestId("compare-result")).toBeVisible({
      timeout: 180_000,
    });

    // follow the anchor's replay link — the comparison restores from
    // the comparisons table, no model loaded
    await page.getByRole("link", { name: /open replay/ }).click();
    await expect(page).toHaveURL(/\/trace\/tr_/);
    const restored = page.getByTestId("compare-result");
    await expect(restored).toBeVisible({ timeout: 30_000 });
    await expect(restored.getByTestId("compare-agreement")).toHaveText(/^\d+\.\d%$/);
    await expect(restored.getByTestId("compare-basis")).toContainText(
      "not internal similarity",
    );
  });

  test("fixture mode says honestly that it cannot compare", async ({ page }) => {
    await page.goto("/explore?fixture=sky-blue");
    const panel = page.getByTestId("compare-panel");
    await expect(panel).toBeVisible();
    await expect(panel.getByTestId("compare-empty")).toContainText(
      "cannot load another model",
    );
  });
});
