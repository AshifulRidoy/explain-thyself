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

  test("sky-blue fixture (RESEARCH) shows per-layer attention with the BOS sink", async ({
    page,
  }) => {
    await page.goto("/explore?fixture=sky-blue");

    // the fixture carries attention — the header dials RESEARCH
    await expect(page.getByText("RESEARCH", { exact: true }).first()).toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByText("Complete")).toBeVisible({ timeout: 60_000 });

    // selecting a token pairs its 12 ATTENTION events into the inspector
    await page.locator('button[title^="p "]').first().click();
    const panel = page.getByTestId("attention-panel");
    await expect(panel).toBeVisible();
    // position −1 — the prepended BOS token GPT-2 sinks attention into
    await expect(panel.getByText("<bos>", { exact: true })).toBeVisible();
    // one selector button per layer
    await expect(page.getByTestId("attention-layers").locator("button")).toHaveCount(12);

    // switching layers re-renders the row set for that layer's measurement
    await page.getByTestId("attention-layers").locator("button", { hasText: "05" }).click();
    await expect(panel.getByText(/L05/)).toBeVisible();
  });

  test("concepts rank by total mass and inspect with their evidence", async ({
    page,
  }) => {
    await page.goto("/explore?fixture=python-rust");

    // the panel fills once concept events land (STANDARD fixture)
    const rows = page.getByTestId("concept-rows");
    await expect(rows).toBeVisible({ timeout: 60_000 });
    await expect(page.getByTestId("concept-row").first()).toBeVisible();
    await expect(
      page.getByText("What concepts were most active?"),
    ).toBeVisible();
    // the method footnote keeps the label honest
    await expect(
      page.getByText(/probability mass on the concept/),
    ).toBeVisible();

    // one canvas node per concept (at its peak) — click inspects it.
    // Wait for Complete: force-clicking a node while the canvas is still
    // stacking new nodes can land on a covering element at 390px.
    await expect(page.getByText("Complete", { exact: true })).toBeVisible({
      timeout: 60_000,
    });
    // at 390px the CONCEPT column can sit beyond the canvas pane — bring
    // the node into the viewport before clicking (force alone cannot)
    const conceptNode = page.locator(".react-flow__node-CONCEPT").first();
    await conceptNode.scrollIntoViewIfNeeded();
    await conceptNode.click({ force: true });
    const inspector = page.getByTestId("inspector");
    await expect(inspector.getByText("CONCEPT", { exact: true })).toBeVisible();
    await expect(inspector.getByText("INTERPRETED", { exact: true })).toBeVisible();
    await expect(
      inspector.getByText(/Evidence — tokens carrying the mass/),
    ).toBeVisible();

    // a token measured with concepts pairs them into its inspector view
    await page.locator('button[title^="p "]').nth(2).click();
    await expect(page.getByTestId("token-concepts")).toBeVisible();
    await expect(
      page.getByTestId("token-concepts").getByText(/interpreted/),
    ).toBeVisible();
  });

  test("uncertainty layer separates four quantities, not one confidence number", async ({
    page,
  }) => {
    await page.goto("/explore?fixture=sky-blue");
    // exact: the panel's empty state says "…after generation completes",
    // which a substring match for "Complete" would race ahead of
    await expect(page.getByText("Complete", { exact: true })).toBeVisible({
      timeout: 60_000,
    });

    const panel = page.getByTestId("uncertainty-panel");
    await expect(panel).toBeVisible();
    await expect(
      page.getByText("What is the model uncertain about?"),
    ).toBeVisible();

    // four separated rows, spec display order
    const rows = panel.getByTestId("uncertainty-row");
    await expect(rows).toHaveCount(4, { timeout: 30_000 });
    // exact: the null rows' basis text repeats the label ("estimating
    // input ambiguity needs…") — a substring match would hit both
    await expect(rows.nth(0).getByText("model uncertainty", { exact: true })).toBeVisible();
    await expect(rows.nth(1).getByText("evidence quality", { exact: true })).toBeVisible();
    await expect(rows.nth(2).getByText("input ambiguity", { exact: true })).toBeVisible();
    await expect(rows.nth(3).getByText("answer stability", { exact: true })).toBeVisible();

    // the refusal is rendered, not hidden — and it ships with its reason
    await expect(
      rows.nth(1).getByText("not measured", { exact: true }),
    ).toBeVisible();
    await expect(rows.nth(1).getByText(/no retrieval sources/)).toBeVisible();

    // stability carries per-perturbation evidence rows
    await expect(panel.getByTestId("uncertainty-variant").first()).toBeVisible();
    await expect(panel.getByText(/never one confidence number/)).toBeVisible();

    // canvas: one node per quantity under the OUTPUT — click inspects it
    await page.locator(".react-flow__node-UNCERTAINTY").first().click({ force: true });
    const inspector = page.getByTestId("inspector");
    await expect(inspector.getByText("UNCERTAINTY", { exact: true })).toBeVisible();
    await expect(inspector.getByText("Basis", { exact: true })).toBeVisible();
  });

  test("counterfactual panel is honest about what a fixture cannot do", async ({
    page,
  }) => {
    await page.goto("/explore?fixture=sky-blue");
    // a committed fixture cannot re-run the model — the panel says so
    // instead of pretending (spec §23's tool needs a live engine)
    const panel = page.getByTestId("counterfactual-panel");
    await expect(panel).toBeVisible({ timeout: 20_000 });
    await expect(panel.getByTestId("counterfactual-empty")).toHaveText(
      /Needs the trace engine/,
    );
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
