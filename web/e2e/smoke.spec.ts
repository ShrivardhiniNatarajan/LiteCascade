import { test, expect } from "@playwright/test";

test.describe("LiteCascade Demo Dashboard E2E", () => {
  test("loads, streams inference, injects attack, turns red, and pauses", async ({ page }) => {
    // 1. Load application
    await page.goto("/");
    await expect(page.getByText("LiteCascade")).toBeVisible();
    await expect(page.getByText(/N-BaIoT test split|Synthetic fixture/i).first()).toBeVisible();

    // Initial state check
    const streamButton = page.getByRole("button", { name: /Resume Stream/i });
    await expect(streamButton).toBeVisible();

    // 2. Start stream
    await streamButton.click();
    await expect(page.getByRole("button", { name: /Pause Stream/i })).toBeVisible();

    // Wait for at least one log entry to appear in inference log
    await expect(
      page.locator("div").filter({ hasText: /E1:Sentinel|E2:Analyst/ }).first()
    ).toBeVisible({ timeout: 10000 });

    // 3. Inject attack
    const injectBtn = page.getByRole("button", { name: /Inject Botnet Flow/i });
    await injectBtn.click();

    // Verify modal opened
    await expect(page.getByRole("heading", { name: /Inject Simulated Botnet/i })).toBeVisible();

    // Click Inject Attack Segment inside modal
    const submitInjectBtn = page.getByRole("button", { name: /Inject Attack Segment/i });
    await submitInjectBtn.click();

    // Wait for intrusion alert status in header (turns red)
    await expect(page.getByText(/INTRUSION ALERT/i)).toBeVisible({ timeout: 15000 });

    // Verify red alert badge/log line appears in the inference log
    await expect(page.getByText(/ALERT:/i).first()).toBeVisible({ timeout: 15000 });

    // 4. Pause stream
    const pauseBtn = page.getByRole("button", { name: /Pause Stream/i });
    await pauseBtn.click();
    await expect(page.getByRole("button", { name: /Resume Stream/i })).toBeVisible();
  });
});
