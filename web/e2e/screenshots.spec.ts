import { test } from "@playwright/test";

test.describe("Visual Quality Loop Screenshots", () => {
  test("capture mobile viewport (390x844)", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    // Let initial data load
    await page.waitForTimeout(1000);
    await page.screenshot({
      path: "../docs/screenshots/mobile_390x844.png",
      fullPage: true,
    });
  });

  test("capture tablet viewport (768x1024)", async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto("/");
    await page.waitForTimeout(1000);
    await page.screenshot({
      path: "../docs/screenshots/tablet_768x1024.png",
      fullPage: true,
    });
  });

  test("capture desktop viewport (1440x900)", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");
    await page.waitForTimeout(1000);
    await page.screenshot({
      path: "../docs/screenshots/desktop_1440x900.png",
      fullPage: true,
    });
  });
});
