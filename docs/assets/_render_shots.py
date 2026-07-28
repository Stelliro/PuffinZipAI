"""Capture live WebUI screenshots for the README.

Prerequisites:
  - WebUI running on http://127.0.0.1:5001 (e.g. `python webui_server.py`)
  - `webui_credentials.json` present (created on first launcher run)
  - Playwright: `pip install playwright && playwright install chromium`

Usage:
  python docs/assets/_render_shots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:5001"
CREDS_PATH = ROOT / "webui_credentials.json"


def main() -> None:
    if not CREDS_PATH.is_file():
        print(f"Missing {CREDS_PATH.name} — start the WebUI once to generate it.", file=sys.stderr)
        sys.exit(1)

    creds = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    user = creds["username"]
    password = creds["password"]
    ASSETS.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=2)

        page.goto(f"{BASE}/login", wait_until="networkidle", timeout=60_000)

        # Login form
        page.fill("input[name='username'], #username, input[type='text']", user)
        page.fill("input[name='password'], #password, input[type='password']", password)
        page.click("button[type='submit'], input[type='submit']")
        page.wait_for_url("**/dashboard**", timeout=30_000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        page.screenshot(path=str(ASSETS / "webui-dashboard.png"), full_page=False, type="png")
        print("wrote webui-dashboard.png")

        # Evolution Deep Dive tab
        page.locator("button[data-top-tab='deep-dive'], .top-tab-btn[data-top-tab='deep-dive']").first.click()
        page.wait_for_timeout(2000)
        page.screenshot(path=str(ASSETS / "webui-deep-dive.png"), full_page=False, type="png")
        print("wrote webui-deep-dive.png")

        browser.close()

    print("done")


if __name__ == "__main__":
    main()
