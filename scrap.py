import asyncio
import json
import random
import pandas as pd

from playwright.async_api import async_playwright

POST_URL = "https://www.instagram.com/p/DO6EIPujOYw/?img_index=1"

TARGET = 1000

all_comments = []


async def main():

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False,
            slow_mo=500
        )

        context = await browser.new_context()

        # ===================================
        # LOAD COOKIES
        # ===================================

        with open("instagram_cookies.json", "r", encoding="utf-8") as f:
            raw_cookies = json.load(f)

        clean_cookies = []

        for cookie in raw_cookies:

            try:

                clean_cookie = {
                    "name": cookie.get("name"),
                    "value": cookie.get("value"),
                    "domain": cookie.get("domain"),
                    "path": cookie.get("path", "/"),
                }

                if cookie.get("expirationDate"):

                    try:
                        clean_cookie["expires"] = int(
                            cookie["expirationDate"]
                        )
                    except:
                        pass

                clean_cookies.append(clean_cookie)

            except:
                pass

        await context.add_cookies(clean_cookies)

        page = await context.new_page()

        # ===================================
        # OPEN POST
        # ===================================

        print("Opening Instagram post...")

        await page.goto(
            POST_URL,
            timeout=120000
        )

        await page.wait_for_timeout(10000)

        print("Current URL:", page.url)

        if "login" in page.url:

            print("Cookies invalid")
            return

        print("Login success")

        # ===================================
        # CLICK COMMENT AREA
        # ===================================

        await page.mouse.wheel(0, 2500)

        await page.wait_for_timeout(5000)

        # ===================================
        # FIND COMMENT SCROLL CONTAINER
        # ===================================

        comment_container = await page.query_selector(
            'div[style*="overflow"]'
        )

        if not comment_container:

            print("Comment container not found")
            return

        print("Comment container found")

        # ===================================
        # SCRAPING LOOP
        # ===================================

        last_count = 0
        same_count = 0

        for i in range(100):

            print(f"\nSCROLL ITERATION {i+1}")

            # ===============================
            # CLICK MORE COMMENTS BUTTON
            # ===============================

            buttons = await page.query_selector_all("button")

            for btn in buttons:

                try:

                    text = await btn.inner_text()

                    text = text.lower()

                    if (
                        "more comments" in text
                        or "view replies" in text
                        or "lihat komentar lainnya" in text
                        or "lihat balasan" in text
                    ):

                        print("CLICK:", text)

                        await btn.click()

                        await page.wait_for_timeout(
                            random.randint(1000, 2500)
                        )

                except:
                    pass

            # ===============================
            # SCROLL COMMENT CONTAINER
            # ===============================

            try:

                await comment_container.evaluate(
                    """
                    (el) => {
                        el.scrollTop = el.scrollHeight;
                    }
                    """
                )

            except Exception as e:

                print("Scroll error:", e)

            await page.wait_for_timeout(
                random.randint(3000, 5000)
            )

            # ===============================
            # GET COMMENTS
            # ===============================

            comments = await page.query_selector_all("ul li")

            current_count = len(comments)

            print("Collected:", current_count)

            # ===============================
            # STOP CONDITION
            # ===============================

            if current_count >= TARGET:

                print("TARGET REACHED")
                break

            if current_count == last_count:
                same_count += 1
            else:
                same_count = 0

            if same_count >= 5:

                print("NO NEW COMMENTS")
                break

            last_count = current_count

        # ===================================
        # EXTRACTION
        # ===================================

        print("\nEXTRACTING DATA")

        comments = await page.query_selector_all("ul li")

        print("TOTAL ELEMENTS:", len(comments))

        for c in comments:

            try:

                username_el = await c.query_selector("a")

                spans = await c.query_selector_all("span")

                if not username_el:
                    continue

                username = await username_el.inner_text()

                comment_text = ""

                for s in spans:

                    try:

                        txt = await s.inner_text()

                        txt = txt.strip()

                        if (
                            txt
                            and txt != username
                            and len(txt) > 2
                        ):

                            comment_text = txt
                            break

                    except:
                        pass

                if comment_text:

                    all_comments.append({
                        "username": username,
                        "comment": comment_text
                    })

            except Exception as e:

                print("Extract error:", e)

        # ===================================
        # SAVE CSV
        # ===================================

        df = pd.DataFrame(all_comments)

        df.drop_duplicates(inplace=True)

        df.to_csv(
            "instagram_comments.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nFINAL RESULT")
        print(df.shape)

        print("\nCSV SAVED")

        await browser.close()


asyncio.run(main())