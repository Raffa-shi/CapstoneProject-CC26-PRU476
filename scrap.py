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
            slow_mo=300,
            args=[
                "--disable-blink-features=AutomationControlled"
            ]
        )

        context = await browser.new_context()

        # ==========================================
        # LOAD COOKIES
        # ==========================================

        with open(
            "instagram_cookies.json",
            "r",
            encoding="utf-8"
        ) as f:

            cookies = json.load(f)

        clean_cookies = []

        for cookie in cookies:

            try:

                clean_cookie = {
                    "name": cookie["name"],
                    "value": cookie["value"],
                    "domain": cookie["domain"],
                    "path": cookie.get("path", "/")
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

        # ==========================================
        # OPEN PAGE
        # ==========================================

        page = await context.new_page()

        print("OPENING POST")

        await page.goto(
            POST_URL,
            timeout=120000
        )

        await page.wait_for_timeout(10000)

        print(page.url)

        if "login" in page.url:

            print("COOKIES INVALID")
            return

        print("LOGIN SUCCESS")

        # ==========================================
        # SCROLL TO COMMENT AREA
        # ==========================================

        await page.mouse.wheel(0, 2500)

        await page.wait_for_timeout(5000)

        # ==========================================
        # LOAD COMMENTS
        # ==========================================

        last_count = 0
        same_count = 0

        for i in range(100):

            print(f"\nITERATION {i+1}")

            # ======================================
            # CLICK MORE COMMENTS BUTTON
            # ======================================

            buttons = await page.query_selector_all("button")

            for btn in buttons:

                try:

                    text = (
                        await btn.inner_text()
                    ).lower().strip()

                    if (
                        "more comments" in text
                        or "load more comments" in text
                        or "lihat komentar lainnya" in text
                        or "view replies" in text
                        or "lihat balasan" in text
                    ):

                        print("CLICK:", text)

                        await btn.click()

                        await page.wait_for_timeout(
                            random.randint(1000, 2500)
                        )

                except:
                    pass

            # ======================================
            # SCROLL COMMENT PANEL
            # ======================================

            await page.mouse.wheel(
                0,
                random.randint(5000, 9000)
            )

            await page.wait_for_timeout(
                random.randint(3000, 5000)
            )

            # ======================================
            # COUNT COMMENT BLOCKS
            # ======================================

            comment_blocks = await page.query_selector_all(
                'article div > span'
            )

            current_count = len(comment_blocks)

            print("COLLECTED:", current_count)

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

        # ==========================================
        # EXTRACTION
        # ==========================================

        print("\nEXTRACTING COMMENTS")

        comment_blocks = await page.query_selector_all(
            'article ul li'
        )

        print("TOTAL BLOCKS:", len(comment_blocks))

        for idx, block in enumerate(comment_blocks):

            try:

                raw_text = await block.inner_text()

                lines = [
                    x.strip()
                    for x in raw_text.split("\n")
                    if x.strip()
                ]

                if len(lines) < 2:
                    continue

                username = lines[0]
                comment = lines[1]

                invalid_words = [
                    "reply",
                    "replies",
                    "like",
                    "liked",
                    "follow",
                    "following",
                    "see translation"
                ]

                if any(
                    x in comment.lower()
                    for x in invalid_words
                ):
                    continue

                if (
                    username
                    and comment
                    and username != comment
                    and len(comment) > 1
                ):

                    all_comments.append({
                        "username": username,
                        "comment": comment
                    })

                    print(
                        f"SAVED {len(all_comments)} -> {username}"
                    )

            except Exception as e:

                print("EXTRACT ERROR:", e)

        # ==========================================
        # SAVE CSV
        # ==========================================

        df = pd.DataFrame(all_comments)

        df.drop_duplicates(inplace=True)

        df.to_csv(
            "instagram_comments.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nFINAL RESULT")
        print(df.shape)

        print("CSV SAVED")

        await browser.close()


asyncio.run(main())