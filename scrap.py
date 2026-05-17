import asyncio
import json
import pandas as pd

from playwright.async_api import async_playwright

POST_URL = "https://www.instagram.com/p/DO6EIPujOYw/?img_index=1"

comments_data = []


async def main():

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        context = await browser.new_context()

        # =========================
        # LOAD COOKIES
        # =========================

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

        # =========================
        # RESPONSE INTERCEPTOR
        # =========================

        async def handle_response(response):

            try:

                url = response.url

                # endpoint komentar instagram
                if "comments" in url:

                    print("\nFOUND COMMENTS API")

                    data = await response.json()

                    # DEBUG
                    print(data.keys())

                    # parsing comment
                    if "comments" in data:

                        for item in data["comments"]:

                            try:

                                username = item["user"]["username"]

                                comment = item["text"]

                                comments_data.append({
                                    "username": username,
                                    "comment": comment
                                })

                                print(username, ":", comment[:40])

                            except:
                                pass

            except:
                pass

        page.on("response", handle_response)

        # =========================
        # OPEN POST
        # =========================

        print("Opening post...")

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

        # =========================
        # AUTO SCROLL
        # =========================

        for i in range(30):

            print(f"Scrolling {i+1}")

            await page.mouse.wheel(0, 8000)

            await page.wait_for_timeout(3000)

        # =========================
        # SAVE DATA
        # =========================

        df = pd.DataFrame(comments_data)

        df.drop_duplicates(inplace=True)

        df.to_csv(
            "instagram_comments.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nFINAL RESULT")
        print(df.shape)

        await browser.close()


asyncio.run(main())