import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from scraper import InstagramScraper

# ================================================================
#  KONFIGURASI
# ================================================================
POST_URLS = [
    "https://www.instagram.com/p/DO6EIPujOYw/",
    # Tambah link lain jika kurang dari 7000:
    # "https://www.instagram.com/p/LINK_2/",
]

# Username akun yang mem-posting (komentar dari akun ini dibuang)
OWNER_ACCOUNTS = {
    "bpjskesehatan_ri",   # ← sesuai filter yang sudah benar
}

TARGET_TOTAL = 7000
OUTPUT_FILE  = "data/hasil_komentar.csv"
# ================================================================


def main():
    os.makedirs("data", exist_ok=True)

    chrome_ops = Options()
    chrome_ops.add_argument("--disable-notifications")
    chrome_ops.add_argument("--start-maximized")
    chrome_ops.add_argument("--disable-blink-features=AutomationControlled")
    chrome_ops.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_ops.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_ops,
    )
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )

    bot = InstagramScraper(driver)

    try:
        bot.login_manual()
        all_comments = []

        for idx, url in enumerate(POST_URLS):
            sisa = TARGET_TOTAL - len(all_comments)
            if sisa <= 0:
                break

            print(f"\n{'#'*60}")
            print(f"  Postingan {idx+1}/{len(POST_URLS)}")
            print(f"  Terkumpul: {len(all_comments)} | Butuh: {sisa}")
            print(f"  Filter   : {OWNER_ACCOUNTS}")
            print(f"{'#'*60}")

            data = bot.extract_comments_from_url(url, target=sisa, owner_accounts=OWNER_ACCOUNTS)
            all_comments.extend(data)

            if len(all_comments) >= TARGET_TOTAL:
                break
            if idx < len(POST_URLS) - 1:
                jeda = random.randint(8, 15)
                print(f"  Jeda {jeda} detik...")
                time.sleep(jeda)

        # Simpan
        df = pd.DataFrame(all_comments)
        print(f"\n{'='*60}")
        if not df.empty:
            df = df.drop_duplicates(subset=["Username", "Komentar"]).reset_index(drop=True)
            df = df.iloc[:TARGET_TOTAL]
            df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            print(f"  ✅ SELESAI — {len(df)} komentar → {OUTPUT_FILE}")
            print(f"\n  Preview 10 baris:")
            print(df[["Username", "Komentar"]].head(10).to_string(index=False))
            if len(df) < TARGET_TOTAL:
                print(f"\n  ⚠  Kurang {TARGET_TOTAL - len(df)} komentar.")
                print(f"     Tambah URL postingan lain di POST_URLS.")
        else:
            print("  ❌ GAGAL — 0 komentar.")
        print(f"{'='*60}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()