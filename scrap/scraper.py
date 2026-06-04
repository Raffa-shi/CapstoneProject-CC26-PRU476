import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


class InstagramScraper:
    def __init__(self, driver):
        self.driver = driver

    def login_manual(self):
        self.driver.get("https://www.instagram.com/accounts/login/")
        print("=" * 60)
        print(">>> LOGIN MANUAL di browser yang terbuka...")
        print(">>> Setelah masuk Beranda, tekan ENTER di sini")
        print("=" * 60)
        input()

    # ----------------------------------------------------------------
    # Cari elemen dialog scrollable
    # ----------------------------------------------------------------
    def _find_dialog(self):
        return self.driver.execute_script("""
            var all = document.querySelectorAll('*');
            var best = null, bestDiff = 0;
            for (var i = 0; i < all.length; i++) {
                var s = window.getComputedStyle(all[i]);
                if (s.overflowY === 'scroll' || s.overflowY === 'auto') {
                    var diff = all[i].scrollHeight - all[i].clientHeight;
                    if (diff > bestDiff) { bestDiff = diff; best = all[i]; }
                }
            }
            return best;
        """)

    def _get_scroll_pos(self, el):
        try:
            if el:
                return self.driver.execute_script("return arguments[0].scrollTop;", el)
        except:
            pass
        return self.driver.execute_script("return window.scrollY;")

    def _get_scroll_height(self, el):
        try:
            if el:
                return self.driver.execute_script("return arguments[0].scrollHeight;", el)
        except:
            pass
        return self.driver.execute_script("return document.body.scrollHeight;")

    # ----------------------------------------------------------------
    # Scroll manusiawi: pelan-pelan dengan variasi kecepatan
    # ----------------------------------------------------------------
    def _human_scroll(self, el, total_amount=3000):
        """
        Scroll seperti manusia: banyak langkah kecil dengan jeda pendek.
        Ini memicu lazy-loading Instagram lebih baik dari scrollTop langsung.
        """
        steps = random.randint(8, 15)
        per_step = total_amount // steps
        for _ in range(steps):
            jitter = random.randint(-100, 200)
            amount = per_step + jitter
            if el:
                try:
                    self.driver.execute_script(
                        "arguments[0].scrollTop += arguments[1];", el, amount)
                except:
                    self.driver.execute_script("window.scrollBy(0, arguments[0]);", amount)
            else:
                self.driver.execute_script("window.scrollBy(0, arguments[0]);", amount)
            time.sleep(random.uniform(0.15, 0.45))

    def _scroll_to_bottom(self, el):
        if el:
            try:
                self.driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight;", el)
                return
            except:
                pass
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def _scroll_to_top(self, el):
        if el:
            try:
                self.driver.execute_script("arguments[0].scrollTop = 0;", el)
                return
            except:
                pass
        self.driver.execute_script("window.scrollTo(0, 0);")

    # ----------------------------------------------------------------
    # Klik semua tombol load more / view replies — lebih lengkap
    # ----------------------------------------------------------------
    def _klik_load_more(self):
        clicked = 0
        # XPath multi-bahasa + berbagai format teks
        xpaths = [
            "//*[contains(text(),'View all') and contains(text(),'repl')]",
            "//*[contains(text(),'Load more')]",
            "//*[contains(text(),'Lihat') and contains(text(),'balasan')]",
            "//*[contains(text(),'Lihat semua')]",
            "//*[contains(text(),'View replies')]",
            "//*[contains(text(),'more replies')]",
            "//*[contains(text(),'balasan lainnya')]",
        ]
        for xp in xpaths:
            try:
                btns = self.driver.find_elements(By.XPATH, xp)
                for btn in btns:
                    try:
                        self.driver.execute_script(
                            "arguments[0].scrollIntoView({block:'center'});", btn)
                        time.sleep(0.15)
                        self.driver.execute_script("arguments[0].click();", btn)
                        clicked += 1
                        time.sleep(0.3)
                    except:
                        pass
            except:
                pass

        # Juga coba CSS selector tombol generik
        for css in ["button._a9y3", "svg[aria-label*='Load']"]:
            try:
                btns = self.driver.find_elements(By.CSS_SELECTOR, css)
                for btn in btns:
                    try:
                        self.driver.execute_script("arguments[0].click();", btn)
                        clicked += 1
                    except:
                        pass
            except:
                pass

        return clicked

    # ----------------------------------------------------------------
    # Harvest komentar via JavaScript
    # ----------------------------------------------------------------
    def _harvest(self, post_url, seen, target, collected, owner_set):
        hasil = self.driver.execute_script("""
            var results = [];
            var ownerAccounts = arguments[0];
            var userSpans = document.querySelectorAll('span.xjp7ctv');

            for (var i = 0; i < userSpans.length; i++) {
                try {
                    var userEl = userSpans[i];
                    var username = userEl.innerText.trim();
                    if (!username || username.length < 2) continue;
                    if (ownerAccounts.indexOf(username.toLowerCase()) !== -1) continue;

                    // Naik ke container komentar
                    var container = userEl.parentElement;
                    for (var j = 0; j < 8; j++) {
                        if (!container) break;
                        if ((container.innerText || '').length > username.length + 8) break;
                        container = container.parentElement;
                    }
                    if (!container) continue;

                    var lines = (container.innerText || '').split('\\n');
                    var komentar_lines = [];
                    var noiseWords = [
                        'likes','like','suka','balas','reply','edited','diedit',
                        'view all','load more','lihat semua','lihat balasan',
                        'follow','following','followers','verified','liked by author',
                        'more replies','balasan lainnya'
                    ];

                    for (var k = 0; k < lines.length; k++) {
                        var line = lines[k].trim();
                        if (!line || line.length < 2) continue;
                        if (line === username) continue;
                        if (/^\\d+[wdhmy]$/.test(line) && line.length <= 4) continue;
                        if (['•','·','-','...','–','—','❤','❤️'].indexOf(line) !== -1) continue;

                        var lineLow = line.toLowerCase();
                        var isNoise = false;
                        for (var n = 0; n < noiseWords.length; n++) {
                            if (lineLow === noiseWords[n] ||
                                lineLow.endsWith(' ' + noiseWords[n])) {
                                isNoise = true; break;
                            }
                        }
                        if (isNoise) continue;
                        if (/^\\d[\\d,.]* like/.test(lineLow)) continue;
                        if (/^view all \\d/.test(lineLow)) continue;
                        if (/^lihat \\d/.test(lineLow)) continue;

                        komentar_lines.push(line);
                    }

                    if (komentar_lines.length === 0) continue;
                    var komentar = komentar_lines.join(' ').trim();
                    if (komentar.length < 2 || komentar === username) continue;

                    results.push({username: username, komentar: komentar});
                } catch(e) {}
            }
            return results;
        """, list(owner_set))

        for item in (hasil or []):
            if len(collected) >= target:
                break
            try:
                u = item['username'].strip()
                k = item['komentar'].strip()
                if not u or not k:
                    continue
                key = f"{u}||{k}"
                if key in seen:
                    continue
                seen.add(key)
                collected.append({"Post_URL": post_url, "Username": u, "Komentar": k})
            except:
                continue

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    def extract_comments_from_url(self, post_url, target=7000, owner_accounts=None):
        if owner_accounts is None:
            owner_accounts = set()

        print(f"\n{'='*60}")
        print(f"  URL    : {post_url}")
        print(f"  Target : {target} komentar")
        print(f"{'='*60}")

        self.driver.get(post_url)
        time.sleep(7)

        collected = []
        seen = set()
        round_num = 0
        no_new_streak = 0
        prev_pos = -1
        same_pos_count = 0
        STUCK_LIMIT = 4      # putaran sebelum anggap stuck
        GIVE_UP = 12         # putaran tanpa hasil sebelum menyerah

        dialog = self._find_dialog()
        print(f"  Dialog scroll: {'Terdeteksi ✓' if dialog else 'Tidak ditemukan, pakai window'}")

        while len(collected) < target:
            round_num += 1
            before = len(collected)
            curr_pos = self._get_scroll_pos(dialog)
            scroll_h = self._get_scroll_height(dialog)

            # ── Strategi scroll tergantung kondisi ──────────────────
            if same_pos_count >= STUCK_LIMIT:
                # Stuck parah → reset total: scroll ke atas, refresh dialog, scroll balik
                print(f"  ♻  Reset scroll (stuck {same_pos_count}x) — scrollH={int(scroll_h)}")
                self._scroll_to_top(dialog)
                time.sleep(2)
                dialog = self._find_dialog()  # refresh referensi elemen
                # Scroll ke bawah perlahan
                for _ in range(6):
                    self._human_scroll(dialog, total_amount=4000)
                    time.sleep(0.8)
                self._scroll_to_bottom(dialog)
                time.sleep(3)
                same_pos_count = 0

            else:
                # Normal: human scroll + ke bawah
                self._human_scroll(dialog, total_amount=random.randint(2000, 5000))
                time.sleep(0.5)
                self._scroll_to_bottom(dialog)
                time.sleep(1.5)

            # Klik load more setelah scroll
            clicked = self._klik_load_more()
            time.sleep(1.0)

            # Scroll lagi setelah klik (konten baru muncul setelah klik)
            self._human_scroll(dialog, total_amount=2000)
            time.sleep(1.0)

            # Harvest
            self._harvest(post_url, seen, target, collected, owner_accounts)
            gained = len(collected) - before
            new_pos = self._get_scroll_pos(dialog)

            print(f"  [Putaran {round_num:3d}] "
                  f"klik={clicked:3d}  "
                  f"+{gained:4d}  |  "
                  f"Total: {len(collected):5d}/{target}  |  "
                  f"pos: {int(new_pos):,}  scrollH: {int(scroll_h):,}")

            # Update stuck counter
            if abs(new_pos - prev_pos) < 100:
                same_pos_count += 1
            else:
                same_pos_count = 0
            prev_pos = new_pos

            if gained == 0:
                no_new_streak += 1
                time.sleep(2)
            else:
                no_new_streak = 0

            if no_new_streak >= GIVE_UP:
                print(f"\n  ⚠  {GIVE_UP} putaran tanpa komentar baru — berhenti.")
                break

            time.sleep(random.uniform(0.3, 0.8))

        collected = collected[:target]
        print(f"\n  ✅ Total: {len(collected)} komentar")
        return collected