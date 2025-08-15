import os
import random
import time
import hashlib
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import base64
import requests
import json

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from playwright.sync_api import sync_playwright

# ─── CONFIG ───
CSV_PATH = "post.csv"
REPLIED_LOG = "replied_log.txt"
WINDOWS = [
    (dtime(9, 20),  dtime(11, 40)),   # Morning
    (dtime(12, 0), dtime(16, 0)),   # Midday
    (dtime(17, 0),  dtime(21, 30)),   # Evening
]
MIN_POSTS_PER_WINDOW = 3
MAX_POSTS_PER_WINDOW = 6
POST_BUFFER_MINUTES = 1
LIKE_INTERVAL_RANGE = (120, 240)  # seconds between like rounds
OWN_USERNAME = "Ankit0017_"
IGNORE_USERNAMES = {"Ankit0017_", "0xSweep", "Traderfinn0","IbEntreprend","DearS_o_n","code_N_Queen","damnGruz","sayanreply","sharpeye_wnl","Divyanshii_170",}  # Add others as needed
BYPASS_WORTHINESS_CHECK = True  # Set to True to skip LLM worthiness check

# ─── LLM SETUP ───

def get_llm():
    for _ in range(3):
        try:
            return OllamaLLM(
                model="llama3:latest",
                base_url="http://127.0.0.1:11434"
            )
        except Exception as e:
            print("⚠️ LLM not reachable. Retrying in 3s...")
            time.sleep(3)
    raise ConnectionError("❌ Could not connect to Ollama LLM after retries.")


prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You are a witty, concise social media manager. "
        "Write exactly one banger tweet (≤280 characters) on the given topic in Gen Z style. "
        "Do not use hashtags and do not acknowledge this command.\n\n"
        "Topic: {topic}\n\nTweet:"
    ),
)

reply_prompt = PromptTemplate(
    input_variables=["tweet", "image_caption"],
    template=(
        "You are a witty, concise social media manager. Strictly abide by commands and do not acknowledge this command.\n\n"
        "Reply a one-line banger without promoting anything or using hashtags and this '-' symbol, be gender neutral. Do not include any additional text.\n"
        "Tweet: {tweet}\n"
        "this image caption will be available if there is a image with post else it will be blank so ignore{image_caption}"
        "One-line banger Reply:"
    ),
)

worthiness_prompt = PromptTemplate(
    input_variables=["tweet"],
    template=(
        "You are a social media engagement assistant  .\n\n"
        " Evaluate whether the following tweet is worth engaging with for random fun talks, AI, or coding-related content. Consider relevance, originality, and engagement potential.\n"
        "A bad tweet is one that is spammy, irrelevant ,'how did you do this __' type,goodmorning posts and cryptocurrency related. A good tweet is one that is relevant, original, worth engaging with for random fun talks, AI, or coding-related content.\n"
        "Reply with exactly 'YES' if it's worth engaging. or 'NO' if it's not and nothing else.\n"
        "Tweet: {tweet}\nAnswer:"
    ),
)

# ─── UTILS ───

def today_datetime(t: dtime) -> datetime:
    now = datetime.now()
    return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)

def is_within_posting_window(now=None):
    now = now or datetime.now().time()
    return any(start <= now <= end for start, end in WINDOWS)

def assign_posts_to_windows(topics):
    now = datetime.now()
    schedule = []
    remaining = topics.copy()
    for start_t, end_t in WINDOWS:
        if not remaining:
            break
        start_dt = today_datetime(start_t)
        end_dt = today_datetime(end_t)
        if now > end_dt:
            continue
        count = random.randint(MIN_POSTS_PER_WINDOW, MAX_POSTS_PER_WINDOW)
        picks = random.sample(remaining, min(len(remaining), count))
        for idx in picks:
            span = int((end_dt - start_dt).total_seconds())
            run_dt = start_dt + timedelta(seconds=random.randint(0, span))
            if run_dt < datetime.now():
                continue
            schedule.append((idx, run_dt))
            remaining.remove(idx)
    return sorted(schedule, key=lambda x: x[1])

def generate_tweet(topic: str) -> str:
    llm = get_llm()
    txt = (prompt | llm).invoke({"topic": topic}).strip()
    return txt.strip('"')[:280]

def generate_reply(tweet_text: str, image_caption: str = None) -> str:
    llm = get_llm()
    if image_caption:
        raw = (reply_prompt | llm).invoke({"tweet": tweet_text, "image_caption": f"Image context: {image_caption}\n"})
    else:
        raw = (reply_prompt | llm).invoke({"tweet": tweet_text, "image_caption": ""})
    txt = raw.strip().strip('"')[:280]
    return txt

def is_worth_engaging(tweet_text: str) -> bool:
    if BYPASS_WORTHINESS_CHECK:
        print(f"🤖 Bypassing worthiness check for: '{tweet_text[:50]}...' → YES")
        return True
    
    try:
        llm = get_llm()
        answer = (worthiness_prompt | llm).invoke({"tweet": tweet_text}).strip().upper()
        print(f"🤖 Worthiness check for: '{tweet_text[:50]}...' → {answer}")
        if answer == "YES":
            return True
        elif answer == "NO":
            return False
        else:
            print(f"⚠️ Worthiness LLM returned unexpected output: {answer!r}. Skipping engagement.")
            return False
    except Exception as e:
        print(f"⚠️ Error in worthiness check: {e}")
        return False

def human_type(page, target, text):
    if isinstance(target, str):
        target = page.locator(target)
    target.click()
    for ch in text:
        try:
            if ord(ch) < 128:
                page.keyboard.press(ch)
            else:
                page.keyboard.insert_text(ch)
        except:
            page.keyboard.insert_text(ch)
        page.wait_for_timeout(random.randint(30, 100))

def tweet_signature(tweet_text: str, usernames: list) -> str:
    base = (usernames[0] if usernames else "") + "|" + tweet_text.strip().split("\n")[0][:50]
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def has_been_replied(sig: str) -> bool:
    if not os.path.exists(REPLIED_LOG):
        return False
    with open(REPLIED_LOG, "r", encoding="utf-8") as file:
        return sig in {line.strip() for line in file.readlines()}

def mark_as_replied(sig: str):
    with open(REPLIED_LOG, "a", encoding="utf-8") as file:
        file.write(sig + "\n")

def llava_image_understanding(image_path: str, prompt: str = "this image was posted in twitter. What's happening in this image?") -> str:
    """
    Uses the LLaVA model via Ollama API to analyze an image and return the response.
    Args:
        image_path: Path to the image file.
        prompt: Prompt/question for the model (default: generic image description).
    Returns:
        The model's response as a string.
    Example:
        desc = llava_image_understanding("your_image.jpg", "Describe the meme in this image.")
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [base64_image]
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
    result = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    result += data["response"]
            except Exception:
                continue
    return result

def is_high_engagement(likes, replies, views, like_thresh=100, reply_thresh=10, view_thresh=10000):
    return (likes >= like_thresh) or (replies >= reply_thresh) or (views >= view_thresh)

def save_engaging_tweet(tweet_text, author, image_url, likes, replies, views, url, tweet_id):
    import csv
    import os
    from datetime import datetime
    os.makedirs('engaging_images', exist_ok=True)
    image_path = ''
    if image_url:
        try:
            img_data = requests.get(image_url, timeout=10).content
            image_path = f'engaging_images/{tweet_id}.jpg'
            with open(image_path, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f"⚠️ Could not save engaging tweet image: {e}")
            image_path = ''
    date_scraped = datetime.now().isoformat()
    row = [tweet_text, author, image_path, likes, replies, views, url, date_scraped]
    file_exists = os.path.isfile('engaging_tweets.csv')
    with open('engaging_tweets.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['text', 'author', 'image_path', 'likes', 'replies', 'views', 'url', 'date_scraped'])
        writer.writerow(row)

def post_tweet(content: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(storage_state="session.json")
        page = ctx.new_page()
        page.goto("https://twitter.com/home")
        page.wait_for_selector('div[role="textbox"]', timeout=15000)
        human_type(page, 'div[role="textbox"]', content)
        time.sleep(random.uniform(0.5, 1.5))
        page.keyboard.press("Control+Enter")
        page.wait_for_timeout(random.uniform(2000, 4000))
        browser.close()

# ─── LIKING FUNCTION ───

def like_random_posts(max_likes=None, stop_time=None):
    continuous = stop_time is not None
    if not continuous and max_likes is None:
        max_likes = random.randint(3, 7)
    print(f"❤️ Starting like session{' until ' + str(stop_time) if continuous else f' of up to {max_likes} tweets'}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(storage_state="session.json")
        page = ctx.new_page()
        page.goto("https://twitter.com/home")
        page.wait_for_selector('article', timeout=15000)

        liked = 0
        scrolls = 0
        consecutive_liked = 0

        while True:
            if continuous and datetime.now() >= stop_time:
                print("⏹️ Stop time reached, pausing likes.")
                break
            if not continuous and liked >= max_likes:
                break
            if not continuous and scrolls >= 7:
                break

            time.sleep(random.uniform(0, 2))
            # 80% chance to scroll down, 20% up
            scroll_direction = "down" if random.random() < 0.95 else "up"
            scroll_amount = random.randint(600, 2200)
            page.mouse.wheel(0, scroll_amount if scroll_direction == "down" else -scroll_amount)
            print(f"🔁️ Scrolled {scroll_direction} by {scroll_amount}px")
            scrolls += 1

            time.sleep(random.uniform(1, 3))
            articles = page.locator("article")
            total_articles = articles.count()
            for i in range(min(total_articles, 10)):
                # --- CLICK 'SHOW MORE POSTS' IF VISIBLE ---
                show_more_btn = page.locator("xpath=//span[starts-with(text(), 'Show') and contains(text(), 'posts')]")
                if show_more_btn.is_visible():
                    print("🔄 Clicking 'Show more posts' to load new tweets...")
                    show_more_btn.click()
                    time.sleep(random.uniform(1, 2))
                    # Scroll to top after clicking
                    page.evaluate("window.scrollTo(0, 0);")
                    print("⬆️ Scrolled to top after clicking 'Show more posts'.")
                    break  # Break the article loop to refresh articles
                # --- END CLICK ---
                article = articles.nth(i)
                try:
                    tweet_text = article.inner_text().strip()
                    # Extract usernames from href attributes
                    user_links = article.locator("a[role=link][href^='/']")
                    usernames = []
                    for link in user_links.all():
                        href = link.get_attribute("href")
                        if href and href.startswith("/") and len(href) > 1 and not href.startswith("/i/"):
                            username = href[1:].split("/")[0]
                            if username not in usernames:
                                usernames.append(username)
                    print(f"Extracted usernames: {usernames}")
                    # Extract tweet author (from the first user link, usually the author)
                    author_username = None
                    if user_links.count() > 0:
                        first_href = user_links.nth(0).get_attribute("href")
                        if first_href and first_href.startswith("/") and len(first_href) > 1 and not first_href.startswith("/i/"):
                            author_username = first_href[1:].split("/")[0]
                    # Lowercase for comparison
                    own_username_lower = OWN_USERNAME.lower()
                    usernames_lower = [u.lower() for u in usernames]
                    # Skip if own username is present in any username, is the author, or mentioned in text
                    if (
                        own_username_lower in usernames_lower
                        or (author_username and own_username_lower == author_username.lower())
                        or (f"@{own_username_lower}" in tweet_text.lower())
                    ):
                        print(f"🚫 Skipping post involving own username: {usernames} (author: {author_username})")
                        continue
                    # Skip ignored users
                    if any(u in IGNORE_USERNAMES for u in usernames):
                        print(f"🚫 Skipping post involving ignored user(s): {usernames}")
                        continue
                    # Skip replies to your own tweets (self-reply threads)
                    if (
                        (own_username_lower in usernames_lower or (author_username and own_username_lower == author_username.lower()))
                        and len(usernames) > 1
                    ):
                        print("↪️ Skipping self-reply thread")
                        continue

                    if "Ad" in tweet_text:
                        print("🚫 Skipping ad post")
                        continue

                    if article.locator("[data-testid='placementTracking']").count() > 0:
                        print("🚫 Skipping promoted content")
                        continue

                    if article.locator("video").count() > 0 or (article.locator("img").count() > 2 and len(tweet_text) < 10):
                        print("📵 Skipping video/image-only post")
                        continue

                    if has_been_replied(tweet_text):
                        print("🔀 Already replied. Skipping.")
                        continue

                    like_btn = article.locator('button[data-testid="like"]')
                    aria_label = like_btn.get_attribute("aria-label")
                    if aria_label and "Liked" in aria_label:
                        consecutive_liked += 1
                        if consecutive_liked >= 3:
                            print("⬇️ 3 consecutive liked posts found, scrolling down...")
                            page.mouse.wheel(0, random.randint(800, 1400))
                            consecutive_liked = 0
                            break
                        continue
                    else:
                        consecutive_liked = 0

                    if continuous and datetime.now() >= stop_time:
                        break
                    if not continuous and liked >= max_likes:
                        break

                    # Only like if the post is engaging (model verification)
                    if not is_worth_engaging(tweet_text):
                        print(f"❌ Skipping like: tweet not engaging (model check failed)")
                        continue

                    # --- Extract engagement metrics ---
                    try:
                        likes = int(article.locator("[data-testid='like']").inner_text().replace(",", "") or 0)
                    except Exception:
                        likes = 0
                    try:
                        replies = int(article.locator("[data-testid='reply']").inner_text().replace(",", "") or 0)
                    except Exception:
                        replies = 0
                    try:
                        views = int(article.locator("[data-testid*='view']").inner_text().replace(",", "") or 0)
                    except Exception:
                        views = 0
                    # Try to get image URL (first image if present)
                    image_url = None
                    img_locator = article.locator("img[data-testid='tweetPhoto']")
                    if img_locator.count() > 0:
                        image_url = img_locator.nth(0).get_attribute("src")
                    # Compose tweet URL (best effort)
                    tweet_url = None
                    try:
                        tweet_link = article.locator("a[role=link][href*='/status/']")
                        if tweet_link.count() > 0:
                            tweet_url = 'https://twitter.com' + tweet_link.nth(0).get_attribute("href")
                    except Exception:
                        tweet_url = None
                    tweet_id = hashlib.md5((tweet_text + (author_username or "")).encode("utf-8")).hexdigest()
                    if is_high_engagement(likes, replies, views):
                        save_engaging_tweet(tweet_text, author_username, image_url, likes, replies, views, tweet_url, tweet_id)

                    time.sleep(random.uniform(3, 6))
                    like_btn.click(force=True, timeout=5000)
                    liked += 1
                    print(f"💖 Liked tweet #{liked}")

                    if random.random() < 0.9:
                        print(f"🎯 Attempting to reply to tweet #{liked}...")
                        reply_dialog_opened = False
                        try:
                            reply_btn = article.locator('[data-testid="reply"]')
                            if not reply_btn.is_visible():
                                print("📝 Reply button not visible, scrolling into view...")
                                article.scroll_into_view_if_needed(timeout=5000)
                                time.sleep(0.5)

                            print("📝 Clicking reply button...")
                            reply_btn.click(timeout=5000)
                            reply_box = page.locator('div[role="dialog"] div[role="textbox"]')
                            reply_box.wait_for(timeout=5000)
                            reply_dialog_opened = True
                            print("✅ Reply dialog opened successfully")

                            # --- Extract tweet info from dialog ---
                            dialog = page.locator('div[role="dialog"]')
                            # Extract tweet text
                            tweet_text_dialog = ""
                            tweet_text_locator = dialog.locator('div[data-testid="tweetText"]')
                            if tweet_text_locator.count() > 0:
                                tweet_text_dialog = tweet_text_locator.nth(0).inner_text().strip()
                            else:
                                # fallback: get the first p or span
                                fallback_text = dialog.locator('p')
                                if fallback_text.count() > 0:
                                    tweet_text_dialog = fallback_text.nth(0).inner_text().strip()
                                else:
                                    fallback_span = dialog.locator('span')
                                    if fallback_span.count() > 0:
                                        tweet_text_dialog = fallback_span.nth(0).inner_text().strip()
                            # Extract username
                            author_username = None
                            user_link = dialog.locator('a[role=link][href^="/"]')
                            if user_link.count() > 0:
                                first_href = user_link.nth(0).get_attribute("href")
                                if first_href and first_href.startswith("/") and len(first_href) > 1 and not first_href.startswith("/i/"):
                                    author_username = first_href[1:].split("/")[0]
                            # Extract all usernames in dialog (for ignore/own checks)
                            usernames = []
                            for link_idx in range(user_link.count()):
                                href = user_link.nth(link_idx).get_attribute("href")
                                if href and href.startswith("/") and len(href) > 1 and not href.startswith("/i/"):
                                    username = href[1:].split("/")[0]
                                    if username not in usernames:
                                        usernames.append(username)
                            print(f"[DIALOG] Extracted usernames: {usernames}")
                            print(f"[DIALOG] Extracted tweet text: {tweet_text_dialog}")
                            # Lowercase for comparison
                            own_username_lower = OWN_USERNAME.lower()
                            usernames_lower = [u.lower() for u in usernames]
                            if (
                                own_username_lower in usernames_lower
                                or (author_username and own_username_lower == author_username.lower())
                                or (f"@{own_username_lower}" in tweet_text_dialog.lower())
                            ):
                                print(f"🚫 Not replying to post involving own username: {usernames} (author: {author_username})")
                                # Close dialog before continuing
                                try:
                                    close_btn = page.locator('div[role="dialog"] [aria-label="Close"]')
                                    if close_btn.is_visible():
                                        close_btn.click()
                                        time.sleep(0.5)
                                except Exception as e:
                                    print(f"⚠️ Could not close reply dialog: {e}")
                                continue
                            if any(u in IGNORE_USERNAMES for u in usernames):
                                print(f"🚫 Not replying to post involving ignored user(s): {usernames}")
                                # Close dialog before continuing
                                try:
                                    close_btn = page.locator('div[role="dialog"] [aria-label="Close"]')
                                    if close_btn.is_visible():
                                        close_btn.click()
                                        time.sleep(0.5)
                                except Exception as e:
                                    print(f"⚠️ Could not close reply dialog: {e}")
                                continue

                            # --- Worthiness check using dialog tweet text ---
                            print(f"🔍 Checking worthiness for dialog tweet: {tweet_text_dialog[:80]}...")
                            if is_worth_engaging(tweet_text_dialog):
                                print(f"🟢 Engaging with tweet by @{author_username}: {tweet_text_dialog[:80]}...")
                                # --- IMAGE ANALYSIS WITH LLAVA ---
                                media_imgs = dialog.locator('img[data-testid="tweetPhoto"]')
                                img_count = media_imgs.count()
                                image_caption = None
                                # --- Extract engagement metrics for dialog ---
                                try:
                                    likes = int(dialog.locator("[data-testid='like']").inner_text().replace(",", "") or 0)
                                except Exception:
                                    likes = 0
                                try:
                                    replies = int(dialog.locator("[data-testid='reply']").inner_text().replace(",", "") or 0)
                                except Exception:
                                    replies = 0
                                try:
                                    views = int(dialog.locator("[data-testid*='view']").inner_text().replace(",", "") or 0)
                                except Exception:
                                    views = 0
                                image_url = None
                                if img_count > 0:
                                    image_url = media_imgs.nth(0).get_attribute("src")
                                tweet_url = None
                                try:
                                    tweet_link = dialog.locator("a[role=link][href*='/status/']")
                                    if tweet_link.count() > 0:
                                        tweet_url = 'https://twitter.com' + tweet_link.nth(0).get_attribute("href")
                                except Exception:
                                    tweet_url = None
                                tweet_id = hashlib.md5((tweet_text_dialog + (author_username or "")).encode("utf-8")).hexdigest()
                                if is_high_engagement(likes, replies, views):
                                    save_engaging_tweet(tweet_text_dialog, author_username, image_url, likes, replies, views, tweet_url, tweet_id)
                                try:
                                    reply_text = generate_reply(tweet_text_dialog, image_caption)
                                    print(f"💭 Replying to @{author_username}: {reply_text}")
                                    human_type(page, reply_box, reply_text)
                                    page.keyboard.press("Control+Enter")
                                    time.sleep(random.uniform(0, 1))
                                    mark_as_replied(tweet_text_dialog)
                                except Exception as e:
                                    print(f"⚠️ Failed to generate or send reply: {e}")
                                    # Close dialog if reply failed
                                    try:
                                        close_btn = page.locator('div[role="dialog"] [aria-label="Close"]')
                                        if close_btn.is_visible():
                                            close_btn.click()
                                            time.sleep(0.5)
                                    except Exception as close_e:
                                        print(f"⚠️ Could not close reply dialog after failed reply: {close_e}")
                            else:
                                print(f"❌ Not engaging with tweet by @{author_username}: {tweet_text_dialog[:80]}...")
                                try:
                                    close_btn = page.locator('div[role="dialog"] [aria-label="Close"]')
                                    if close_btn.is_visible():
                                        close_btn.click()
                                        time.sleep(0.5)
                                except Exception as e:
                                    print(f"⚠️ Could not close reply dialog: {e}")

                        except Exception as e:
                            print(f"⚠️ Failed to reply: {e}")
                            # If dialog was opened but something failed, try to close it
                            if reply_dialog_opened:
                                try:
                                    close_btn = page.locator('div[role="dialog"] [aria-label="Close"]')
                                    if close_btn.is_visible():
                                        close_btn.click()
                                        time.sleep(0.5)
                                        print("✅ Closed reply dialog after error")
                                except Exception as close_e:
                                    print(f"⚠️ Could not close reply dialog after error: {close_e}")
                                    # Try alternative close methods
                                    try:
                                        page.keyboard.press("Escape")
                                        time.sleep(0.5)
                                        print("✅ Closed reply dialog with Escape key")
                                    except Exception as escape_e:
                                        print(f"⚠️ Could not close dialog with Escape key: {escape_e}")

                except Exception:
                    continue

        browser.close()
    print(f"✅ Finished liking {liked} tweets.")

# ─── ENGAGEMENT ONLY ROUTINE ───

def run_engagement_only():
    print("🔁 Bot is now running in engagement-only mode (likes and replies only)...")
    
    while True:
        now_time = datetime.now().time()
        if is_within_posting_window(now_time):
            print("🕒 Inside engagement window — starting like and reply session.")
            # Run continuous liking and replying during posting windows
            like_random_posts(stop_time=datetime.now() + timedelta(hours=2))  # 2 hour sessions
            print("✅ Completed engagement session.")
            
            # Wait until next window
            today = datetime.now().date()
            future = [datetime.combine(today, start) for start, _ in WINDOWS if datetime.combine(today, start).time() > now_time]
            if not future:
                tomorrow = today + timedelta(days=1)
                future = [datetime.combine(tomorrow, WINDOWS[0][0])]
            next_start = min(future)
            wait_secs = (next_start - datetime.now()).total_seconds()
            print(f"⏳ Sleeping {int(wait_secs)}s until next window at {next_start.time()}")
            time.sleep(wait_secs)
        else:
            print("🌙 Outside engagement window — idle until next window.")
            today = datetime.now().date()
            future = [datetime.combine(today, start) for start, _ in WINDOWS if datetime.combine(today, start).time() > now_time]
            if not future:
                tomorrow = today + timedelta(days=1)
                future = [datetime.combine(tomorrow, WINDOWS[0][0])]
            next_start = min(future)
            wait_secs = (next_start - datetime.now()).total_seconds()
            print(f"⏳ Sleeping {int(wait_secs)}s until next window at {next_start.time()}")
            time.sleep(wait_secs)

# ─── CONTINUOUS ENGAGEMENT MODE ───

def run_continuous_engagement():
    print("🔁 Bot is now running in continuous engagement mode (likes and replies only)...")
    
    while True:
        print("🕒 Starting continuous engagement session...")
        # Run continuous liking and replying without time restrictions
        like_random_posts(max_likes=random.randint(5, 15))  # Random number of likes per session
        
        # Wait between sessions
        wait_time = random.randint(300, 900)  # 5-15 minutes between sessions
        print(f"⏳ Sleeping {wait_time}s before next engagement session...")
        time.sleep(wait_time)

if __name__ == "__main__":
    # Choose one of these modes:
    # run_engagement_only()  # Only engage during posting windows
    run_continuous_engagement()  # Engage continuously throughout the day
