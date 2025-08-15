from playwright.sync_api import sync_playwright

def save_session():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://twitter.com/login")

        print("🔐 Please log in manually.")
        print("✅ Once logged in, close the browser window to save the session.")

        page.wait_for_timeout(60000)  # 60 seconds to login
        context.storage_state(path="session.json")
        print("💾 Session saved as session.json")

        browser.close()

if __name__ == "__main__":
    save_session() 