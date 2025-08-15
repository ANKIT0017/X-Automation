import os
import random
import time

from langchain import LLMChain, PromptTemplate
from langchain.llms import Ollama  # or `from langchain.llms import OpenAI`
from playwright.sync_api import sync_playwright

# ---- 1. Configure your LLM here ----
def get_llm():
    # Local Ollama model:
    return Ollama(
        model="llama3:latest",         # replace with the Ollama model you’ve pulled
        verbose=False,
        base_url="http://127.0.0.1:11434"  # default Ollama HTTP endpoint
    )

    # Or, to use OpenAI instead, comment out above and uncomment:
    # return OpenAI(temperature=0.7, model_name="gpt-4")


# ---- 2. Build a simple LangChain prompt ----
prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You’re a witty, concise social media manager. "
        "Write a single tweet (≤280 chars) on this topic: dont use hashtags and dont reply this message.\n\n"
        "Topic: {topic}\n\nTweet:"
    ),
)

def generate_tweet(topic: str) -> str:
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    tweet = chain.run(topic).strip()
    return tweet[:280]


# ---- 3. Humanized Playwright posting (same as before) ----
def human_type(page, selector, text):
    page.click(selector)
    for char in text:
        page.keyboard.press(char)
        page.wait_for_timeout(random.randint(50, 200))

def tweet_via_playwright(content: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(storage_state="session.json")
        page = ctx.new_page()

        page.goto("https://twitter.com/home")
        page.wait_for_selector('div[role="textbox"]', timeout=15000)

        box = page.locator('div[role="textbox"]')
        box.hover()
        page.wait_for_timeout(random.randint(300, 800))

        human_type(page, 'div[role="textbox"]', content)

        time.sleep(random.uniform(0.5, 1.5))
        page.keyboard.press("Control+Enter")
        print(f"✅ Tweet posted: {content}")

        page.wait_for_timeout(random.randint(2000, 4000))
        browser.close()


# ---- 4. Main flow ----
if __name__ == "__main__":
    topic = "genz style post about comparison between 1700s and 1800s in respect of tech"
    print("🤖 Generating tweet with LangChain + Ollama...")
    tweet_text = generate_tweet(topic)
    print("\n🔍 Preview:\n", tweet_text, "\n")
    
    tweet_via_playwright(tweet_text)
    
