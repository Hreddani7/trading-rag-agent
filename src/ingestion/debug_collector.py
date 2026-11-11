import os
import feedparser
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import pymupdf
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
DOWNLOAD_FOLDER = "data/raw"
NLG_TARGET_PAGE = "https://nlg.news/haiwaiyanbao.html"
NLG_COOKIES = {
    cookie_name: os.environ[env_name]
    for cookie_name, env_name in {
        "PHPSESSID": "NLG_PHPSESSID",
        "token": "NLG_TOKEN",
        "uid": "NLG_UID",
    }.items()
    if os.environ.get(env_name)
}

def debug_pdfs():
    print("\n--- 🛑 DEBUGGING LOCAL PDFs ---")
    if not os.path.exists(DOWNLOAD_FOLDER):
        print(f"❌ Folder {DOWNLOAD_FOLDER} does not exist! Creating it now...")
        os.makedirs(DOWNLOAD_FOLDER)
        return

    files = [f for f in os.listdir(DOWNLOAD_FOLDER) if f.endswith('.pdf')]
    print(f"Found {len(files)} PDF files in {DOWNLOAD_FOLDER}.")
    
    for f in files:
        filepath = os.path.join(DOWNLOAD_FOLDER, f)
        mtime = os.path.getmtime(filepath)
        file_date = datetime.fromtimestamp(mtime)
        days_old = (datetime.now() - file_date).days
        print(f"  📄 {f} | Age: {days_old} days old")
        
        if days_old <= 7:
            doc = pymupdf.open(filepath)
            text = doc[0].get_text()[:200].replace('\n', ' ')
            print(f"     📖 First 200 chars: {text}...")

def debug_rss():
    print("\n--- 🌐 DEBUGGING PUBLIC RSS ---")
    RSS_FEEDS = {
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "Investing.com": "https://news.google.com/rss/search?q=site%3Ainvesting.com%2Fnews+%28%22S%26P+500%22+OR+Nasdaq+OR+%22Wall+Street%22+OR+Fed%29+when%3A7d&hl=en-US&gl=US&ceid=US:en",
        "Reuters": "https://news.google.com/rss/search?q=site%3Areuters.com+%28%22S%26P+500%22+OR+Nasdaq+OR+%22Wall+Street%22+OR+%22US+stocks%22%29+when%3A7d&hl=en-US&gl=US&ceid=US:en",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            print(f"  {source}: Found {len(feed.entries)} total entries.")
            if feed.entries:
                print(f"    Latest headline: '{feed.entries[0].get('title', '')}'")
        except requests.RequestException as e:
            print(f"  {source}: Failed to fetch feed: {e}")

def debug_private_site():
    print("\n--- 🔒 DEBUGGING PRIVATE SITE ---")
    missing_cookies = [
        name
        for name in ("NLG_PHPSESSID", "NLG_TOKEN", "NLG_UID")
        if not os.environ.get(name)
    ]
    if missing_cookies:
        print(f"  Missing NLG login cookies: {', '.join(missing_cookies)}")
    session = requests.Session()
    session.cookies.update(NLG_COOKIES)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        print(f"  Connecting to {NLG_TARGET_PAGE}...")
        response = session.get(NLG_TARGET_PAGE, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"  Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [
            card.select_one("h3.article-title a[href]")
            for card in soup.select("div.article-item")
        ]
        links = [link for link in links if link]
        print(f"  Found {len(links)} article listings on the page.")
        
        print("  Sample article listings:")
        for link in links[:5]:
            print(f"    - Text: '{link.get_text(strip=True)[:80]}' | URL: {link['href'][:100]}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    debug_pdfs()
    debug_rss()
    debug_private_site()