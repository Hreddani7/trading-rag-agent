import os
import re
import requests
from bs4 import BeautifulSoup
import feedparser
import trafilatura
import pymupdf
import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import sys
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ==========================================
# ⚙️ CONFIGURATION 
# ==========================================
MAX_DAYS_OLD = 7
MAX_ARTICLES = 8
MAX_PRIVATE_ARTICLES_TO_FETCH = 40

# 🧠 PRIVATE WEBSITE SETUP
NLG_BASE_URL = os.environ.get("NLG_BASE_URL", "https://nlg.news").rstrip("/")
NLG_TARGET_PAGES = (
    f"{NLG_BASE_URL}/shishishichangzixun.html",
    f"{NLG_BASE_URL}/xinwenyingwenyuanwen.html",
    f"{NLG_BASE_URL}/haiwaiyanbao.html",
)
NLG_COOKIES = {
    cookie_name: os.environ[env_name]
    for cookie_name, env_name in (
        ("PHPSESSID", "NLG_PHPSESSID"),
        ("token", "NLG_TOKEN"),
        ("uid", "NLG_UID"),
    )
    if os.environ.get(env_name)
}

# 🧠 PUBLIC RSS FEEDS
RSS_FEEDS = {
    "CNBC_Finance": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Investing_Com": "https://news.google.com/rss/search?q=site%3Ainvesting.com%2Fnews+%28%22S%26P+500%22+OR+Nasdaq+OR+%22Wall+Street%22+OR+Fed%29+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
    "Reuters_Finance": "https://news.google.com/rss/search?q=site%3Areuters.com+%28%22S%26P+500%22+OR+Nasdaq+OR+%22Wall+Street%22+OR+%22US+stocks%22%29+when%3A7d&hl=en-US&gl=US&ceid=US:en"
}

DOWNLOAD_FOLDER = "data/raw"
DB_DIR = "data/chroma_db"

US_MARKET_ANCHORS = (
    ("S&P 500", re.compile(r"\bS\s*&\s*P\s*500\b", re.IGNORECASE)),
    ("Nasdaq", re.compile(r"\bNasdaq(?:-\d{1,2})?\b", re.IGNORECASE)),
    ("Dow Jones", re.compile(r"\bDow Jones\b", re.IGNORECASE)),
    ("NYSE", re.compile(r"\bNYSE\b", re.IGNORECASE)),
    ("Wall Street", re.compile(r"\bWall Street\b", re.IGNORECASE)),
    ("US stocks/equities", re.compile(r"\b(?:U\.?S\.?|American)\s+(?:stocks?|equities)\b", re.IGNORECASE)),
    ("US markets", re.compile(r"\b(?:U\.?S\.?|American)\s+markets?\b", re.IGNORECASE)),
)

MARKET_ANGLES = (
    ("price movement", re.compile(r"\b(?:rall(?:y|ied|ies)|surge[sd]?|soar(?:ed|s)?|jump(?:ed|s)?|gain(?:ed|s)?|rise[sd]?|fell|drop(?:ped|s)?|slid(?:e[sd]?)?|declin(?:e|ed|es)|sell[\s-]?off|wipeout|worst month|loss(?:es)?|higher|lower|advance[sd]?)\b", re.IGNORECASE)),
    ("earnings", re.compile(r"\b(?:earnings|revenue|profits?|quarterly results?)\b", re.IGNORECASE)),
    ("valuation", re.compile(r"\b(?:valuation|valued|overvalued|undervalued|price-to-earnings|P/?E ratio)\b", re.IGNORECASE)),
    ("sector performance", re.compile(r"\b(?:sector performance|sectors? (?:outperform|underperform|rallied|fell|rose)|tech stocks?|bank stocks?|energy stocks?)\b", re.IGNORECASE)),
    ("Fed policy", re.compile(r"\b(?:Fed(?:eral Reserve)?|rate cuts?|interest rates?)\b", re.IGNORECASE)),
    ("inflation", re.compile(r"\b(?:inflation|CPI|consumer prices?)\b", re.IGNORECASE)),
    ("jobs data", re.compile(r"\b(?:jobs report|payrolls?|unemployment|employment report|labor market data)\b", re.IGNORECASE)),
    ("Treasury yields", re.compile(r"\b(?:Treasury yields?|Treasuries|10-year yield)\b", re.IGNORECASE)),
    ("market outlook", re.compile(r"\b(?:market outlook|market forecast|investor outlook|market expectations)\b", re.IGNORECASE)),
)

EXPLAINER_PATTERN = re.compile(
    r"\b(?:what is|how .* works?|explainer|explained|guide to|market basics|primer)\b",
    re.IGNORECASE,
)
ACCESS_GATE_PATTERN = re.compile(
    r"温馨提示|请先登录|登录后(?:查看|阅读)|仅限会员|会员专享|VIP专享|购买会员|"
    r"login to view|members only|subscription required|subscribe to continue",
    re.IGNORECASE,
)

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def parse_published_date(value):
    if not value:
        return None
    if hasattr(value, "tm_year"):
        return datetime(*value[:6], tzinfo=timezone.utc)
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    date_text = str(value).strip()
    chinese_date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_text)
    if chinese_date:
        year, month, day = map(int, chinese_date.groups())
        return datetime(year, month, day, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(date_text)
        except (TypeError, ValueError, OverflowError):
            return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

def article_relevance(title, text, published_at, *, now=None):
    combined = f"{title}\n{text}"
    anchors = [
        (label, match.group(0))
        for label, pattern in US_MARKET_ANCHORS
        if (match := pattern.search(combined))
    ]
    angles = [
        label
        for label, pattern in MARKET_ANGLES
        if pattern.search(combined)
    ]
    if not anchors or not angles:
        return None

    is_explainer = bool(EXPLAINER_PATTERN.search(title))
    published_at = parse_published_date(published_at)
    current_time = now or datetime.now(timezone.utc)
    if published_at is None:
        if not is_explainer:
            return None
    elif current_time - published_at > timedelta(days=MAX_DAYS_OLD) and not is_explainer:
        return None

    score = 5 if any(
        pattern.search(title) for _, pattern in US_MARKET_ANCHORS
    ) and any(pattern.search(title) for _, pattern in MARKET_ANGLES) else 4
    date_label = published_at.date().isoformat() if published_at else "Undated"
    angle = f"The article links {anchors[0][0]} to {angles[0]}."
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", BeautifulSoup(text, "html.parser").get_text(" ", strip=True))
        if sentence.strip()
    ][:3]
    if not sentences:
        return None

    formatted = (
        f"Title: {title}\n"
        f"Date: {date_label}\n"
        f"US angle: {angle}\n"
        "Summary:\n"
        + "\n".join(f"- {sentence}" for sentence in sentences)
        + f"\nScore: {score}/5"
    )
    return {
        "text": formatted,
        "date": date_label,
        "published_at": published_at.isoformat() if published_at else "",
        "score": score,
        "us_angle": angle,
    }

def _article_document(title, text, published_at, source, article_url, document_type):
    relevance = article_relevance(title, text, published_at)
    if not relevance:
        return None
    return Document(
        page_content=relevance["text"],
        metadata={
            "id": get_hash(relevance["text"]),
            "source": source,
            "title": title,
            "type": document_type,
            "date": relevance["date"],
            "published_at": relevance["published_at"],
            "relevance_score": relevance["score"],
            "us_angle": relevance["us_angle"],
            "url": article_url,
        },
    )

def _sort_and_limit_articles(docs):
    unique_docs = {doc.metadata["id"]: doc for doc in docs}
    ordered_docs = sorted(
        unique_docs.values(),
        key=lambda doc: doc.metadata.get("published_at", ""),
        reverse=True,
    )
    return ordered_docs[:MAX_ARTICLES]

# ==========================================
# 📥 SCRAPING LOGIC
# ==========================================
def scrape_private_website():
    print(f"\n🔐 Scraping Private Site: {NLG_BASE_URL}")
    required_cookies = ("NLG_PHPSESSID", "NLG_TOKEN", "NLG_UID")
    missing_cookies = [name for name in required_cookies if not os.environ.get(name)]
    if missing_cookies:
        print(
            "   ⚠️ Missing NLG login cookies "
            f"({', '.join(missing_cookies)}). Add them to .env to access member-only article text."
        )
        return []

    session = requests.Session()
    session.cookies.update(NLG_COOKIES)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    listings = {}
    for page_url in NLG_TARGET_PAGES:
        try:
            response = session.get(page_url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for card in soup.select("div.article-item"):
                title_link = card.select_one("h3.article-title a[href]")
                if not title_link:
                    continue
                article_url = urljoin(response.url, title_link["href"])
                parsed_url = urlparse(article_url)
                if parsed_url.netloc != urlparse(NLG_BASE_URL).netloc:
                    continue
                if not re.search(r"/\d+\.html$", parsed_url.path):
                    continue

                title = title_link.get_text(" ", strip=True)
                date_element = card.select_one('[itemprop="date"]')
                published_at = parse_published_date(
                    date_element.get_text(" ", strip=True) if date_element else None
                )
                listings[article_url] = (title, published_at)
        except requests.RequestException as e:
            print(f"   ⚠️ Could not fetch private listing {page_url}: {e}")

    now = datetime.now(timezone.utc)
    candidates = [
        (url, title, published_at)
        for url, (title, published_at) in listings.items()
        if published_at
        and (
            now - published_at <= timedelta(days=MAX_DAYS_OLD)
            or EXPLAINER_PATTERN.search(title)
        )
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)

    docs = []
    blocked_count = 0
    filtered_count = 0
    for article_url, title, listed_date in candidates[:MAX_PRIVATE_ARTICLES_TO_FETCH]:
        try:
            response = session.get(article_url, headers=headers, timeout=15)
            response.raise_for_status()
            metadata = trafilatura.extract_metadata(response.text)
            text = trafilatura.extract(response.text, url=response.url)
            page_title = BeautifulSoup(response.text, "html.parser").title
            gate_text = page_title.get_text(" ", strip=True) if page_title else ""
            if ACCESS_GATE_PATTERN.search(f"{gate_text}\n{text or ''}"):
                blocked_count += 1
                break
            if not text:
                blocked_count += 1
                continue

            title = metadata.title or title
            published_at = parse_published_date(metadata.date) or listed_date
            doc = _article_document(
                title,
                text,
                published_at,
                "NLG Private",
                response.url,
                "private_research",
            )
            if doc:
                docs.append(doc)
            else:
                filtered_count += 1
        except requests.RequestException as e:
            print(f"   ⚠️ Could not fetch private article '{title}': {e}")
        except Exception as e:
            print(f"   ⚠️ Could not process private article '{title}': {e}")

    docs = _sort_and_limit_articles(docs)
    if blocked_count:
        print(
            "   ⚠️ The site did not return article text. The login cookies may be "
            "expired/invalid, or the account may not have VIP access."
        )
    print(
        f"   ✅ Kept {len(docs)} recent US-market private articles; "
        f"checked {min(len(candidates), MAX_PRIVATE_ARTICLES_TO_FETCH)}, "
        f"filtered {filtered_count} off-topic articles."
    )
    return docs

def process_local_pdfs():
    print("\n📄 Reading local PDFs in data/raw...")
    docs = []
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(DOWNLOAD_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"   No PDFs found in {DOWNLOAD_FOLDER}.")
        return docs

    for filename in pdf_files:
        filepath = os.path.join(DOWNLOAD_FOLDER, filename)
        try:
            pdf_document = pymupdf.open(filepath)
            text = "".join(page.get_text() for page in pdf_document)
            
            docs.append(Document(
                page_content=text,
                metadata={"id": get_hash(text), "source": "PDF Report", "title": filename, "type": "research_report"}
            ))
            print(f"   ✅ Processed PDF: {filename}")
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
            
    return docs

def process_public_rss():
    print("\n🌐 Fetching Public RSS...")
    docs_by_source = {}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    
    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            if not feed.entries:
                print(f"   ⚠️ {source}: Feed returned no articles (HTTP {response.status_code}).")
                continue

            source_docs = []
            skipped_old_or_irrelevant = 0
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                published_at = parse_published_date(
                    entry.get("published_parsed") or entry.get("updated_parsed")
                )
                summary = BeautifulSoup(
                    entry.get("summary", ""), "html.parser"
                ).get_text(" ", strip=True)

                # Avoid downloading pages that already fail the strict US-market/date rules.
                if not article_relevance(
                    title, summary or title, published_at
                ):
                    skipped_old_or_irrelevant += 1
                    continue

                try:
                    text = ""
                    article_url = entry.get("link")
                    if article_url:
                        try:
                            article_response = requests.get(
                                article_url, headers=headers, timeout=10
                            )
                            article_response.raise_for_status()
                            text = trafilatura.extract(
                                article_response.text, url=article_response.url
                            )
                        except requests.RequestException as e:
                            print(
                                f"   ⚠️ {source}: Could not fetch full article "
                                f"'{title}', trying RSS summary: {e}"
                            )

                    if not text:
                        text = summary or title

                    doc = _article_document(
                        title,
                        text,
                        published_at,
                        source,
                        article_url or "",
                        "news_article",
                    )
                    if doc:
                        source_docs.append(doc)
                except Exception as e:
                    print(f"   ⚠️ {source}: Could not process '{title}': {e}")

            docs_by_source[source] = source_docs
            print(
                f"   ✅ {source}: Kept {len(source_docs)} US-market articles; "
                f"filtered {skipped_old_or_irrelevant} others."
            )
        except Exception as e:
            print(f"   ❌ Failed to fetch {source}: {e}")

    return _sort_and_limit_articles(
        [doc for source_docs in docs_by_source.values() for doc in source_docs]
    )

# ==========================================
# 🧠 MAIN PIPELINE
# ==========================================
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("🚀 Starting Data Ingestion Pipeline...")
    
    all_docs = _sort_and_limit_articles(
        scrape_private_website() + process_local_pdfs() + process_public_rss()
    )
    
    print(f"\n📊 Total new documents gathered: {len(all_docs)}")
    if not all_docs:
        print("⚠️ No documents found. Exiting.")
        return

    print("✂️ Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    print("🔌 Connecting to local Ollama (nomic-embed-text) and ChromaDB...")
    # NOTE: Chroma creates the DB folder automatically. You don't need a separate server running!
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    chunk_ids = [get_hash(chunk.page_content) for chunk in chunks]
    
    print(f"💾 Saving {len(chunks)} chunks to database...")
    # Add documents (Chroma automatically handles updating if IDs already exist)
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)
    
    print(f"🎉 Success! Database now holds {vectorstore._collection.count()} chunks.")

if __name__ == "__main__":
    main()