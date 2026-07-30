"""
Dataset Retriever Module
"""

import io
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, quote, parse_qs

import httpx
import pandas as pd
from bs4 import BeautifulSoup

# Suppress the duckduckgo_search rename warning to keep your console clean
warnings.filterwarnings("ignore", module="duckduckgo_search")
from ddgs import DDGS
from rapidfuzz import fuzz

# -----------------------------
# Configuration & Constants
# -----------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DOWNLOAD_DIR = Path("datasets")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_SCORES: Dict[str, int] = {
    "data.gov.in": 100,
    "mospi.gov.in": 100,
    "rbi.org.in": 100,
    "censusindia.gov.in": 100,
    "worldbank.org": 85,
    "who.int": 85,
    "kaggle.com": 60,
    "github.com": 10, # Drastically lowered so benchmark logs stop hijacking your agent
    "gist.github.com": 0
}

FILE_SCORES: Dict[str, int] = {
    ".csv": 100,
    ".parquet": 95,
    ".xlsx": 90,
    ".xls": 85,
    ".json": 80,
    ".html": 5,   
    ".pdf": 5,
}

DOWNLOADABLE: tuple[str, ...] = (".csv", ".parquet", ".xlsx", ".xls", ".json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
}

# -----------------------------
# Scoring Mechanisms
# -----------------------------

def score_domain(url: str) -> int:
    host = urlparse(url).netloc.lower()
    for domain, score in DOMAIN_SCORES.items():
        if domain in host:
            return score
    return 0

def score_file(url: str) -> int:
    lower_url = url.lower()
    for ext, score in FILE_SCORES.items():
        if lower_url.endswith(ext):
            return score
    return 0

def keyword_bonus(text: str, keywords: List[str]) -> int:
    score = 0
    text_lower = text.lower()
    for keyword_phrase in keywords:
        for word in keyword_phrase.split():
            if word.lower() in text_lower:
                score += 10
    return score

def score_result(result: Dict[str, str], planner: Dict[str, Any]) -> int:
    """Scores a search result based on domain authority and keyword relevance."""
    url = result.get("href", "").lower()
    title = result.get("title", "").lower()
    keywords = planner.get("search_keywords", [])
    
    score = 0
    
    # 1. Base Domain Authority
    for domain, weight in DOMAIN_SCORES.items():
        if domain in url:
            score += weight
            break

    # 2. Keyword Relevance
    for keyword_phrase in keywords:
        for word in keyword_phrase.split():
            word = word.lower()
            if len(word) > 3:
                if word in title:
                    score += 15
                if word in url:
                    score += 5
                    
    # 3. THE FIX: The "Named Source" Override
    # If the planner specifically asks for a source (like RBI or MOSPI), 
    # and that source's official domain is in the URL, give it a massive boost.
    # 3. THE ELEGANT FIX: Dynamic Domain Override
    expected_domain = planner.get("expected_domain", "").lower()
    if expected_domain and expected_domain in url:
        score += 500

    return score

# -----------------------------
# Web & Download Helpers
# -----------------------------

def to_raw_url(url: str) -> str:
    """Converts GitHub and Hugging Face web URLs to their raw data equivalents."""
    lower_url = url.lower()
    
    # Handle GitHub URLs
    if "github.com" in lower_url and "/blob/" in lower_url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
        return url
        
    # Handle Hugging Face URLs
    if "huggingface.co" in lower_url and "/blob/" in lower_url:
        url = url.replace("/blob/", "/resolve/")
        return url
        
    return url

def download(url: str) -> Path:
    """Downloads a file from a URL and saves it to the datasets directory."""
    # Instantly convert GitHub and Hugging Face UI links to raw file streams
    url = to_raw_url(url)
    
    filename = url.split("/")[-1].split("?")[0]
    if not filename or not filename.lower().endswith(DOWNLOADABLE):
        filename = "dataset.csv"

    filepath = DOWNLOAD_DIR / filename
    logger.info(f"Downloading: {url} -> {filepath}")

    with httpx.Client(follow_redirects=True, timeout=60.0, headers=HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

    return filepath

def find_download_links(page_url: str, planner: Dict[str, Any]) -> List[str]:
    """Scrapes a webpage to find direct links, scoring them by surrounding text context."""
    logger.info(f"Scanning webpage for files: {page_url}")
    
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0, headers=HEADERS) as client:
            response = client.get(page_url)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch {page_url}: {e}")
        return []

    soup = BeautifulSoup(html, "lxml")
    files = []
    
    # Grab the concept we are actually looking for
   # Grab the concepts we are actually looking for
    dataset_name = planner.get("dataset_name", "")
    requested_fields = " ".join(planner.get("requested_fields", []))
    
    # NEW: Extract the values from the filters (e.g., "2021-22 current prices")
    filters = " ".join(str(v) for v in planner.get("filters", {}).values())
    
    # Combine them for a hyper-specific search target
    target_concept = f"{dataset_name} {requested_fields} {filters}".lower().strip()

    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        
        if href.lower().endswith(DOWNLOADABLE):
            base_score = score_file(href)
            
            # THE FIX: Don't just read the link. Read the entire row or paragraph it sits in!
            parent_element = a.find_parent(['tr', 'li', 'p', 'div'])
            
            if parent_element:
                link_text = parent_element.get_text(separator=" ", strip=True).lower()
            else:
                link_text = a.get_text(separator=" ", strip=True).lower()
            
            # Fuzzy match the surrounding text against our target dataset name
            relevance_score = fuzz.token_set_ratio(target_concept, link_text)
            
            # If the link has no text (like a download icon), check the URL string itself
            if not link_text:
                relevance_score = fuzz.token_set_ratio(target_concept, href.lower())
                
            total_score = base_score + relevance_score
            files.append((href, total_score, link_text))

    # Sort files by highest score (relevance + extension)
    files.sort(key=lambda x: x[1], reverse=True)
    
    if files:
        logger.info(f"Target Concept: '{target_concept}'")
        logger.info("Top 5 Found Files on Page:")
        for href, score, text in files[:5]:
            # Truncate text for clean logging
            safe_text = text[:60].replace('\n', ' ')
            logger.info(f"  -> Score: {score} | Context: '{safe_text}...' | File: {href.split('/')[-1]}")

    return [f[0] for f in files]

def extract_table_from_html(url: str) -> Optional[Path]:
    """Scrapes the largest HTML table from a webpage and saves it as a CSV."""
    logger.info(f"Attempting to extract HTML tables from: {url}")
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0, headers=HEADERS) as client:
            response = client.get(url)
            response.raise_for_status()
            
        # FIX: Pandas 2.0+ strict requirement for StringIO
        html_io = io.StringIO(response.text)
        tables = pd.read_html(html_io)
        
        if not tables:
            return None
            
        # Get the table with the most cells
        largest_table = max(tables, key=lambda df: df.shape[0] * df.shape[1])
        
        if largest_table.shape[0] < 3 or largest_table.shape[1] < 2:
            return None
            
        filename = DOWNLOAD_DIR / "extracted_html_table.csv"
        largest_table.to_csv(filename, index=False)
        logger.info(f"Successfully extracted table to: {filename}")
        
        return filename
        
    except Exception as e:
        logger.debug(f"HTML table extraction failed for {url}: {e}")
        return None

def fallback_search(query: str) -> List[Dict[str, Any]]:
    logger.info("Triggering fallback HTML search...")
    safe_query = quote(query)
    url = f"https://html.duckduckgo.com/html/?q={safe_query}"
    
    try:
        with httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
            res = client.get(url)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "lxml")
            
            results = []
            for result in soup.find_all("div", class_="result"):
                title_tag = result.find("a", class_="result__url")
                snippet_tag = result.find("a", class_="result__snippet")
                
                if title_tag:
                    href = title_tag.get("href", "")
                    if "uddg=" in href:
                        parsed_url = urlparse(href)
                        href = parse_qs(parsed_url.query).get("uddg", [href])[0]
                        
                    results.append({
                        "title": title_tag.text.strip(),
                        "href": href,
                        "body": snippet_tag.text.strip() if snippet_tag else ""
                    })
            return results[:10]
    except Exception as e:
        logger.error(f"Fallback search failed: {e}")
        return []

# -----------------------------
# Main Retrieval Logic
# -----------------------------

def retrieve_dataset(planner: Dict[str, Any]) -> Optional[Path]:

    dataset_url = planner.get("dataset_link", "")
    if dataset_url:
        logger.info(f"Direct link found in planner. Bypassing search: {dataset_url}")
        try:
            return download(dataset_url)
        except Exception as e:
            logger.error(f"Failed to download direct dataset link {dataset_url}: {e}")
    search_keywords = planner.get("search_keywords", [])
    if not search_keywords:
        logger.error("No search keywords provided in planner.")
        return None

    query = search_keywords[0]
    logger.info(f"Initiating search for: '{query}'")

    results = []
    
    # PASS 1: Standard Search
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=10))
    except Exception as e:
        logger.warning(f"Primary search API failed: {e}")

    # PASS 2: "The Cheat Code" - Force DuckDuckGo to only return files
    file_query = f"{query} ext:csv OR ext:xlsx"
    logger.info(f"Initiating secondary file-only search for: '{file_query}'")
    try:
        with DDGS() as ddgs:
            file_results = list(ddgs.text(file_query, max_results=10))
            results.extend(file_results)
    except Exception as e:
        logger.warning(f"Secondary file search failed: {e}")

    if not results:
        logger.warning("Primary searches returned 0 results. Activating fallback scraper...")
        results = fallback_search(query)

    if not results:
        logger.error("All search methods exhausted. No results found.")
        return None

    # Deduplicate results based on URL
    seen_urls = set()
    unique_results = []
    for r in results:
        if r["href"] not in seen_urls:
            unique_results.append(r)
            seen_urls.add(r["href"])

    for r in unique_results:
        r["score"] = score_result(r, planner)

    unique_results.sort(key=lambda x: x["score"], reverse=True)

    logger.info("Ranked Results:")
    for r in unique_results[:15]:
        logger.info(f"Score: {r['score']} | {r.get('title', 'No Title')} ({r.get('href', '')})")

    # ==========================================
    # EXTRACTION PHASE 1: STRICT FILE SEARCH
    # ==========================================
    logger.info("Phase 1: Hunting for direct file downloads...")
    for result in unique_results:
        url = result.get("href", "")
        if not url:
            continue

        # 1A. Try Direct File Download
        if url.lower().endswith(DOWNLOADABLE):
            try:
                return download(url)
            except Exception as e:
                logger.debug(f"Direct download failed for {url}: {e}")

        # 1B. Try Webpage Scraping for Links
        try:
            links = find_download_links(url, planner)
            if links:
                links.sort(key=lambda x: score_file(x), reverse=True)
                logger.info("Found potential files on page:")
                for link in links:
                    logger.info(f"  -> File Score: {score_file(link)} | {link}")

                for link in links:
                    try:
                        return download(link)
                    except Exception as e:
                        logger.debug(f"Failed to download scraped file link {link}: {e}")
        except Exception as e:
            logger.debug(f"Failed to process webpage {url}: {e}")

    # ==========================================
    # EXTRACTION PHASE 2: HTML TABLE SCRAPING
    # ==========================================
    logger.info("Phase 2: No explicit files found. Attempting HTML table extraction...")
    for result in unique_results:
        url = result.get("href", "")
        if not url:
            continue
            
        try:
            table_path = extract_table_from_html(url)
            if table_path:
                return table_path
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")

    logger.error("Exhausted all results. No dataset could be downloaded.")
    return None