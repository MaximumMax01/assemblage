import asyncio
import io
import re
import aiohttp
from PIL import Image

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def clean_query_for_apis(query: str) -> str:
    """Simplifies long 8-word queries into core 2-3 word keywords for tag-based APIs."""
    # Strip common expansion modifiers to find the core subject
    stopwords = ["orthographic", "blueprint", "diagram", "schematic", "neutral", "lighting", "macro", "texture", "drawing", "4k", "photogrammetry"]
    words = query.split()
    core_words = [w for w in words if w.lower() not in stopwords]
    return " ".join(core_words[:4]) if core_words else query

def upgrade_image_url(url: str) -> str:
    """Strips thumbnail constraints from CDNs."""
    if "pinimg.com" in url:
        url = re.sub(r"/\d+x/", "/originals/", url)
    elif "upload.wikimedia.org" in url and "/thumb/" in url:
        url = url.replace("/thumb/", "/")
        url = re.sub(r"/[^/]+$", "", url)
    elif "images-na.ssl-images-amazon.com" in url or "m.media-amazon.com" in url:
        url = re.sub(r"\._AC_.*_\.", ".", url)
    return url

async def search_duckduckgo(query: str, max_results: int = 6) -> list[str]:
    """Engine 1: DuckDuckGo (Best for general web images)."""
    loop = asyncio.get_event_loop()
    try:
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results, type_image="photo"))
                return [upgrade_image_url(r["image"]) for r in results if r.get("image")]
        return await loop.run_in_executor(None, _search)
    except Exception:
        return []

async def search_wikimedia(query: str, max_results: int = 6) -> list[str]:
    """Engine 2: Wikimedia Commons API (Unblocked, high-res architectural & technical scans)."""
    simplified = clean_query_for_apis(query)
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={simplified}&gsrnamespace=6&gsrlimit={max_results}&prop=imageinfo&iiprop=url|mime&format=json"
    connector = aiohttp.TCPConnector(ssl=False)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pages = data.get("query", {}).get("pages", {})
                    urls = []
                    for page in pages.values():
                        info = page.get("imageinfo", [{}])[0]
                        img_url = info.get("url")
                        mime = info.get("mime", "")
                        if img_url and ("jpeg" in mime or "png" in mime or "jpg" in mime):
                            urls.append(upgrade_image_url(img_url))
                    return urls
    except Exception as e:
        print(f"[Scraper] Wikimedia error for '{simplified}': {e}")
    return []

async def search_openverse(query: str, max_results: int = 6) -> list[str]:
    """Engine 3: Openverse API (Over 700M CC-licensed photographs)."""
    simplified = clean_query_for_apis(query)
    url = f"https://api.openverse.org/v1/images/?q={simplified}&page_size={max_results}"
    connector = aiohttp.TCPConnector(ssl=False)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [r["url"] for r in data.get("results", []) if r.get("url")]
    except Exception:
        pass
    return []

async def search_query_urls(query: str) -> list[str]:
    # Fetch up to 12 candidates per query
    urls = await search_duckduckgo(query, max_results=12)
    if not urls or len(urls) < 6:
        wiki_urls, openverse_urls = await asyncio.gather(
            search_wikimedia(query, max_results=8),
            search_openverse(query, max_results=8)
        )
        urls = list(set(urls + wiki_urls + openverse_urls))
    return urls

async def download_image(session: aiohttp.ClientSession, url: str) -> tuple[bytes, str] | None:
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=6)) as resp:
            if resp.status == 200 and "image" in resp.headers.get("Content-Type", "").lower():
                content = await resp.read()
                if len(content) > 25_000:  # Minimum 25KB
                    return content, url
    except Exception:
        pass
    return None

async def fetch_all_candidates(queries: list[str]) -> list[tuple[bytes, str]]:
    tasks = [search_query_urls(q) for q in queries]
    url_batches = await asyncio.gather(*tasks)
    unique_urls = list({u for batch in url_batches for u in batch})

    print(f"[Scraper] Found {len(unique_urls)} candidate URLs across search engines. Downloading...")

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        downloads = [download_image(session, u) for u in unique_urls]
        results = await asyncio.gather(*downloads)

    successful = [r for r in results if r is not None]
    print(f"[Scraper] Successfully downloaded {len(successful)} image byte payloads into memory.")
    return successful