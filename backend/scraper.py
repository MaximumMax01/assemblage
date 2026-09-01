import asyncio
<<<<<<< HEAD
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
=======
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence

import aiohttp
from urllib.parse import urlparse

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from net import HEADERS, make_connector

try:
    from ddgs import DDGS
    _DDG_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on which package is installed
    try:
        from duckduckgo_search import DDGS
        _DDG_AVAILABLE = True
    except ImportError:
        DDGS = None  # type: ignore[assignment]
        _DDG_AVAILABLE = False


class Candidate(NamedTuple):
    """A downloaded image payload, tagged with the slot whose query found it."""
    data: bytes
    url: str
    slot: str


# Stock photo hosts serve visibly watermarked comps. CLIP negative anchors catch
# some of them, but a domain check is deterministic, free, and runs before the
# download rather than after. ftcdn.net is Adobe Stock's CDN, which is where the
# "Adobe Stock" strip along the bottom of a result comes from.
STOCK_HOSTS = (
    "shutterstock.com", "dreamstime.com", "alamy.com", "123rf.com",
    "depositphotos.com", "istockphoto.com", "gettyimages.com",
    "stock.adobe.com", "ftcdn.net", "canstockphoto.com", "vectorstock.com",
    "bigstockphoto.com", "agefotostock.com", "imago-images.de", "zoonar.com",
    "stockfresh.com", "photostockeditor.com", "picfair.com",
)


def is_stock_host(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in STOCK_HOSTS)


def clean_query_for_apis(query: str) -> str:
    """Simplifies long queries into core keywords for tag-based APIs."""
    stopwords = {
        "orthographic", "blueprint", "diagram", "schematic", "neutral",
        "lighting", "macro", "texture", "drawing", "4k", "photogrammetry",
        "photograph", "closeup", "close", "up", "detail", "view",
    }
>>>>>>> b3ed0a9 (second commit, for school)
    words = query.split()
    core_words = [w for w in words if w.lower() not in stopwords]
    return " ".join(core_words[:4]) if core_words else query

<<<<<<< HEAD
def upgrade_image_url(url: str) -> str:
    """Strips thumbnail constraints from CDNs."""
=======

def upgrade_image_url(url: str) -> str:
    """Strips thumbnail constraints from known CDNs."""
>>>>>>> b3ed0a9 (second commit, for school)
    if "pinimg.com" in url:
        url = re.sub(r"/\d+x/", "/originals/", url)
    elif "upload.wikimedia.org" in url and "/thumb/" in url:
        url = url.replace("/thumb/", "/")
        url = re.sub(r"/[^/]+$", "", url)
    elif "images-na.ssl-images-amazon.com" in url or "m.media-amazon.com" in url:
        url = re.sub(r"\._AC_.*_\.", ".", url)
    return url

<<<<<<< HEAD
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
=======

async def search_duckduckgo(query: str, max_results: int = 12) -> List[str]:
    """
    Engine 1: DuckDuckGo.

    This runs through an unofficial wrapper that breaks whenever DuckDuckGo
    changes their endpoints, so every failure mode here is non-fatal and the
    open APIs below carry the load when it goes down.
    """
    if not _DDG_AVAILABLE or DDGS is None:
        return []

    loop = asyncio.get_event_loop()

    def _search() -> List[str]:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
            return [upgrade_image_url(r["image"]) for r in results if r.get("image")]

    try:
        return await loop.run_in_executor(None, _search)
    except Exception as exc:
        print(f"[Scraper] DuckDuckGo unavailable for '{query[:40]}': {exc}")
        return []


async def search_wikimedia(
    session: aiohttp.ClientSession, query: str, max_results: int = 8
) -> List[str]:
    """Engine 2: Wikimedia Commons. High-res technical and architectural scans."""
    simplified = clean_query_for_apis(query)
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={simplified}&gsrnamespace=6&gsrlimit={max_results}"
        "&prop=imageinfo&iiprop=url|mime&format=json"
    )
    try:
        async with session.get(
            url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=6)
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            pages = data.get("query", {}).get("pages", {})
            urls = []
            for page in pages.values():
                info = (page.get("imageinfo") or [{}])[0]
                img_url = info.get("url")
                mime = info.get("mime", "")
                if img_url and any(t in mime for t in ("jpeg", "png", "jpg")):
                    urls.append(upgrade_image_url(img_url))
            return urls
    except Exception as exc:
        print(f"[Scraper] Wikimedia error for '{simplified}': {exc}")
    return []


async def search_openverse(
    session: aiohttp.ClientSession, query: str, max_results: int = 8
) -> List[str]:
    """Engine 3: Openverse. Creative Commons index."""
    simplified = clean_query_for_apis(query)
    url = f"https://api.openverse.org/v1/images/?q={simplified}&page_size={max_results}"
    try:
        async with session.get(
            url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=6)
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return [r["url"] for r in data.get("results", []) if r.get("url")]
    except Exception as exc:
        print(f"[Scraper] Openverse error for '{simplified}': {exc}")
    return []


async def search_slot_urls(
    session: aiohttp.ClientSession, slot: str, query: str
) -> List[tuple[str, str]]:
    """
    Runs all engines for one slot's query and returns (url, slot) pairs.

    All three engines are always queried rather than using the open APIs only as
    a fallback. Wikimedia in particular is the best source of orthographic plates
    and scientific illustration, so skipping it whenever DuckDuckGo happened to
    return six results was quietly starving the ortho slot.
    """
    ddg, wiki, openverse = await asyncio.gather(
        search_duckduckgo(query, max_results=12),
        search_wikimedia(session, query, max_results=8),
        search_openverse(session, query, max_results=8),
    )

    seen: set[str] = set()
    ordered: List[tuple[str, str]] = []
    blocked = 0
    for url in [*ddg, *wiki, *openverse]:
        if url in seen:
            continue
        if is_stock_host(url):
            blocked += 1
            continue
        seen.add(url)
        ordered.append((url, slot))
    if blocked:
        print(f"[Scraper] {slot}: skipped {blocked} watermarked stock results.")
    return ordered


async def download_image(
    session: aiohttp.ClientSession, url: str, slot: str
) -> Optional[Candidate]:
    try:
        async with session.get(
            url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=8)
        ) as resp:
            if resp.status == 200 and "image" in resp.headers.get("Content-Type", "").lower():
                content = await resp.read()
                if len(content) > 25_000:
                    return Candidate(data=content, url=url, slot=slot)
>>>>>>> b3ed0a9 (second commit, for school)
    except Exception:
        pass
    return None

<<<<<<< HEAD
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
=======

async def fetch_all_candidates(slot_queries: Sequence) -> List[Candidate]:
    """
    Fetches candidates for every slot query, preserving which slot found what.

    A URL that turns up under two different slots is downloaded once and
    assigned to the first slot that found it, so the ordering of SLOT_ORDER
    decides ties.
    """
    connector = make_connector()
    async with aiohttp.ClientSession(connector=connector) as session:
        url_batches = await asyncio.gather(
            *[search_slot_urls(session, sq.slot, sq.query) for sq in slot_queries]
        )

        first_slot_for_url: Dict[str, str] = {}
        for batch in url_batches:
            for url, slot in batch:
                first_slot_for_url.setdefault(url, slot)

        per_slot: Dict[str, int] = {}
        for slot in first_slot_for_url.values():
            per_slot[slot] = per_slot.get(slot, 0) + 1
        print(f"[Scraper] {len(first_slot_for_url)} unique URLs by slot: {per_slot}")

        results = await asyncio.gather(
            *[download_image(session, u, s) for u, s in first_slot_for_url.items()]
        )

    successful = [r for r in results if r is not None]
    downloaded: Dict[str, int] = {}
    for c in successful:
        downloaded[c.slot] = downloaded.get(c.slot, 0) + 1
    print(f"[Scraper] Downloaded {len(successful)} payloads by slot: {downloaded}")
>>>>>>> b3ed0a9 (second commit, for school)
    return successful