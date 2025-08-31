import re
import time
import typing as t
from collections import deque, defaultdict
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from urllib import robotparser

import json
import os
import hashlib
from datetime import datetime
from html import unescape

try:
    import requests
except ImportError:
    raise SystemExit("Please `pip install requests`")
try:
    from bs4 import BeautifulSoup  # optional but recommended
except ImportError:
    BeautifulSoup = None  # we’ll fall back to a simple regex if missing



def seed_url_discover(
    seeds: t.Iterable[str],
    *,
    allowed_domains: t.Optional[t.Set[str]] = None,
    allow_patterns: t.Optional[t.List[t.Pattern]] = None,
    deny_patterns: t.Optional[t.List[t.Pattern]] = None,
    same_host: bool = True,
    max_pages: int = 200,
    max_depth: int = 3,
    respect_robots: bool = True,
    delay_seconds: float = 0.5,
    timeout_seconds: float = 10.0,
    user_agent: str = "RAG-DiscoveryBot/1.0 (+https://example.com) Python-requests",
) -> t.Tuple[t.List[str], t.Dict[str, t.List[str]], t.Dict[str, str]]:
    """
    Discover internal article URLs starting from seed pages, breadth-first.

    Parameters
    ----------
    seeds : iterable of str
        Starting URLs (e.g., help center home).
    allowed_domains : set[str], optional
        If provided, only keep URLs whose netloc ends with any in this set.
        If None, inferred from seed domains.
    allow_patterns : list[Pattern], optional
        Only enqueue URLs that match AT LEAST ONE of these regex patterns.
        Example (Confluence/Scroll): [re.compile(r"/td/")]
    deny_patterns : list[Pattern], optional
        Drop URLs that match ANY of these patterns (e.g., search/login).
        Example: [re.compile(r"/search"), re.compile(r"/login")]
    same_host : bool
        If True, restrict to the SAME host(s) as the seeds (exact netloc match).
        If False, allow any host that passes allowed_domains (if set).
    max_pages : int
        Hard cap on pages fetched.
    max_depth : int
        Limit crawl depth from each seed.
    respect_robots : bool
        Honor robots.txt disallow rules per host.
    delay_seconds : float
        Throttle between requests (very simple polite crawl).
    timeout_seconds : float
        Per-request timeout.
    user_agent : str
        User-Agent header used for HTTP requests & robots.

    Returns
    -------
    discovered_urls : list[str]
        Unique list of canonicalized article URLs discovered (in crawl order).
    graph : dict[str, list[str]]
        Parent → child link adjacency (only kept URLs).
    skip_reasons : dict[str, str]
        Map of URL → reason it was skipped (debug aid).
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    # Helpers
    def canonicalize(u: str) -> str:
        """Normalize URL: lower host, strip fragments, sort query params."""
        p = urlparse(u)
        # only keep http/https
        if p.scheme not in ("http", "https"):
            return ""
        # sort query params for stability
        q = urlencode(sorted(parse_qsl(p.query, keep_blank_values=True)))
        return urlunparse((p.scheme, p.netloc.lower(), p.path or "/", "", q, ""))

    def same_site_ok(u: str, seed_hosts: t.Set[str]) -> bool:
        host = urlparse(u).netloc.lower()
        return host in seed_hosts

    def domain_allowed(u: str, domains: t.Set[str]) -> bool:
        host = urlparse(u).netloc.lower()
        # allow host if it ends with any allowed domain (handles subdomains)
        return any(host == d or host.endswith("." + d) for d in domains)

    def matches_any(u: str, patterns: t.List[t.Pattern]) -> bool:
        return any(p.search(u) for p in patterns)

    # robots.txt cache per host
    robots_cache: dict[str, robotparser.RobotFileParser] = {}

    def robots_allows(u: str) -> bool:
        if not respect_robots:
            return True
        p = urlparse(u)
        base = f"{p.scheme}://{p.netloc}"
        rp = robots_cache.get(base)
        if rp is None:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(base, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                # If robots cannot be fetched, be conservative: allow fetch
                pass
            robots_cache[base] = rp
        try:
            return rp.can_fetch(user_agent, u)
        except Exception:
            return True  # on parser failure, default-allow

    def extract_links(html: str, base_url: str) -> t.List[str]:
        links: t.List[str] = []
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                links.append(urljoin(base_url, a["href"]))
        else:
            # Fallback: very simple regex; not perfect but works OK
            for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.I):
                links.append(urljoin(base_url, m.group(1)))
        return links

    # Seed setup
    seed_list = [canonicalize(s) for s in seeds if canonicalize(s)]
    if not seed_list:
        return [], {}, {}
    seed_hosts = {urlparse(s).netloc.lower() for s in seed_list}
    if allowed_domains is None:
        allowed_domains = {h for h in seed_hosts}  # exact hosts by default

    allow_patterns = allow_patterns or []
    deny_patterns = deny_patterns or []

    # State
    q: deque[tuple[str, int]] = deque((s, 0) for s in seed_list)
    visited: set[str] = set()
    discovered: list[str] = []
    graph: dict[str, list[str]] = defaultdict(list)
    skip_reasons: dict[str, str] = {}

    # BFS crawl
    while q and len(visited) < max_pages:
        url, depth = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        # Domain/host checks
        if same_host and not same_site_ok(url, seed_hosts):
            skip_reasons[url] = "different_host"
            continue
        if allowed_domains and not domain_allowed(url, allowed_domains):
            skip_reasons[url] = "outside_allowed_domains"
            continue
        if not robots_allows(url):
            skip_reasons[url] = "robots_disallow"
            continue
        if any(p.search(url) for p in deny_patterns):
            skip_reasons[url] = "deny_pattern"
            continue
        if allow_patterns and not matches_any(url, allow_patterns):
            # If allow list is provided, require a match; BUT always allow the seeds
            if url not in seed_list:
                skip_reasons[url] = "no_allow_pattern_match"
                continue

        # Fetch
        try:
            resp = session.get(url, timeout=timeout_seconds)
            # Be polite
            time.sleep(delay_seconds)
            if resp.status_code >= 400:
                skip_reasons[url] = f"http_{resp.status_code}"
                continue
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                skip_reasons[url] = "non_html"
                continue
            enc = resp.encoding or getattr(resp, "apparent_encoding", None) or "utf-8"
            try:
                html = resp.content.decode(enc, errors="replace")
            except Exception:
                html = resp.text
        except requests.RequestException as e:
            skip_reasons[url] = f"request_error:{e.__class__.__name__}"
            continue

        # Record this page as discovered (it passed filters & fetched OK)
        discovered.append(url)

        # Depth limit
        if depth >= max_depth:
            continue

        # Extract & enqueue child links
        for link in extract_links(html, url):
            c = canonicalize(link)
            if not c or c in visited:
                continue
            # Do quick pre-filters for queueing efficiency
            if same_host and not same_site_ok(c, seed_hosts):
                continue
            if allowed_domains and not domain_allowed(c, allowed_domains):
                continue
            if any(p.search(c) for p in deny_patterns):
                continue
            if allow_patterns and not matches_any(c, allow_patterns):
                continue
            graph[url].append(c)
            q.append((c, depth + 1))

    return discovered, graph, skip_reasons

# === Parse & Clean helpers ===
def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove obvious non-article elements in-place."""
    if soup is None:
        return
    for tag in soup.find_all(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    # common layout chrome
    for sel in [
        "header", "footer", "nav", "aside",
        "div[class*='header']", "div[class*='Footer']", "div[class*='footer']",
        "div[class*='nav']", "div[id*='nav']", "div[class*='sidebar']",
        "div[class*='toc']", "div[id*='toc']",
    ]:
        for tag in soup.select(sel):
            tag.decompose()

def _find_main_container(soup: BeautifulSoup) -> t.Optional["Tag"]:
    """Heuristic: pick a likely article container for Confluence/Scroll Viewport pages."""
    if soup is None:
        return None
    candidates = []
    # Common Scroll/Confluence / knowledge base containers
    selectors = [
        "article",
        "main",
        "div.viewport-article__content",
        "div.viewport-article",
        "div#content",
        "div[id*='content']",
        "div[class*='content']",
        "div[class*='article']",
        "div[class*='page']",
        "section[role='main']",
    ]
    for sel in selectors:
        for el in soup.select(sel):
            text_len = len(el.get_text(" ", strip=True))
            p_count = len(el.find_all("p"))
            candidates.append((text_len + 50 * p_count, el))
    if not candidates:
        return soup.body
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def _extract_title(soup: BeautifulSoup) -> str:
    # Try og:title, then h1, then <title>
    meta = soup.find("meta", attrs={"property": "og:title"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    h1 = soup.find(["h1", "h2"])  # some KBs use h2 as top
    if h1:
        return h1.get_text(" ", strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""

def _extract_updated_at(soup: BeautifulSoup) -> t.Optional[str]:
    # Look for <time> or meta dates first
    t_tag = soup.find("time")
    if t_tag and (t_tag.get("datetime") or t_tag.get_text(strip=True)):
        return (t_tag.get("datetime") or t_tag.get_text(strip=True)).strip()
    for meta_name in [
        ("meta", {"property": "article:modified_time"}),
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "last-modified"}),
        ("meta", {"http-equiv": "last-modified"}),
        ("meta", {"name": "date"}),
    ]:
        m = soup.find(*meta_name)
        if m and m.get("content"):
            return m["content"].strip()
    # Fallback: scan for "Last updated" text
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(?i)(last\s*updated\s*[:\-]?\s*)([^\n]+)", text)
    if m:
        return m.group(2).strip()
    return None

def _extract_breadcrumbs(soup: BeautifulSoup) -> t.List[str]:
    # ARIA breadcrumb
    nav = soup.find("nav", attrs={"aria-label": re.compile("breadcrumb", re.I)})
    if nav:
        parts = [li.get_text(" ", strip=True) for li in nav.find_all("li")]
        return [p for p in parts if p]
    # Common breadcrumb classes
    for cls in ["breadcrumbs", "breadcrumb", "viewport-breadcrumbs"]:
        bc = soup.find(class_=re.compile(cls, re.I))
        if bc:
            parts = [li.get_text(" ", strip=True) for li in bc.find_all("li")]
            return [p for p in parts if p]
    return []

def _table_to_markdown(table) -> str:
    # Simple Markdown table from <table>
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        rows.append(cells)
    if not rows:
        return ""
    # Header handling
    header = rows[0]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)

def _html_to_clean_text(container) -> str:
    lines: t.List[str] = []
    for el in container.descendants:
        if getattr(el, "name", None) in {"h1", "h2", "h3", "h4", "h5"}:
            level = int(el.name[1])
            txt = el.get_text(" ", strip=True)
            if txt:
                lines.append("#" * level + " " + txt)
        elif getattr(el, "name", None) == "p":
            txt = el.get_text(" ", strip=True)
            if txt:
                lines.append(txt)
        elif getattr(el, "name", None) in {"li"}:
            txt = el.get_text(" ", strip=True)
            if txt:
                lines.append("- " + txt)
        elif getattr(el, "name", None) in {"pre", "code"}:
            txt = el.get_text("\n", strip=True)
            if txt:
                lines.append("```\n" + txt + "\n```")
        elif getattr(el, "name", None) == "table":
            md = _table_to_markdown(el)
            if md:
                lines.append(md)
    # Collapse excessive blank lines
    text = "\n\n".join([unescape(s).strip() for s in lines if s and s.strip()])
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_article(html: str, url: str) -> t.Dict[str, t.Any]:
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup is required for parse_article(); please `pip install beautifulsoup4`.")
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    title = _extract_title(soup)
    updated_at = _extract_updated_at(soup)
    breadcrumbs = _extract_breadcrumbs(soup)
    container = _find_main_container(soup)
    body_text = _html_to_clean_text(container) if container else ""
    return {
        "url": url,
        "title": title,
        "updated_at": updated_at,
        "breadcrumbs": breadcrumbs,
        "body_text": body_text,
        "char_len": len(body_text),
    }

def fetch_and_parse(url: str, timeout_seconds: float = 10.0, user_agent: str = "RAG-DiscoveryBot/1.0") -> t.Optional[t.Dict[str, t.Any]]:
    try:
        resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout_seconds)
        if resp.status_code >= 400:
            return None
        ct = resp.headers.get("Content-Type", "").lower()
        if "text/html" not in ct:
            return None
        enc = resp.encoding or getattr(resp, "apparent_encoding", None) or "utf-8"
        html = resp.content.decode(enc, errors="replace")
        parsed = parse_article(html, url)
        parsed["raw_html"] = html
        return parsed
    except requests.RequestException:
        return None


def _doc_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_many_to_jsonl(
    urls: t.Iterable[str],
    out_path: str,
    max_items: int = 50,
    assets_dir: t.Optional[str] = "assets",
) -> int:
    """Fetch, parse, and write cleaned article records to JSONL.
    If assets_dir is provided, also save raw HTML and plain-text bodies per doc.
    """
    if assets_dir:
        _ensure_dir(assets_dir)
        _ensure_dir(os.path.join(assets_dir, "raw_html"))
        _ensure_dir(os.path.join(assets_dir, "plain_text"))

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, u in enumerate(urls):
            if i >= max_items:
                break
            rec = fetch_and_parse(u)
            if not rec:
                continue
            if rec.get("char_len", 0) < 200:  # skip empty/super short pages
                continue

            docid = _doc_id(rec["url"])
            raw_html_path = None
            plain_text_path = None
            if assets_dir and rec.get("raw_html"):
                raw_html_path = os.path.join(assets_dir, "raw_html", f"{docid}.html")
                with open(raw_html_path, "w", encoding="utf-8") as hf:
                    hf.write(rec["raw_html"])
                plain_text_path = os.path.join(assets_dir, "plain_text", f"{docid}.txt")
                with open(plain_text_path, "w", encoding="utf-8") as tf:
                    tf.write(rec.get("body_text", ""))
                # Don't persist the full HTML blob inside JSONL
                rec.pop("raw_html", None)

            rec["doc_id"] = docid
            rec["raw_html_path"] = raw_html_path
            rec["plain_text_path"] = plain_text_path

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count

# === Chunking helpers ===
def _simple_tokenize(text: str) -> t.List[str]:
    # Whitespace tokenization as an approximation
    return re.findall(r"\S+", text)

def chunk_text(
    text: str,
    *,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
) -> t.List[str]:
    """Split text into overlapping chunks by approximate tokens.
    Uses a simple whitespace tokenizer; swap with tiktoken if desired.
    """
    tokens = _simple_tokenize(text)
    if not tokens:
        return []
    chunks: t.List[str] = []
    start = 0
    n = len(tokens)
    step = max(1, target_tokens - overlap_tokens)
    while start < n:
        end = min(n, start + target_tokens)
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start += step
    return chunks

def chunk_records(
    parsed_jsonl_path: str,
    chunks_out_path: str,
    *,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
    site_default: t.Optional[str] = None,
) -> int:
    """Read parsed records and emit chunk JSONL with metadata suitable for indexing."""
    written = 0
    with open(parsed_jsonl_path, "r", encoding="utf-8") as f_in, open(chunks_out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            text = rec.get("body_text", "")
            if not text:
                continue
            url = rec.get("url")
            title = rec.get("title", "")
            breadcrumbs = rec.get("breadcrumbs") or []
            section = breadcrumbs[-1] if breadcrumbs else ""
            updated_at = rec.get("updated_at")
            site = site_default or urlparse(url).netloc
            doc_id = rec.get("doc_id") or _doc_id(url)

            pieces = chunk_text(text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
            for idx, chunk in enumerate(pieces):
                out = {
                    "chunk_id": f"{doc_id}-{idx}",
                    "doc_id": doc_id,
                    "url": url,
                    "title": title,
                    "section": section,
                    "updated_at": updated_at,
                    "site": site,
                    "chunk_index": idx,
                    "n_tokens": len(_simple_tokenize(chunk)),
                    "text": chunk,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
    return written

if __name__ == "__main__":
    seeds = ["https://help.98point6.com/"]
    allow = [re.compile(r"/td/")]  # Confluence/Scroll articles often live under /td/
    deny = [re.compile(r"/search"), re.compile(r"/login"), re.compile(r"[?#]preview=")]

    urls, graph, skips = seed_url_discover(
        seeds,
        allow_patterns=allow,
        deny_patterns=deny,
        same_host=True,
        max_pages=300,
        max_depth=3,
        delay_seconds=0.6,
        user_agent="RAG-DiscoveryBot/1.0 (+https://yourdomain.example)",
        respect_robots=False,
    )

    print(f"Discovered {len(urls)} pages")
    # print a few:
    for u in urls[:10]:
        print(" -", u)

    print(graph)
    if not urls:
        from collections import Counter
        reason_counts = Counter(skips.values())
        print("\nSkip reasons (most common):")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")
        print("\nSample skipped URLs:")
        for u, r in list(skips.items())[:10]:
            print(f"  {u} -> {r}")

    # === Parse & Clean demo ===
    sample_out = "parsed_98point6.jsonl"
    kept = parse_many_to_jsonl(urls, sample_out, max_items=40, assets_dir="assets_98point6")
    print(f"\nParsed & saved {kept} records to {sample_out}")
    # Peek at one record
    if kept:
        with open(sample_out, "r", encoding="utf-8") as f:
            line = f.readline()
            try:
                rec = json.loads(line)
                print("\nSample record:\nTitle:", rec.get("title"))
                print("Updated:", rec.get("updated_at"))
                print("URL:", rec.get("url"))
                preview = rec.get("body_text", "")[:400]
                print("Body preview:\n", preview, "...\n")
            except Exception:
                pass

    # === Chunk & Enrich ===
    chunks_out = "chunks_98point6.jsonl"
    num_chunks = chunk_records(sample_out, chunks_out, target_tokens=600, overlap_tokens=100, site_default="help.98point6.com")
    print(f"Wrote {num_chunks} chunks to {chunks_out}")
    # Peek at one chunk
    if num_chunks:
        with open(chunks_out, "r", encoding="utf-8") as cf:
            print("\nSample chunk:")
            print(cf.readline().strip()[:400] + "...")



