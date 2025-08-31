#!/usr/bin/env python3
"""
Simple, modular crawl → parse → chunk pipeline
- Reads a JSON config to avoid hardcoding
- Strict same-domain controls
- Encoding-safe fetch
- Clean, reusable classes: Crawler, Parser, Chunker

Run:
  python web_crawler.py --config config.json

Example config (save as config.json and edit as needed):
{
  "crawl": {
    "seed_urls": ["https://help.98point6.com/"],
    "allow_patterns": ["/td/"],
    "deny_patterns": ["/search", "/login", "[?#]preview="],
    "allowed_domains": ["help.98point6.com"],
    "same_host": true,
    "max_pages": 300,
    "max_depth": 5,
    "respect_robots": true,
    "delay_seconds": 0.6,
    "timeout_seconds": 10.0,
    "user_agent": "RAG-DiscoveryBot/1.0 (+https://yourdomain.example)"
  },
  "outputs": {
    "assets_dir": "assets_98point6",
    "parsed_jsonl_path": "parsed_98point6.jsonl",
    "chunks_jsonl_path": "chunks_98point6.jsonl",
    "fresh_run": true
  },
  "chunk": {
    "target_tokens": 600,
    "overlap_tokens": 100,
    "site": "help.98point6.com"
  }
}
"""

# ===== utils & imports =====
import os
import re
import json
import time
import shutil
import hashlib
import typing as t
from dataclasses import dataclass, field
from collections import deque, defaultdict, Counter
from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl, urlencode
from urllib import robotparser
from html import unescape

try:
    import requests
except ImportError:
    raise SystemExit("Please `pip install requests`.")

try:
    from bs4 import BeautifulSoup  # optional but recommended
except ImportError:
    BeautifulSoup = None

# ===== config types =====
@dataclass
class CrawlConfig:
    seed_urls: t.List[str]
    allow_patterns: t.List[str] = field(default_factory=list)
    deny_patterns: t.List[str] = field(default_factory=list)
    allowed_domains: t.List[str] = field(default_factory=list)
    same_host: bool = True
    max_pages: int = 300
    max_depth: int = 5
    respect_robots: bool = True
    delay_seconds: float = 0.6
    timeout_seconds: float = 10.0
    user_agent: str = "RAG-DiscoveryBot/1.0"

@dataclass
class OutputConfig:
    assets_dir: str = "assets"
    parsed_jsonl_path: str = "parsed.jsonl"
    chunks_jsonl_path: str = "chunks.jsonl"
    fresh_run: bool = False

@dataclass
class ChunkConfig:
    target_tokens: int = 600
    overlap_tokens: int = 100
    site: t.Optional[str] = None

# ===== helpers =====
def canonicalize(u: str) -> str:
    p = urlparse(u)
    if p.scheme not in ("http", "https"):
        return ""
    q = urlencode(sorted(parse_qsl(p.query, keep_blank_values=True)))
    return urlunparse((p.scheme, p.netloc.lower(), p.path or "/", "", q, ""))

def domain_allowed(url: str, allowed: t.Set[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in allowed)

def same_site_ok(url: str, seed_hosts: t.Set[str]) -> bool:
    return urlparse(url).netloc.lower() in seed_hosts

def compile_patterns(patterns: t.List[str]) -> t.List[re.Pattern]:
    return [re.compile(p) for p in patterns]

# ===== crawler =====
class Crawler:
    def __init__(self, cfg: CrawlConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent})
        self.seed_list = [canonicalize(s) for s in cfg.seed_urls if canonicalize(s)]
        if not self.seed_list:
            raise ValueError("No valid seed URLs.")
        self.seed_hosts = {urlparse(s).netloc.lower() for s in self.seed_list}
        self.allowed_domains = set(cfg.allowed_domains) if cfg.allowed_domains else set(self.seed_hosts)
        self.allow_patterns = compile_patterns(cfg.allow_patterns)
        self.deny_patterns = compile_patterns(cfg.deny_patterns)
        self.robots_cache: dict[str, robotparser.RobotFileParser] = {}

    # --- robots ---
    def robots_allows(self, url: str) -> bool:
        if not self.cfg.respect_robots:
            return True
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        rp = self.robots_cache.get(base)
        if rp is None:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(base, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                pass
            self.robots_cache[base] = rp
        try:
            return rp.can_fetch(self.cfg.user_agent, url)
        except Exception:
            return True

    # --- fetch ---
    def fetch_html(self, url: str) -> t.Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.cfg.timeout_seconds)
            time.sleep(self.cfg.delay_seconds)
            if resp.status_code >= 400:
                return None
            ct = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in ct:
                return None
            enc = resp.encoding or getattr(resp, "apparent_encoding", None) or "utf-8"
            return resp.content.decode(enc, errors="replace")
        except requests.RequestException:
            return None

    # --- link extraction ---
    def extract_links(self, html: str, base_url: str) -> t.List[str]:
        links: t.List[str] = []
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                links.append(urljoin(base_url, a["href"]))
        else:
            for m in re.finditer(r'href["\\\']?=\\s*["\\\']([^"\\\']+)["\\\']', html, flags=re.I):
                links.append(urljoin(base_url, m.group(1)))
        return links

    def discover(self) -> t.Tuple[t.List[str], t.Dict[str, t.List[str]], t.Dict[str, str]]:
        q: deque[tuple[str, int]] = deque((s, 0) for s in self.seed_list)
        visited: set[str] = set()
        discovered: list[str] = []
        graph: dict[str, list[str]] = defaultdict(list)
        skip_reasons: dict[str, str] = {}

        while q and len(visited) < self.cfg.max_pages:
            url, depth = q.popleft()
            if url in visited:
                continue
            visited.add(url)

            # host/domain checks
            if self.cfg.same_host and not same_site_ok(url, self.seed_hosts):
                skip_reasons[url] = "different_host"
                continue
            if self.allowed_domains and not domain_allowed(url, self.allowed_domains):
                skip_reasons[url] = "outside_allowed_domains"
                continue
            if not self.robots_allows(url):
                skip_reasons[url] = "robots_disallow"
                continue
            if any(p.search(url) for p in self.deny_patterns):
                skip_reasons[url] = "deny_pattern"
                continue
            if self.allow_patterns and not any(p.search(url) for p in self.allow_patterns):
                if url not in self.seed_list:
                    skip_reasons[url] = "no_allow_pattern_match"
                    continue

            html = self.fetch_html(url)
            if not html:
                skip_reasons[url] = "fetch_failed_or_non_html"
                continue

            discovered.append(url)

            if depth >= self.cfg.max_depth:
                continue

            for link in self.extract_links(html, url):
                c = canonicalize(link)
                if not c or c in visited:
                    continue
                if self.cfg.same_host and not same_site_ok(c, self.seed_hosts):
                    skip_reasons[c] = "different_host_child"
                    continue
                if self.allowed_domains and not domain_allowed(c, self.allowed_domains):
                    skip_reasons[c] = "outside_allowed_domains_child"
                    continue
                if any(p.search(c) for p in self.deny_patterns):
                    continue
                if self.allow_patterns and not any(p.search(c) for p in self.allow_patterns):
                    continue
                graph[url].append(c)
                q.append((c, depth + 1))

        # Safety check
        if self.cfg.same_host:
            other_hosts = {urlparse(u).netloc.lower() for u in discovered if urlparse(u).netloc.lower() not in self.seed_hosts}
            if other_hosts:
                print("WARNING: discovered URLs outside seed hosts:", sorted(other_hosts))

        return discovered, graph, skip_reasons

# ===== parser =====
class Parser:
    def __init__(self):
        if BeautifulSoup is None:
            raise RuntimeError("BeautifulSoup is required. Please `pip install beautifulsoup4`.")

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        for tag in soup.find_all(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()
        for sel in [
            "header", "footer", "nav", "aside",
            "div[class*='header']", "div[class*='Footer']", "div[class*='footer']",
            "div[class*='nav']", "div[id*='nav']", "div[class*='sidebar']",
            "div[class*='toc']", "div[id*='toc']",
        ]:
            for tag in soup.select(sel):
                tag.decompose()

    def _find_main(self, soup: BeautifulSoup):
        candidates = []
        selectors = [
            "article", "main", "div.viewport-article__content", "div.viewport-article",
            "div#content", "div[id*='content']", "div[class*='content']",
            "div[class*='article']", "div[class*='page']", "section[role='main']",
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

    def _extract_title(self, soup: BeautifulSoup) -> str:
        meta = soup.find("meta", attrs={"property": "og:title"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        h1 = soup.find(["h1", "h2"])  # some KBs use h2
        if h1:
            return h1.get_text(" ", strip=True)
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        return ""

    def _extract_updated(self, soup: BeautifulSoup) -> t.Optional[str]:
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
        text = soup.get_text("\n", strip=True)
        m = re.search(r"(?i)(last\\s*updated\\s*[:\\-]?\\s*)([^\\n]+)", text)
        if m:
            return m.group(2).strip()
        return None

    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> t.List[str]:
        nav = soup.find("nav", attrs={"aria-label": re.compile("breadcrumb", re.I)})
        if nav:
            parts = [li.get_text(" ", strip=True) for li in nav.find_all("li")]
            return [p for p in parts if p]
        for cls in ["breadcrumbs", "breadcrumb", "viewport-breadcrumbs"]:
            bc = soup.find(class_=re.compile(cls, re.I))
            if bc:
                parts = [li.get_text(" ", strip=True) for li in bc.find_all("li")]
                return [p for p in parts if p]
        return []

    def _table_to_markdown(self, table) -> str:
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            rows.append(cells)
        if not rows:
            return ""
        header = rows[0]
        out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for r in rows[1:]:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

    def _html_to_text(self, container) -> str:
        lines: t.List[str] = []
        for el in container.descendants:
            name = getattr(el, "name", None)
            if name in {"h1","h2","h3","h4","h5"}:
                level = int(name[1])
                txt = el.get_text(" ", strip=True)
                if txt:
                    lines.append("#"*level + " " + txt)
            elif name == "p":
                txt = el.get_text(" ", strip=True)
                if txt:
                    lines.append(txt)
            elif name == "li":
                txt = el.get_text(" ", strip=True)
                if txt:
                    lines.append("- " + txt)
            elif name in {"pre","code"}:
                txt = el.get_text("\n", strip=True)
                if txt:
                    lines.append("```\n" + txt + "\n```")
            elif name == "table":
                md = self._table_to_markdown(el)
                if md:
                    lines.append(md)
        text = "\n\n".join([unescape(s).strip() for s in lines if s and s.strip()])
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_article(self, html: str, url: str) -> t.Dict[str, t.Any]:
        soup = BeautifulSoup(html, "html.parser")
        self._strip_noise(soup)
        title = self._extract_title(soup)
        updated_at = self._extract_updated(soup)
        breadcrumbs = self._extract_breadcrumbs(soup)
        container = self._find_main(soup)
        body_text = self._html_to_text(container) if container else ""
        return {
            "url": url,
            "title": title,
            "updated_at": updated_at,
            "breadcrumbs": breadcrumbs,
            "body_text": body_text,
            "char_len": len(body_text),
        }

# ===== chunker =====
class Chunker:
    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg

    @staticmethod
    def _doc_id(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _simple_tokenize(text: str) -> t.List[str]:
        return re.findall(r"\S+", text)

    def chunk_text(self, text: str) -> t.List[str]:
        if not text:
            return []
        toks = self._simple_tokenize(text)
        tgt = max(50, int(self.cfg.target_tokens))
        ovl = max(0, int(self.cfg.overlap_tokens))
        step = max(1, tgt - ovl)
        n = len(toks)
        out: t.List[str] = []
        s = 0
        while s < n:
            e = min(n, s + tgt)
            out.append(" ".join(toks[s:e]))
            if e == n:
                break
            s += step
        return out

    def save_assets(self, rec: dict, assets_dir: str) -> t.Tuple[t.Optional[str], t.Optional[str]]:
        raw_html_path = None
        plain_text_path = None
        if rec.get("raw_html"):
            self._ensure_dir(os.path.join(assets_dir, "raw_html"))
            self._ensure_dir(os.path.join(assets_dir, "plain_text"))
            docid = rec.get("doc_id") or self._doc_id(rec["url"])
            raw_html_path = os.path.join(assets_dir, "raw_html", f"{docid}.html")
            with open(raw_html_path, "w", encoding="utf-8") as hf:
                hf.write(rec["raw_html"])
            plain_text_path = os.path.join(assets_dir, "plain_text", f"{docid}.txt")
            with open(plain_text_path, "w", encoding="utf-8") as tf:
                tf.write(rec.get("body_text", ""))
        return raw_html_path, plain_text_path

# ===== pipeline ops =====
def fetch_and_parse_many(urls: t.Iterable[str], crawler: Crawler, parser: Parser, out_path: str, assets_dir: str, max_items: int = 100) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, u in enumerate(urls):
            if i >= max_items:
                break
            html = crawler.fetch_html(u)
            if not html:
                continue
            rec = parser.parse_article(html, u)
            if rec.get("char_len", 0) < 200:
                continue
            # attach raw html for asset saving
            rec["raw_html"] = html
            rec["doc_id"] = Chunker._doc_id(u)
            # we defer writing raw assets to the chunk/save phase
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count

def chunk_from_parsed(parsed_jsonl: str, out_chunks: str, chunker: Chunker, site_default: t.Optional[str], assets_dir: str) -> int:
    os.makedirs(os.path.dirname(out_chunks) or ".", exist_ok=True)
    written = 0
    with open(parsed_jsonl, "r", encoding="utf-8") as f_in, open(out_chunks, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            url = rec.get("url"); title = rec.get("title", "")
            breadcrumbs = rec.get("breadcrumbs") or []
            section = breadcrumbs[-1] if breadcrumbs else ""
            updated_at = rec.get("updated_at")
            site = site_default or urlparse(url).netloc
            doc_id = rec.get("doc_id") or Chunker._doc_id(url)

            raw_html_path, plain_text_path = chunker.save_assets(rec, assets_dir)
            # do not persist raw_html blob inside JSONL
            rec.pop("raw_html", None)

            for idx, chunk in enumerate(chunker.chunk_text(rec.get("body_text", ""))):
                out = {
                    "chunk_id": f"{doc_id}-{idx}",
                    "doc_id": doc_id,
                    "url": url,
                    "title": title,
                    "section": section,
                    "updated_at": updated_at,
                    "site": site,
                    "chunk_index": idx,
                    "n_tokens": len(Chunker._simple_tokenize(chunk)),
                    "text": chunk,
                    "raw_html_path": raw_html_path,
                    "plain_text_path": plain_text_path,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
    return written

# ===== CLI / main =====
@dataclass
class FullConfig:
    crawl: CrawlConfig
    outputs: OutputConfig
    chunk: ChunkConfig

def load_config(path: str) -> t.Tuple[CrawlConfig, OutputConfig, ChunkConfig]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    c = cfg.get("crawl", {})
    o = cfg.get("outputs", {})
    k = cfg.get("chunk", {})
    crawl_cfg = CrawlConfig(
        seed_urls=c.get("seed_urls", []),
        allow_patterns=c.get("allow_patterns", []),
        deny_patterns=c.get("deny_patterns", []),
        allowed_domains=c.get("allowed_domains", []),
        same_host=c.get("same_host", True),
        max_pages=int(c.get("max_pages", 300)),
        max_depth=int(c.get("max_depth", 5)),
        respect_robots=bool(c.get("respect_robots", True)),
        delay_seconds=float(c.get("delay_seconds", 0.6)),
        timeout_seconds=float(c.get("timeout_seconds", 10.0)),
        user_agent=c.get("user_agent", "RAG-DiscoveryBot/1.0"),
    )
    out_cfg = OutputConfig(
        assets_dir=o.get("assets_dir", "assets"),
        parsed_jsonl_path=o.get("parsed_jsonl_path", "parsed.jsonl"),
        chunks_jsonl_path=o.get("chunks_jsonl_path", "chunks.jsonl"),
        fresh_run=bool(o.get("fresh_run", False)),
    )
    chunk_cfg = ChunkConfig(
        target_tokens=int(k.get("target_tokens", 600)),
        overlap_tokens=int(k.get("overlap_tokens", 100)),
        site=k.get("site"),
    )
    return crawl_cfg, out_cfg, chunk_cfg

def cleanup_outputs(out_cfg: OutputConfig):
    if out_cfg.fresh_run:
        try:
            shutil.rmtree(out_cfg.assets_dir)
            print(f"Removed directory: {out_cfg.assets_dir}")
        except FileNotFoundError:
            pass
        for p in (out_cfg.parsed_jsonl_path, out_cfg.chunks_jsonl_path):
            try:
                os.remove(p)
                print(f"Removed file: {p}")
            except FileNotFoundError:
                pass

def summarize_discovery(discovered: t.List[str], skip_reasons: t.Dict[str, str]):
    print(f"Discovered {len(discovered)} pages")
    if discovered:
        for u in discovered[:10]:
            print(" -", u)
    if skip_reasons:
        cnt = Counter(skip_reasons.values())
        print("\nSkip reasons (top):")
        for reason, n in cnt.most_common(10):
            print(f"  {reason}: {n}")


# ===== diagnostics =====
def preflight_report(crawler: Crawler) -> bool:
    """Print a quick diagnostics table for seed URLs. Returns True if at least one seed is crawlable."""
    ok_any = False
    print("Preflight on seeds (robots/patterns):")
    for s in crawler.seed_list:
        reasons = []
        host_ok = same_site_ok(s, crawler.seed_hosts) if crawler.cfg.same_host else True
        dom_ok = domain_allowed(s, crawler.allowed_domains) if crawler.allowed_domains else True
        robots_ok = crawler.robots_allows(s)
        deny_hit = any(p.search(s) for p in crawler.deny_patterns)
        allow_hit = (not crawler.allow_patterns) or any(p.search(s) for p in crawler.allow_patterns) or (s in crawler.seed_list)

        status_ok = host_ok and dom_ok and robots_ok and not deny_hit
        ok_any = ok_any or status_ok

        if not host_ok: reasons.append("different_host")
        if not dom_ok: reasons.append("outside_allowed_domains")
        if not robots_ok: reasons.append("robots_disallow")
        if deny_hit: reasons.append("deny_pattern")
        if not allow_hit: reasons.append("no_allow_pattern_match")

        status = "OK" if status_ok else "BLOCKED"
        print(f" - {s} → {status}" + ("" if not reasons else f" ({', '.join(reasons)})"))

    if not ok_any:
        print("All seeds are blocked. Tip: use a seed URL that is allowed by robots and matches allow_patterns (e.g., a topic page like '/td/').")
    return ok_any


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    args = ap.parse_args()

    crawl_cfg, out_cfg, chunk_cfg = load_config(args.config)

    # fresh run cleanup
    cleanup_outputs(out_cfg)

    # discovery
    crawler = Crawler(crawl_cfg)
    # Preflight diagnostics: check seeds against robots/patterns
    if not preflight_report(crawler):
        return

    discovered, graph, skips = crawler.discover()
    summarize_discovery(discovered, skips)

    if not discovered:
        print("No pages discovered — adjust allow/deny patterns or seeds.")
        return

    # parse & clean
    parser = Parser()
    kept = fetch_and_parse_many(
        discovered,
        crawler,
        parser,
        out_path=out_cfg.parsed_jsonl_path,
        assets_dir=out_cfg.assets_dir,
        max_items=crawl_cfg.max_pages,
    )
    print(f"\nParsed & saved {kept} records to {out_cfg.parsed_jsonl_path}")

    # peek
    try:
        with open(out_cfg.parsed_jsonl_path, "r", encoding="utf-8") as f:
            sample = json.loads(next(iter(f)))
            print("Sample record:")
            print(" Title:", sample.get("title"))
            print(" Updated:", sample.get("updated_at"))
            print(" URL:", sample.get("url"))
            print(" Body preview:", (sample.get("body_text") or "")[:200].replace("\n", " "), "...")
    except Exception:
        pass

    # chunk & enrich
    chunker = Chunker(chunk_cfg)
    n_chunks = chunk_from_parsed(
        out_cfg.parsed_jsonl_path,
        out_cfg.chunks_jsonl_path,
        chunker,
        site_default=chunk_cfg.site,
        assets_dir=out_cfg.assets_dir,
    )
    print(f"Wrote {n_chunks} chunks to {out_cfg.chunks_jsonl_path}")

if __name__ == "__main__":
    main()