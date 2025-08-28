from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from urllib.parse import urlparse, unquote

from httpx import AsyncClient, Response, TimeoutException, HTTPError, RequestError
from parsel import Selector
from loguru import logger as log
import csv


# -------------------------
# --- HTTP & Utilities  ---
# -------------------------

DEFAULT_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Cookie": "intl_splash=false",
}

def normalize_profile_url(url: str) -> str:
    """Remplace www par old et normalise l'URL profil (termine par /)."""
    if not url:
        return url
    url = re.sub(r"^https?://www\.reddit\.com", "https://old.reddit.com", url.strip())
    url = re.sub(r"[?#].*$", "", url)
    if not url.endswith("/"):
        url += "/"
    return url

def join_url(base: str, suffix: str) -> str:
    base = normalize_profile_url(base)
    if not base.endswith("/"):
        base += "/"
    if suffix.startswith("/"):
        suffix = suffix[1:]
    return base + suffix

def safe_author_from_profile(profile_url: str) -> str:
    """Extrait /user/<author>/ et sanitise pour le nom de fichier."""
    try:
        path = urlparse(profile_url).path  # e.g. /user/Haunting-Constant973/
        m = re.search(r"/user/([^/]+)/?", path)
        author = m.group(1) if m else "unknown"
    except Exception:
        author = "unknown"
    author = unquote(author)
    author = re.sub(r"[^\w\-.@+]+", "_", author)
    return author

async def _fetch_with_retries(
    client: AsyncClient,
    url: str,
    max_retries: int = 3,
    backoff: float = 0.8,
) -> Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp
        except (TimeoutException, HTTPError, RequestError) as e:
            last_exc = e
            log.warning(f"[{attempt}/{max_retries}] fetch failed: {url} -> {e}")
            await asyncio.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc


# ----------------------------
# --- Parsers (old.reddit) ---
# ----------------------------

def _extract_text(selector: Selector, xpath: str) -> Optional[str]:
    v = selector.xpath(xpath).get()
    if v:
        v = re.sub(r"\s+", " ", v).strip()
    return v

def _extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d+", text.replace(",", ""))
    return int(m.group(0)) if m else None

def _find_next_url(selector: Selector) -> Optional[str]:
    return selector.xpath("//span[@class='next-button']/a/@href").get()

def parse_user_comments(response: Response, author_profile: str) -> List[Dict]:
    sel = Selector(response.text)
    items = sel.xpath("//div[contains(@class,'thing') and contains(@class,'comment')]")
    out: List[Dict] = []
    for it in items:
        link = it.xpath(".//a[contains(@class,'bylink') and contains(@class,'comments')]/@href").get()
        if not link:
            link = it.xpath(".//a[contains(@class,'bylink')]/@href").get()
        title = it.xpath(".//a[contains(@href,'/r/')]/text()").get()
        # score_text = it.xpath(".//span[contains(@class,'score')]/text()").get()
        # created = it.xpath(".//time/@datetime").get() or it.xpath(".//time/@title").get()
        body_text = " ".join(it.xpath(".//div[contains(@class,'md')]/p//text()").getall()).strip() or None
        # author_name = it.xpath(".//a[contains(@class,'author')]/text()").get()

        out.append({
            "type": "comment",
            # "author": author_name,
            # "authorProfile": author_profile,
            "link": link,
            "title": title,
            # "score": _extract_int(score_text),
            # "created": created,
            "body_text": body_text,
        })
    return out

def parse_user_submitted(response: Response, author_profile: str) -> List[Dict]:
    sel = Selector(response.text)
    items = sel.xpath("//div[contains(@class,'thing') and contains(@class,'link')]")
    out: List[Dict] = []
    for it in items:
        link: str = it.xpath(".//a[contains(@class,'title')]/@href").get()
        if not link.startswith("/r/"):
            continue
        link = "https://www.reddit.com" + link
        title = it.xpath(".//a[contains(@class,'title')]/text()").get()
        
        # subreddit = it.xpath(".//a[contains(@href,'/r/')]/text()").get()
        # score_text = it.xpath(".//div[contains(@class,'score')]/text()").get() or it.xpath(".//span[contains(@class,'score')]/text()").get()
        # num_comments_text = it.xpath(".//a[contains(@class,'comments')]/text()").get()
        created = it.xpath(".//time/@datetime").get() or it.xpath(".//time/@title").get()
        # author_name = it.xpath(".//a[contains(@class,'author')]/text()").get()

        # print(title, link, subreddit, score_text, num_comments_text, created, author_name)

        out.append({
            "type": "post",
            # "author": author_name,
            # "authorProfile": author_profile,
            "title": title.strip() if title else None,
            "link": link,
            # "subreddit": subreddit,
            # "score": _extract_int(score_text),
            # "num_comments": _extract_int(num_comments_text),
            "created": created,
        })
    return out


# -----------------------------
# --- The Scraper Class     ---
# -----------------------------

class RedditUserScraper:
    """
    - Prend un fichier JSON (schéma plat OU info+comments+replies) et en extrait les authorProfile.
    - Normalise vers old.reddit.
    - Méthodes distinctes:
        * scrape_user_comments(profile_url): .../comments/ (pagination)
        * scrape_user_submitted(profile_url): .../submitted/ (pagination)
    - Export configurable (JSON/CSV) dans un dossier cible, un fichier par auteur:
        * <author>_post_comments.json|csv
    """

    def __init__(
        self,
        output_format: str = "json",
        output_dir: Union[str, Path] = "out",
        client: Optional[AsyncClient] = None,
        concurrency: int = 4,
    ) -> None:
        assert output_format in {"json", "csv"}, "output_format must be 'json' or 'csv'"
        self.output_format = output_format
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = client or AsyncClient(http2=True, headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30.0)
        self._sem = asyncio.Semaphore(max(1, concurrency))

    # ---- JSON ingestion (collect profiles) ----
    def _add_profile(self, profiles: Set[str], node: dict) -> None:
        """Ajoute un authorProfile trouvé dans n'importe quel dict."""
        if not isinstance(node, dict):
            return
        url = node.get("authorProfile")
        if url:
            profiles.add(normalize_profile_url(url))

    def _walk_json(self, record: dict, profiles: Set[str]) -> None:
        """
        Gère les 2 schémas:
          - Schéma plat: dict avec authorProfile directement -> on l'ajoute.
          - Schéma 'post': dict avec 'info' + 'comments' -> on parcourt récursivement.
        """
        # 1) user plat possible
        if "authorProfile" in record and isinstance(record["authorProfile"], str):
            self._add_profile(profiles, record)

        # 2) schéma post
        info = record.get("info")
        if isinstance(info, dict):
            self._add_profile(profiles, info)

        def walk_comments(nodes: Optional[List[dict]]):
            if not nodes:
                return
            for c in nodes:
                self._add_profile(profiles, c)
                reps = c.get("replies", [])
                if isinstance(reps, list):
                    walk_comments(reps)

        comments = record.get("comments")
        if isinstance(comments, list):
            walk_comments(comments)

    def collect_profiles_from_json_file(self, json_path: Union[str, Path]) -> List[str]:
        """
        Lit un fichier JSON et retourne la liste des authorProfile normalisés (old.reddit).
        Accepte:
          - Liste de users (schéma plat)
          - Un seul post (dict)
          - Liste de posts (list[dict])
        """
        json_path = Path(json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        profiles: Set[str] = set()

        if isinstance(data, dict):
            self._walk_json(data, profiles)
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    self._walk_json(rec, profiles)
        else:
            log.warning("JSON format inattendu: ni dict ni list.")

        out = sorted(profiles)
        log.info(f"Collected {len(out)} unique authorProfile URLs from JSON.")
        return out

    # ---- Scraping with pagination ----
    async def _scrape_paginated(
        self,
        start_url: str,
        parse_fn,
        author_profile: str,
        max_pages: Optional[int] = None,
    ) -> List[Dict]:
        results: List[Dict] = []
        url = start_url
        pages = 0

        while url and (max_pages is None or pages < max_pages):
            async with self._sem:
                resp = await _fetch_with_retries(self.client, url)

            page_results = parse_fn(resp, author_profile)
            results.extend(page_results)

            sel = Selector(resp.text)
            url = _find_next_url(sel)
            pages += 1

        return results

    async def scrape_user_comments(self, profile_url: str, max_pages: Optional[int] = None) -> List[Dict]:
        base = normalize_profile_url(profile_url)
        url = join_url(base, "comments/")
        return await self._scrape_paginated(url, parse_user_comments, base, max_pages=max_pages)

    async def scrape_user_submitted(self, profile_url: str, max_pages: Optional[int] = None) -> List[Dict]:
        base = normalize_profile_url(profile_url)
        url = join_url(base, "submitted/")
        return await self._scrape_paginated(url, parse_user_submitted, base, max_pages=max_pages)

    async def scrape_user_both(self, profile_url: str, max_pages: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        comments_task = asyncio.create_task(self.scrape_user_comments(profile_url, max_pages=max_pages))
        posts_task = asyncio.create_task(self.scrape_user_submitted(profile_url, max_pages=max_pages))
        comments, posts = await asyncio.gather(comments_task, posts_task)
        return comments, posts

    # ---- Export ----
    def _export_json(self, author: str, author_profile: str, comments: List[Dict], posts: List[Dict]) -> Path:
        payload = {
            "author": author,
            "authorProfile": author_profile,
            "counts": {"comments": len(comments), "posts": len(posts)},
            "comments": comments,
            "posts": posts,
        }
        out_path = self.output_dir / f"{author}_post_comments.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def _export_csv(self, author: str, author_profile: str, comments: List[Dict], posts: List[Dict]) -> Path:
        """Aplati en lignes: une ligne par 'comment' ou 'post' (colonne 'type')."""
        rows: List[Dict] = []

        def keep(keys: List[str], src: Dict) -> Dict:
            return {k: src.get(k) for k in keys}

        post_cols = ["type", "title", "created", "body_text", "link"]


        for c in comments:
            rows.append(keep(post_cols, c))
        for p in posts:
            rows.append(keep(post_cols, p))

        fieldnames = list(dict.fromkeys(post_cols + post_cols))

        out_path = self.output_dir / f"{author}_post_comments.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return out_path

    def export_user(self, author_profile: str, comments: List[Dict], posts: List[Dict]) -> Path:
        author = safe_author_from_profile(author_profile)
        if self.output_format == "json":
            return self._export_json(author, author_profile, comments, posts)
        else:
            return self._export_csv(author, author_profile, comments, posts)

    # ---- Orchestration helpers ----
    async def export_from_json(
        self,
        json_path: Union[str, Path],
        max_pages: Optional[int] = None,
    ) -> List[Path]:
        """
        1) Récupère les profils depuis un JSON (schéma plat ou info/comments/replies).
        2) Scrape comments + submitted en parallèle par user.
        3) Exporte <author>_post_comments.(json|csv) dans output_dir.
        """
        profiles = self.collect_profiles_from_json_file(json_path)
        export_paths: List[Path] = []

        async def _process(profile: str) -> Optional[Path]:
            try:
                comments, posts = await self.scrape_user_both(profile, max_pages=max_pages)
                out = self.export_user(profile, comments, posts)
                log.success(f"Exported {out.name} ({len(comments)} comments, {len(posts)} posts)")
                return out
            except Exception as e:
                log.error(f"Failed scraping/export for {profile}: {e}")
                return None

        tasks = [asyncio.create_task(_process(p)) for p in profiles]
        results = await asyncio.gather(*tasks)
        for r in results:
            if r:
                export_paths.append(r)

        return export_paths

    async def aclose(self):
        await self.client.aclose()
