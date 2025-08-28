from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Literal

from httpx import AsyncClient, Response, HTTPError, RequestError, TimeoutException
from parsel import Selector
from loguru import logger as log


# -------------------------
# --- Parsing helpers   ---
# -------------------------

def parse_modern_listing(response: Response) -> Dict:
    """
    Parse la page moderne d'un subreddit (vue /r/{sub} standard).
    Retourne { "info": {...}, "post_data": [...], "cursor": <str|None> }
    """
    sel = Selector(response.text)
    url = str(response.url)

    info: Dict = {}
    info["id"] = url.split("/r")[-1]
    members = sel.xpath("//shreddit-subreddit-header/@subscribers").get()
    rank = sel.xpath("//strong[@id='position']/*/@number").get()
    info["members"] = int(members) if members else None
    info["rank"] = int(rank) if rank else None
    info["url"] = url

    posts: List[Dict] = []
    for box in sel.xpath("//article"):
        link = box.xpath(".//a/@href").get()
        author = box.xpath(".//shreddit-post/@author").get()
        post_label = box.xpath(".//faceplate-tracker[@source='post']/a/span/div/text()").get()
        upvotes = box.xpath(".//shreddit-post/@score").get()
        comment_count = box.xpath(".//shreddit-post/@comment-count").get()
        attachment_type = box.xpath(".//shreddit-post/@post-type").get()

        if attachment_type == "image":
            attachment_link = box.xpath(".//div[@slot='thumbnail']/*/*/@src").get()
        elif attachment_type == "video":
            attachment_link = box.xpath(".//shreddit-player/@preview").get()
        else:
            attachment_link = box.xpath(".//div[@slot='thumbnail']/a/@href").get()

        posts.append({
            "authorProfile": f"https://www.reddit.com/user/{author}" if author else None,
            "authorId": box.xpath(".//shreddit-post/@author-id").get(),
            "title": box.xpath("./@aria-label").get(),
            "link": f"https://www.reddit.com{link}" if link else None,
            "publishingDate": box.xpath(".//shreddit-post/@created-timestamp").get(),
            "postId": box.xpath(".//shreddit-post/@id").get(),
            "postLabel": post_label.strip() if post_label else None,
            "postUpvotes": int(upvotes) if upvotes else None,
            "commentCount": int(comment_count) if comment_count else None,
            "attachmentType": attachment_type,
            "attachmentLink": attachment_link,
        })

    cursor_id = sel.xpath("//shreddit-post/@more-posts-cursor").get()
    return {"info": info, "post_data": posts, "cursor": cursor_id}


def parse_old_search(response: Response) -> Dict:
    """
    Parse une page de recherche old.reddit (old.reddit.com/r/{sub}/search?...).
    Retourne { "info": {...}, "post_data": [...], "next_url": <str|None> }
    """
    sel = Selector(response.text)
    url = str(response.url)

    info: Dict = {"id": url.split("/r")[-1], "members": None, "url": url}
    members = sel.xpath('//span[@class="subscribers"]/span[@class="number"]/text()').get()
    if members:
        try:
            info["members"] = int(members.replace(",", "").strip())
        except ValueError:
            pass

    posts: List[Dict] = []
    for box in sel.xpath('//div[contains(@class,"search-result")]'):
        link = box.xpath('.//a[contains(@class,"search-title")]/@href').get()
        title = box.xpath('.//a[contains(@class,"search-title")]/text()').get()
        author = box.xpath('.//span[contains(@class,"search-author")]/a/text()').get()
        upvotes_txt = box.xpath('.//span[contains(@class,"search-score")]/text()').get()
        comments_txt = box.xpath('.//a[contains(@class,"search-comments")]/text()').re_first(r'\d+')

        upvotes = None
        if upvotes_txt:
            try:
                upvotes = int(upvotes_txt.split()[0])
            except Exception:
                pass

        posts.append({
            "authorProfile": f"https://www.reddit.com/user/{author}" if author else None,
            "authorId": None,  # pas fiable sur old.reddit search
            "title": title,
            "link": link,
            "publishingDate": None,
            "postId": None,
            "postLabel": "Discussion",
            "postUpvotes": upvotes,
            "commentCount": int(comments_txt) if comments_txt else None,
            "attachmentType": None,
            "attachmentLink": None,
        })

    next_url = sel.xpath('//span[@class="next-button"]/a/@href').get()
    return {"info": info, "post_data": posts, "next_url": next_url}


# -------------------------
# --- Scraper unifié    ---
# -------------------------

async def scrape_subreddit(
    subreddit_id: str,
    sort: Literal["new", "hot", "old"] = "new",
    query: Optional[str] = None,
    max_posts: int = 50,          # acts as the limit
    delay_s: float = 0.5,
    old_reddit: bool = False,
) -> Dict:
    """
    Récupère jusqu'à `max_posts` posts avec un SEUL flux de contrôle.
    - old_reddit=True  -> utilise old.reddit + bouton Next (nécessite query)
    - old_reddit=False -> utilise la vue moderne + cursor
    -> Ajoute chaque post seulement si son 'link' n'a pas déjà été vu.
    """

    def _norm_link(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        # Normalisation simple pour éviter les faux doublons:
        u = url.strip()
        if u.endswith("/"):
            u = u[:-1]
        return u

    async with AsyncClient(
        http2=True,
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Cookie": "intl_splash=false",
        },
        follow_redirects=True,
        timeout=30.0,
    ) as client:

        # ---------- Sélection de la stratégie ----------
        if old_reddit:
            if not query:
                raise ValueError("old_reddit=True nécessite un `query` (old.reddit/search).")

            first_url = (
                f"https://old.reddit.com/r/{subreddit_id}/search/"
                f"?q={query}&sort={sort}&restrict_sr=on&limit=50"
            )

            async def fetch_first():
                resp = await client.get(first_url)
                page = parse_old_search(resp)  # -> info, post_data, next_url
                return {"info": page["info"], "posts": page["post_data"], "cursor_like": page["next_url"]}

            async def fetch_next(cursor_like: Optional[str]):
                if not cursor_like:
                    return {"posts": [], "cursor_like": None}
                resp = await client.get(cursor_like)
                page = parse_old_search(resp)
                return {"posts": page["post_data"], "cursor_like": page["next_url"]}

        else:
            base_url = f"https://www.reddit.com/r/{subreddit_id}"

            async def fetch_first():
                resp = await client.get(base_url, params={"sort": sort})
                page = parse_modern_listing(resp)  # -> info, post_data, cursor
                return {"info": page["info"], "posts": page["post_data"], "cursor_like": page["cursor"]}

            def make_pagination_url(cursor_id: str) -> str:
                return (
                    f"https://www.reddit.com/svc/shreddit/community-more-posts/{sort}/"
                    f"?after={cursor_id}%3D%3D&t=DAY&name={subreddit_id}&feedLength=3&sort={sort}"
                )

            async def fetch_next(cursor_like: Optional[str]):
                if not cursor_like:
                    return {"posts": [], "cursor_like": None}
                resp = await client.get(make_pagination_url(cursor_like))
                page = parse_modern_listing(resp)
                return {"posts": page["post_data"], "cursor_like": page["cursor"]}

        # ---------- Boucle unifiée avec déduplication ----------
        try:
            first_page = await fetch_first()
        except (HTTPError, RequestError, TimeoutException) as e:
            log.error(f"Initial fetch failed: {e}")
            url_fallback = first_url if old_reddit else base_url
            return {"info": {"id": subreddit_id, "members": None, "url": url_fallback}, "posts": []}

        info = first_page.get("info") or {"id": subreddit_id, "members": None}
        posts: List[Dict] = []
        seen_links: set[str] = set()

        def _append_unique(batch: List[Dict]) -> None:
            """Ajoute seulement les posts avec un link unique ET des commentaires > 0"""
            nonlocal posts, seen_links
            for p in batch or []:
                link = _norm_link(p.get("link"))
                if not link:
                    continue
                if link in seen_links:
                    continue
                # filtre: pas de post si 0 commentaires
                if not p.get("commentCount") or p["commentCount"] == 0:
                    continue
                seen_links.add(link)
                posts.append(p)
                if len(posts) >= max_posts:
                    break

        # Ajout du premier lot (dédupliqué)
        _append_unique(first_page.get("posts") or [])
        cursor_like = first_page.get("cursor_like")

        while cursor_like and len(posts) < max_posts:
            try:
                page = await fetch_next(cursor_like)
                _append_unique(page.get("posts") or [])
                cursor_like = page.get("cursor_like")
                if len(posts) >= max_posts or not cursor_like:
                    break
                await asyncio.sleep(delay_s)
            except (HTTPError, RequestError, TimeoutException) as e:
                log.warning(f"Pagination fetch failed: {e}")
                break

        log.success(f"scraped {len(posts)} unique posts from r/{subreddit_id}{' (old search)' if old_reddit else ''}")
        return {"info": info, "posts": posts}



# -------------------------
# --- Exemple d'usage   ---
# -------------------------

# if __name__ == "__main__":
#     import json

#     async def main():
#         # Exemple 1 : recherche (old.reddit) — va suivre "Next" jusqu'à max_posts
#         data = await scrape_subreddit(
#             subreddit_id="salesforce",
#             sort="new",
#             query="agentforce",
#             max_posts=30,
#         )
#         with open("results/subreddit_search.json", "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

#         # Exemple 2 : listing moderne (sans recherche) — va suivre les cursors
#         data2 = await scrape_subreddit(
#             subreddit_id="salesforce",
#             sort="hot",
#             query=None,
#             max_posts=60,
#         )
#         with open("results/subreddit_listing.json", "w", encoding="utf-8") as f:
#             json.dump(data2, f, indent=2, ensure_ascii=False)

#     asyncio.run(main())
