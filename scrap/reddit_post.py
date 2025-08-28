from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional, Literal

from httpx import AsyncClient, Response
from parsel import Selector
from loguru import logger as log
from urllib.parse import urlparse, urlunparse


# =============================
# HTTP client durci
# =============================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Cookies pour contourner écrans consentement/NSFW
    "Cookie": "over18=1; csv=1",
}

client = AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True)


# =============================
# URL helpers
# =============================
def _canonical_path(path: str) -> str:
    # Force /r/<sub>/comments/<id>/
    canon = re.sub(r"^/r/([^/]+)/comments/([^/]+).*", r"/r/\1/comments/\2/", path or "")
    if not canon.endswith("/"):
        canon += "/"
    return canon

def to_www_url(url: str) -> str:
    u = urlparse(url)
    return urlunparse((u.scheme or "https", "www.reddit.com", _canonical_path(u.path), "", "", ""))

def to_old_url(url: str) -> str:
    u = urlparse(url)
    return urlunparse((u.scheme or "https", "old.reddit.com", _canonical_path(u.path), "", "", ""))

def extract_post_id(url: str) -> Optional[str]:
    m = re.search(r"/comments/([a-z0-9]+)/", url)
    return m.group(1) if m else None


# =============================
# Post info (new / old / json)
# =============================
def parse_post_info_new(selector: Selector) -> Optional[Dict]:
    if not selector.xpath("//shreddit-post"):
        return None
    get = selector.xpath

    def _i(s: Optional[str]) -> Optional[int]:
        return int(s) if s and s.isdigit() else None

    msg_parts = get('//div[@property="schema:articleBody"]//p//text()').getall()
    author = get("//shreddit-post/@author").get()

    info = {
        "authorId": get("//shreddit-post/@author-id").get(),
        "author": author,
        "authorProfile": f"https://www.reddit.com/user/{author}" if author else None,
        "subreddit": get("//shreddit-post/@subreddit-prefixed-name").get(),
        "postId": get("//shreddit-post/@id").get() or get("//shreddit-post/@post-id").get(),
        "postLabel": (get("//faceplate-tracker[@source='post']//a//text()").get() or None),
        "publishingDate": get("//shreddit-post/@created-timestamp").get(),
        "postTitle": get("//shreddit-post/@post-title").get(),
        "postMessage": " ".join(t.strip() for t in msg_parts if t and t.strip()),
        "postLink": get("//shreddit-canonical-url-updater/@value").get(),
        "commentCount": _i(get("//shreddit-post/@comment-count").get()),
        "upvoteCount": _i(get("//shreddit-post/@score").get()),
        "attachmentType": get("//shreddit-post/@post-type").get(),
        "attachmentLink": get("//shreddit-post/@content-href").get(),
    }
    if info["postLabel"]:
        info["postLabel"] = info["postLabel"].strip()
    return info

def parse_post_info_old(selector: Selector) -> Optional[Dict]:
    thing = selector.xpath("//div[contains(@class,'thing') and @data-fullname]")
    if not thing:
        return None

    def _join(nodes: List[str]) -> str:
        return " ".join(t.strip() for t in nodes if isinstance(t, str) and t.strip())

    score = selector.xpath("//div[contains(@class,'score')]/@title").get() or \
            selector.xpath("//div[contains(@class,'score')]/text()").re_first(r"\d+")
    comments = selector.xpath("//a[contains(@class,'comments')]/text()").re_first(r"(\d+)")
    paras = selector.xpath("//div[contains(@class,'expando')]//div[@class='md']//text()").getall()

    author = thing.xpath("./@data-author").get()
    info = {
        "authorId": thing.xpath("./@data-author-fullname").get(),
        "author": author,
        "authorProfile": f"https://www.reddit.com/user/{author}" if author else None,
        "subreddit": "r/" + (thing.xpath("./@data-subreddit").get() or ""),
        "postId": thing.xpath("./@data-fullname").get(),  # ex: t3_xxxxx
        "postLabel": selector.xpath("//span[contains(@class,'linkflairlabel')]/text()").get(),
        "publishingDate": selector.xpath("//time/@datetime").get(),
        "postTitle": selector.xpath("//a[contains(@class,'title')]/text()").get(),
        "postMessage": _join(paras),
        "postLink": selector.xpath("//link[@rel='canonical']/@href").get(),
        "commentCount": int(comments) if comments else None,
        "upvoteCount": int(score) if score and score.isdigit() else None,
        "attachmentType": None,
        "attachmentLink": None,
    }
    return info

async def parse_post_info_json(post_id: str) -> Optional[Dict]:
    url = f"https://www.reddit.com/comments/{post_id}.json"
    resp = await client.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    try:
        j = resp.json()
        post = j[0]["data"]["children"][0]["data"]
        author = post.get("author")
        return {
            "authorId": None,
            "author": author,
            "authorProfile": f"https://www.reddit.com/user/{author}" if author else None,
            "subreddit": "r/" + (post.get("subreddit") or ""),
            "postId": "t3_" + (post.get("id") or ""),
            "postLabel": post.get("link_flair_text") or None,
            "publishingDate": post.get("created_utc"),
            "postTitle": post.get("title"),
            "postMessage": post.get("selftext") or "",
            "postLink": "https://www.reddit.com" + (post.get("permalink") or ""),
            "commentCount": post.get("num_comments"),
            "upvoteCount": post.get("score"),
            "attachmentType": "image" if post.get("post_hint") == "image" else post.get("post_hint"),
            "attachmentLink": post.get("url_overridden_by_dest") or post.get("url"),
        }
    except Exception as e:
        log.warning(f"JSON fallback parse error: {e}")
        return None

async def parse_post_info_any(url: str) -> Dict:
    # 1) new reddit
    resp = await client.get(to_www_url(url))
    info = parse_post_info_new(Selector(resp.text))
    if info:
        return info
    # 2) old reddit
    resp2 = await client.get(to_old_url(url))
    info = parse_post_info_old(Selector(resp2.text))
    if info:
        return info
    # 3) JSON
    pid = extract_post_id(url)
    if pid:
        info = await parse_post_info_json(pid)
        if info:
            return info
    # 4) défaut
    log.warning("Aucune info de post trouvée via new/old/json.")
    return {
        "authorId": None, "author": None, "authorProfile": None, "subreddit": None,
        "postId": None, "postLabel": None, "publishingDate": None, "postTitle": None,
        "postMessage": "", "postLink": None, "commentCount": None, "upvoteCount": None,
        "attachmentType": None, "attachmentLink": None
    }


# =============================
# Comments: HTML (old) + JSON
# =============================
def _text_join(parts: List[str]) -> str:
    return " ".join(t.strip() for t in parts if isinstance(t, str) and t.strip())

def parse_post_comments_html(response: Response) -> List[Dict]:
    """
    Parse HTML old.reddit (structure et sélecteurs robustifiés).
    """
    sel = Selector(response.text)

    # Racines : variantes fréquentes
    roots = sel.xpath("//div[@class='sitetable nestedlisting']/div[@data-type='comment']")
    if not roots:
        roots = sel.xpath("//div[@id='siteTable']/div[@data-type='comment'] | //div[contains(@class,'sitetable')]/div[@data-type='comment']")
    if not roots:
        log.warning("Aucun commentaire racine trouvé (old.reddit) - DOM alternatif ou interstitiel.")
        return []

    def parse_comment(node: Selector) -> Dict:
        author = node.xpath("./@data-author").get()
        link = node.xpath("./@data-permalink").get()
        body = _text_join(node.xpath(".//div[@class='md']//text()").getall())

        def _safe_int(s: Optional[str]) -> Optional[int]:
            return int(s) if (s and s.isdigit()) else None

        dislikes = node.xpath(".//span[contains(@class, 'dislikes')]/@title").get()
        upvotes  = node.xpath(".//span[contains(@class, 'likes')]/@title").get()
        unvoted  = node.xpath(".//span[contains(@class, 'unvoted')]/@title").get()

        item = {
            "authorId": node.xpath("./@data-author-fullname").get(),
            "author": author,
            "authorProfile": f"https://www.reddit.com/user/{author}" if author else None,
            "commentId": node.xpath("./@data-fullname").get(),
            "link": f"https://www.reddit.com{link}" if link else None,
            "publishingDate": node.xpath(".//time/@datetime").get(),
            "commentBody": body,
            "upvotes": _safe_int(upvotes),
            "dislikes": _safe_int(dislikes),
            "downvotes": _safe_int(unvoted),
        }

        # sous-réponses
        children_nodes = node.xpath(".//div[@data-type='comment']")
        replies: List[Dict] = []
        for child in children_nodes:
            c = {
                "authorId": child.xpath("./@data-author-fullname").get(),
                "author": child.xpath("./@data-author").get(),
                "authorProfile": f"https://www.reddit.com/user/{child.xpath('./@data-author').get()}" if child.xpath("./@data-author").get() else None,
                "commentId": child.xpath("./@data-fullname").get(),
                "link": f"https://www.reddit.com{child.xpath('./@data-permalink').get()}" if child.xpath("./@data-permalink").get() else None,
                "publishingDate": child.xpath(".//time/@datetime").get(),
                "commentBody": _text_join(child.xpath(".//div[@class='md']//text()").getall()),
                "upvotes": _safe_int(child.xpath(".//span[contains(@class, 'likes')]/@title").get()),
                "dislikes": _safe_int(child.xpath(".//span[contains(@class, 'dislikes')]/@title").get()),
                "downvotes": _safe_int(child.xpath(".//span[contains(@class, 'unvoted')]/@title").get()),
            }
            replies.append(c)
        if replies:
            item["replies"] = replies
        return item

    out: List[Dict] = []
    for node in roots:
        out.append(parse_comment(node))
    log.info(f"parse_post_comments_html: {len(out)} commentaires racine parsés (HTML).")
    return out

async def parse_post_comments_json(post_id: str) -> List[Dict]:
    """
    Fallback JSON: https://www.reddit.com/comments/{post_id}.json
    """
    url = f"https://www.reddit.com/comments/{post_id}.json?limit=500"
    resp = await client.get(url, headers=HEADERS)
    if resp.status_code != 200:
        log.warning(f"JSON comments fetch failed: {resp.status_code} for {url}")
        return []
    try:
        data = resp.json()
    except Exception as e:
        log.warning(f"JSON comments parse error: {e}")
        return []

    if not isinstance(data, list) or len(data) < 2:
        return []

    def map_comment(d: Dict) -> Dict:
        author = d.get("author")
        perma = d.get("permalink")
        item = {
            "authorId": None,
            "author": author,
            "authorProfile": f"https://www.reddit.com/user/{author}" if author else None,
            "commentId": f"t1_{d.get('id')}" if d.get("id") else None,
            "link": f"https://www.reddit.com{perma}" if perma else None,
            "publishingDate": d.get("created_utc"),
            "commentBody": d.get("body") or "",
            "upvotes": d.get("score"),
            "dislikes": None,
            "downvotes": None,
        }
        replies = d.get("replies")
        if replies and isinstance(replies, dict):
            children = replies.get("data", {}).get("children", [])
            reps = []
            for c in children:
                if c.get("kind") == "t1":
                    reps.append(map_comment(c.get("data", {})))
            if reps:
                item["replies"] = reps
        return item

    out: List[Dict] = []
    children = data[1].get("data", {}).get("children", [])
    for c in children:
        if c.get("kind") == "t1":
            out.append(map_comment(c.get("data", {})))

    log.info(f"parse_post_comments_json: {len(out)} commentaires racine via JSON.")
    return out


# =============================
# Orchestrateur (avec filtre)
# =============================
async def scrape_post(
    url: str,
    sort: Literal["old", "new", "top"] = "new",
    require_comments: bool = True,
) -> Optional[Dict]:
    """
    - INFO: essaie new→old→json
    - COMMENTS: old.reddit (HTML), puis fallback JSON si HTML vide alors que commentCount>0
    - Si `require_comments=True`, retourne None si aucun commentaire final.
    """
    info = await parse_post_info_any(url)

    # early skip si compteur fiable et = 0
    if require_comments and isinstance(info.get("commentCount"), int) and info["commentCount"] == 0:
        log.info(f"skip (commentCount=0) → {url}")
        return None

    # old.reddit comments
    sort_old = {"new": "new", "top": "top", "old": "old"}.get(sort, "new")
    comments_url = to_old_url(info.get("postLink") or url) + f"?sort={sort_old}&limit=500"
    resp = await client.get(comments_url)
    comments = parse_post_comments_html(resp)

    # Fallback JSON si HTML vide mais compteur > 0
    if (not comments) and isinstance(info.get("commentCount"), int) and info["commentCount"] > 0:
        pid = extract_post_id(info.get("postLink") or url)
        if pid:
            log.info(f"HTML comments empty; trying JSON fallback for {pid}")
            comments = await parse_post_comments_json(pid)

    # Filtre final
    if require_comments and not comments:
        log.info(f"skip (no comments after fallbacks) → {url}")
        return None

    log.success(f"scraped {len(comments)} comments from {comments_url}")
    return {"info": info, "comments": comments}


# =============================
# Exemple rapide
# =============================
# async def _demo():
#     data = await scrape_post(
#         "https://www.reddit.com/r/salesforce/comments/1mqvhmy/need_suggestions_to_upskill_in_salesforce/",
#         sort="new",
#         require_comments=True,
#     )
#     print(data)
# asyncio.run(_demo())
