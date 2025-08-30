from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Literal, Protocol

from httpx import AsyncClient, Response, HTTPError, RequestError, TimeoutException
from parsel import Selector
from loguru import logger as log


# -------------------------
# --- Data Models      ---
# -------------------------

@dataclass
class SubredditInfo:
    """Informations sur un subreddit"""
    id: str
    members: Optional[int] = None
    rank: Optional[int] = None
    url: Optional[str] = None

@dataclass
class Post:
    """Représentation d'un post Reddit"""
    author_profile: Optional[str]
    author_id: Optional[str]
    title: Optional[str]
    link: Optional[str]
    publishing_date: Optional[str]
    post_id: Optional[str]
    post_label: Optional[str]
    post_upvotes: Optional[int]
    comment_count: Optional[int]
    attachment_type: Optional[str]
    attachment_link: Optional[str]

@dataclass
class ParsedPage:
    """Résultat du parsing d'une page"""
    info: SubredditInfo
    posts: List[Post]
    cursor: Optional[str] = None
    next_url: Optional[str] = None


# -------------------------
# --- Parsers          ---
# -------------------------

class RedditParser(ABC):
    """Classe abstraite pour parser les pages Reddit"""
    
    @abstractmethod
    def parse(self, response: Response) -> ParsedPage:
        """Parse une réponse HTTP et retourne les données structurées"""
        pass

class ModernRedditParser(RedditParser):
    """Parser pour la vue moderne de Reddit"""
    
    def parse(self, response: Response) -> ParsedPage:
        sel = Selector(response.text)
        url = str(response.url)
        
        # Parse les infos du subreddit
        subreddit_id = url.split("/r")[-1]
        members = sel.xpath("//shreddit-subreddit-header/@subscribers").get()
        rank = sel.xpath("//strong[@id='position']/*/@number").get()
        
        info = SubredditInfo(
            id=subreddit_id,
            members=int(members) if members else None,
            rank=int(rank) if rank else None,
            url=url
        )
        
        # Parse les posts
        posts = []
        for box in sel.xpath("//article"):
            post = self._parse_post(box)
            if post:
                posts.append(post)
        
        cursor_id = sel.xpath("//shreddit-post/@more-posts-cursor").get()
        
        return ParsedPage(info=info, posts=posts, cursor=cursor_id)
    
    def _parse_post(self, box) -> Optional[Post]:
        """Parse un post individuel"""
        link = box.xpath(".//a/@href").get()
        author = box.xpath(".//shreddit-post/@author").get()
        post_label = box.xpath(".//faceplate-tracker[@source='post']/a/span/div/text()").get()
        upvotes = box.xpath(".//shreddit-post/@score").get()
        comment_count = box.xpath(".//shreddit-post/@comment-count").get()
        attachment_type = box.xpath(".//shreddit-post/@post-type").get()
        
        # Parse l'attachment link selon le type
        attachment_link = self._parse_attachment_link(box, attachment_type)
        
        return Post(
            author_profile=f"https://www.reddit.com/user/{author}" if author else None,
            author_id=box.xpath(".//shreddit-post/@author-id").get(),
            title=box.xpath("./@aria-label").get(),
            link=f"https://www.reddit.com{link}" if link else None,
            publishing_date=box.xpath(".//shreddit-post/@created-timestamp").get(),
            post_id=box.xpath(".//shreddit-post/@id").get(),
            post_label=post_label.strip() if post_label else None,
            post_upvotes=int(upvotes) if upvotes else None,
            comment_count=int(comment_count) if comment_count else None,
            attachment_type=attachment_type,
            attachment_link=attachment_link
        )
    
    def _parse_attachment_link(self, box, attachment_type: Optional[str]) -> Optional[str]:
        """Parse le lien d'attachment selon le type"""
        if attachment_type == "image":
            return box.xpath(".//div[@slot='thumbnail']/*/*/@src").get()
        elif attachment_type == "video":
            return box.xpath(".//shreddit-player/@preview").get()
        else:
            return box.xpath(".//div[@slot='thumbnail']/a/@href").get()

class OldRedditParser(RedditParser):
    """Parser pour old.reddit.com"""
    
    def parse(self, response: Response) -> ParsedPage:
        sel = Selector(response.text)
        url = str(response.url)
        
        # Parse les infos du subreddit
        subreddit_id = url.split("/r")[-1]
        members = sel.xpath('//span[@class="subscribers"]/span[@class="number"]/text()').get()
        
        info = SubredditInfo(
            id=subreddit_id,
            members=self._parse_members(members),
            url=url
        )
        
        # Parse les posts
        posts = []
        for box in sel.xpath('//div[contains(@class,"search-result")]'):
            post = self._parse_post(box)
            if post:
                posts.append(post)
        
        # Récupérer le dernier lien de pagination (NEXT) au lieu du premier (PREV)
        next_url = sel.xpath('//span[@class="nextprev"]/a/@href').getall()
        if next_url:
            # Prendre le dernier href qui devrait être "NEXT"
            next_url = next_url[-1]
        
        # Si l'URL est relative, la rendre absolue
        if next_url and next_url.startswith('/'):
            next_url = f"https://old.reddit.com{next_url}"
        
        log.info(f"Found next_url: {next_url}")
        log.info(f"Found {len(posts)} posts on this page")
        
        return ParsedPage(info=info, posts=posts, next_url=next_url)
    
    def _parse_members(self, members_text: Optional[str]) -> Optional[int]:
        """Parse le nombre de membres"""
        if not members_text:
            return None
        try:
            return int(members_text.replace(",", "").strip())
        except ValueError:
            return None
    
    def _parse_post(self, box) -> Optional[Post]:
        """Parse un post individuel"""
        link = box.xpath('.//a[contains(@class,"search-title")]/@href').get()
        title = box.xpath('.//a[contains(@class,"search-title")]/text()').get()
        author = box.xpath('.//span[contains(@class,"search-author")]/a/text()').get()
        upvotes_txt = box.xpath('.//span[contains(@class,"search-score")]/text()').get()
        comments_txt = box.xpath('.//a[contains(@class,"search-comments")]/text()').re_first(r'\d+')
        
        upvotes = self._parse_upvotes(upvotes_txt)
        
        return Post(
            author_profile=f"https://www.reddit.com/user/{author}" if author else None,
            author_id=None,  # pas fiable sur old.reddit search
            title=title,
            link=link,
            publishing_date=None,
            post_id=None,
            post_label="Discussion",
            post_upvotes=upvotes,
            comment_count=int(comments_txt) if comments_txt else None,
            attachment_type=None,
            attachment_link=None
        )
    
    def _parse_upvotes(self, upvotes_text: Optional[str]) -> Optional[int]:
        """Parse le nombre d'upvotes"""
        if not upvotes_text:
            return None
        try:
            return int(upvotes_text.split()[0])
        except Exception:
            return None


# -------------------------
# --- HTTP Client      ---
# -------------------------

class RedditClient:
    """Client HTTP pour Reddit avec configuration par défaut"""
    
    def __init__(self):
        self.headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Cookie": "intl_splash=false",
        }
    
    async def get(self, url: str, **kwargs) -> Response:
        """Effectue une requête GET avec la configuration par défaut"""
        async with AsyncClient(
            http2=True,
            headers=self.headers,
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            return await client.get(url, **kwargs)


# -------------------------
# --- Deduplication   ---
# -------------------------

class PostFilter:
    """Gère la déduplication des posts basée sur les liens"""
    
    def __init__(self, query: Optional[str] = None):
        self.seen_links: set[str] = set()
        self.query: str = query
    
    def normalize_link(self, url: Optional[str]) -> Optional[str]:
        """Normalise un URL pour éviter les faux doublons"""
        if not url:
            return None
        u = url.strip()
        if u.endswith("/"):
            u = u[:-1]
        return u
    
    def is_unique(self, post: Post) -> bool:
        """Vérifie si un post est unique et valide"""
        link = self.normalize_link(post.link)
        if not link:
            return False
        
        if link in self.seen_links:
            return False
        
        # Filtre: pas de post si 0 commentaires
        if not post.comment_count or post.comment_count == 0:
            return False
        
        return True
    
    def post_contains_query(self, post: Post, query: str) -> bool:
        """Vérifie si le titre du post contient la requête"""
        
        # Si pas de requête, accepter tous les posts
        if not query:
            return True
            
        if not post.title:
            return False
        title = re.sub(r'[^a-z0-9]+', '', post.title.lower())
        q = re.sub(r'[^a-z0-9]+', '', query.lower())
        return q in title
    
    def add_post(self, post: Post) -> bool:
        """Ajoute un post s'il est unique, retourne True si ajouté"""
        if not self.is_unique(post):
            return False
        
        if not self.post_contains_query(post, self.query):
            return False
        
        link = self.normalize_link(post.link)
        self.seen_links.add(link)
        return True


# -------------------------
# --- Scraping Strategy ---
# -------------------------

class ScrapingStrategy(ABC):
    """Stratégie abstraite pour le scraping"""
    
    def __init__(self, client: RedditClient, parser: RedditParser):
        self.client = client
        self.parser = parser
    
    @abstractmethod
    async def fetch_first_page(self, subreddit_id: str, **kwargs) -> ParsedPage:
        """Récupère la première page"""
        pass
    
    @abstractmethod
    async def fetch_next_page(self, cursor: Optional[str], **kwargs) -> ParsedPage:
        """Récupère la page suivante"""
        pass

# class ModernRedditStrategy(ScrapingStrategy):
#     """Stratégie pour la vue moderne de Reddit"""
    
#     async def fetch_first_page(self, subreddit_id: str, sort: str = "relevance", query: str = "", **kwargs) -> ParsedPage:
#         url = f"https://www.reddit.com/r/{subreddit_id}"
#         response = await self.client.get(url, params={"sort": sort, "q": query, "type": "posts"})
#         print(response.url)
#         return self.parser.parse(response)
    
#     async def fetch_next_page(self, cursor: Optional[str], subreddit_id: str, sort: str = "relevance", **kwargs) -> ParsedPage:
#         if not cursor:
#             return ParsedPage(info=SubredditInfo(id=subreddit_id), posts=[])
        
#         url = (
#             f"https://www.reddit.com/svc/shreddit/community-more-posts/{sort}/"
#             f"?after={cursor}%3D%3D&t=DAY&name={subreddit_id}&feedLength=3&sort={sort}"
#         )
#         response = await self.client.get(url)
#         return self.parser.parse(response)

class OldRedditStrategy(ScrapingStrategy):
    """Stratégie pour old.reddit.com"""
    
    async def fetch_first_page(self, subreddit_id: str, sort: str = "relevance", query: str = "", **kwargs) -> ParsedPage:
        url = (
            f"https://old.reddit.com/r/{subreddit_id}/search/"
            f"?q={query}&sort={sort}&type=posts&limit=50&restrict_sr=on"
        )
        print(url)
        response = await self.client.get(url)
        return self.parser.parse(response)
    
    async def fetch_next_page(self, cursor: Optional[str], **kwargs) -> ParsedPage:
        log.info(f"Fetching next page: {cursor}")
        if not cursor:
            log.warning("No cursor provided, returning empty page")
            return ParsedPage(info=SubredditInfo(id=""), posts=[])
        
        response = await self.client.get(cursor)
        log.info(f"Response status: {response.status_code}")
        return self.parser.parse(response)


# -------------------------
# --- Main Scraper     ---
# -------------------------

class SubredditScraper:
    """Scraper principal pour les subreddits"""
    
    def __init__(
        self, 
        max_posts: int,
        delay_s: float = 0.5,
        subreddit_id: str = "",
        sort: Literal["new", "hot", "old", "relevance"] = "relevance",
        query: Optional[str] = None,
        old_reddit: bool = False
    ):
        self.delay_s = delay_s
        self.subreddit_id = subreddit_id
        self.sort = sort
        self.query = query
        self.max_posts = max_posts
        self.old_reddit = old_reddit
        self.post_filter = PostFilter(query=query)
        self.client = RedditClient()

    
    async def scrape(self) -> Dict:
        """
        Scrape un subreddit avec la stratégie appropriée
        """
        strategy: ScrapingStrategy = OldRedditStrategy(client=self.client, parser=OldRedditParser())

        # # Sélection de la stratégie
        # if old_reddit:
        #     # if not query:
        #     #     raise ValueError("old_reddit=True nécessite un `query` (old.reddit/search).")
        #     strategy: ScrapingStrategy = OldRedditStrategy(client=self.client, parser=OldRedditParser())
        # else:
        #     strategy: ScrapingStrategy = ModernRedditStrategy(client=self.client, parser=ModernRedditParser())
        
        # Reset du filtre
        self.post_filter = PostFilter(query=self.query)
        
        # Récupération de la première page
        try:
            first_page: ParsedPage = await strategy.fetch_first_page(
                subreddit_id=self.subreddit_id,
                sort=self.sort,
                query=self.query
            )
        except (HTTPError, RequestError, TimeoutException) as e:
            log.error(f"Initial fetch failed: {e}")
            url_fallback = f"https://{'old.' if self.old_reddit else ''}reddit.com/r/{self.subreddit_id}"
            return {
                "info": {"id": self.subreddit_id, "members": None, "url": url_fallback},
                "posts": []
            }
        
        # Traitement des posts
        posts = []
        self._add_unique_posts(first_page.posts, posts, self.max_posts)
        
        # Pagination
        cursor = first_page.cursor or first_page.next_url
        log.info(f"Starting pagination with cursor: {cursor}")
        
        while cursor and len(posts) < self.max_posts:
            print("len(posts)", len(posts))
            print("self.max_posts", self.max_posts)
            try:
                log.info(f"Fetching next page with cursor: {cursor}")
                next_page: ParsedPage = await strategy.fetch_next_page(
                    cursor=cursor,
                    subreddit_id=self.subreddit_id,
                    sort=self.sort
                )
                
                log.info(f"Next page fetched, found {len(next_page.posts)} posts")
                
                # Vérifier combien de posts on peut encore ajouter
                remaining_slots = self.max_posts - len(posts)
                if remaining_slots <= 0:
                    log.info(f"Max posts reached ({len(posts)}), stopping pagination")
                    break
                
                self._add_unique_posts(next_page.posts, posts, self.max_posts)
                log.info(f"Total posts after filtering: {len(posts)}")
                
                cursor = next_page.cursor or next_page.next_url
                log.info(f"New cursor: {cursor}")
                
                # Vérifier si on a atteint max_posts après ajout
                if len(posts) >= self.max_posts:
                    log.info(f"Max posts reached ({len(posts)}), stopping pagination")
                    break
                    
                if not cursor:
                    log.info("No more cursor available, stopping pagination")
                    break
                    
                await asyncio.sleep(self.delay_s)
                
            except (HTTPError, RequestError, TimeoutException) as e:
                log.warning(f"Pagination fetch failed: {e}")
                break
        
        # Conversion des posts en dict pour compatibilité
        posts_dict = [self._post_to_dict(post) for post in posts]
        info_dict = self._info_to_dict(first_page.info)
        
        log.success(f"scraped {len(posts)} unique posts from r/{self.subreddit_id}{' (old search)' if self.old_reddit else ''}")
        return {"info": info_dict, "posts": posts_dict}
    
    def _add_unique_posts(self, new_posts: List[Post], posts_list: List[Post], max_posts: int) -> None:
        """Ajoute les posts uniques à la liste jusqu'à max_posts"""
        for post in new_posts:
            # Vérifier si on a déjà atteint la limite
            if len(posts_list) >= max_posts:
                log.debug(f"Max posts limit reached ({max_posts}), stopping post addition")
                break
                
            if self.post_filter.add_post(post):
                posts_list.append(post)
                log.debug(f"Added post {len(posts_list)}/{max_posts}")
    
    def _post_to_dict(self, post: Post) -> Dict:
        """Convertit un Post en dictionnaire"""
        return {
            "authorProfile": post.author_profile,
            "authorId": post.author_id,
            "title": post.title,
            "link": post.link,
            "publishingDate": post.publishing_date,
            "postId": post.post_id,
            "postLabel": post.post_label,
            "postUpvotes": post.post_upvotes,
            "commentCount": post.comment_count,
            "attachmentType": post.attachment_type,
            "attachmentLink": post.attachment_link,
        }
    
    def _info_to_dict(self, info: SubredditInfo) -> Dict:
        """Convertit SubredditInfo en dictionnaire"""
        return {
            "id": info.id,
            "members": info.members,
            "rank": info.rank,
            "url": info.url,
        }


# -------------------------
# --- Interface simplifiée ---
# -------------------------

async def scrape_subreddit(
    max_posts: int,
    subreddit_id: str,
    sort: Literal["new", "hot", "old", "relevance"] = "relevance",
    query: Optional[str] = None,
    delay_s: float = 0.5,
    old_reddit: bool = True,
) -> Dict:
    """
    Interface simplifiée pour la compatibilité avec l'ancien code
    """
    scraper = SubredditScraper(
        delay_s=delay_s,
        subreddit_id=subreddit_id,
        sort=sort,
        query=query,
        max_posts=max_posts,
        old_reddit=old_reddit
    )
    return await scraper.scrape()

