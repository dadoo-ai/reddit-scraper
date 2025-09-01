from __future__ import annotations

import re
from typing import Dict, List, Optional, Literal
from abc import ABC, abstractmethod
import json
import os
from urllib.parse import urlparse
from httpx import AsyncClient, Response
from parsel import Selector
from loguru import logger as log
from urllib.parse import urlunparse
import random
import datetime

# =============================
# HTTP Client
# =============================
class RedditHttpClient:
    """Client HTTP pour Reddit avec configuration par défaut"""
    
    def __init__(self):
        # Charger les User-Agents depuis le fichier JSON
        try:
            with open("settings/user-agents.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.user_agents = data["user-agent"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            log.warning(f"Impossible de charger user-agents.json: {e}. Utilisation d'un fallback.")
            # Fallback si le fichier n'existe pas
            self.user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ]
        
        self.base_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            # Cookies pour contourner écrans consentement/NSFW
            "Cookie": "over18=1; csv=1",
        }
        self.client = AsyncClient(timeout=30, follow_redirects=True)
    
    def _get_random_headers(self) -> Dict[str, str]:
        """Génère des headers avec un User-Agent aléatoire"""
        headers = self.base_headers.copy()
        headers["User-Agent"] = random.choice(self.user_agents)
        return headers
    
    async def get(self, url: str, **kwargs) -> Response:
        """Effectue une requête GET avec rotation de User-Agent"""
        try:
            # Utiliser des headers avec User-Agent aléatoire pour chaque requête
            headers = self._get_random_headers()
            return await self.client.get(url, headers=headers, **kwargs)
        except Exception as e:
            log.error(f"Error getting {url}: {e}")
            return None


# =============================
# URL Utilities
# =============================
class UrlHelper:
    """Utilitaires pour la manipulation d'URLs Reddit"""
    
    @staticmethod
    def _canonical_path(path: str) -> str:
        """Force le format /r/<sub>/comments/<id>/"""
        canon = re.sub(r"^/r/([^/]+)/comments/([^/]+).*", r"/r/\1/comments/\2/", path or "")
        if not canon.endswith("/"):
            canon += "/"
        return canon
    
    @staticmethod
    def to_www_url(url: str) -> str:
        """Convertit une URL vers www.reddit.com"""
        u = urlparse(url)
        return urlunparse((u.scheme or "https", "www.reddit.com", UrlHelper._canonical_path(u.path), "", "", ""))
    
    @staticmethod
    def to_old_url(url: str) -> str:
        """Convertit une URL vers old.reddit.com"""
        u = urlparse(url)
        return urlunparse((u.scheme or "https", "old.reddit.com", UrlHelper._canonical_path(u.path), "", "", ""))
    
    @staticmethod
    def extract_post_id(url: str) -> Optional[str]:
        """Extrait l'ID du post depuis l'URL"""
        m = re.search(r"/comments/([a-z0-9]+)/", url)
        return m.group(1) if m else None


# =============================
# Post Info Parsers
# =============================
class PostInfoParser(ABC):
    """Classe abstraite pour parser les infos de post"""
    
    @abstractmethod
    def parse(self, selector: Selector) -> Optional[Dict]:
        """Parse les informations du post"""
        pass


class ModernRedditPostParser(PostInfoParser):
    """Parser pour la vue moderne de Reddit"""
    
    def parse(self, selector: Selector) -> Optional[Dict]:
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


class OldRedditPostParser(PostInfoParser):
    """Parser pour old.reddit.com"""
    
    def parse(self, selector: Selector) -> Optional[Dict]:
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


class JsonPostParser(PostInfoParser):
    """Parser pour l'API JSON de Reddit"""
    
    def __init__(self, http_client: RedditHttpClient):
        self.http_client = http_client
    
    async def parse(self, post_id: str) -> Optional[Dict]:
        """Parse les infos via l'API JSON"""
        url = f"https://www.reddit.com/comments/{post_id}.json"
        resp = await self.http_client.get(url)
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


# =============================
# Comment Parsers
# =============================
class CommentParser(ABC):
    """Classe abstraite pour parser les commentaires"""
    
    @abstractmethod
    def parse(self, response: Response) -> List[Dict]:
        """Parse les commentaires"""
        pass


class HtmlCommentParser(CommentParser):
    """Parser pour les commentaires HTML (old.reddit)"""
    
    def parse(self, response: Response) -> List[Dict]:
        print("recupere les commentaires")
        sel = Selector(response.text)
        
        # Racines : variantes fréquentes
        roots = sel.xpath("//div[@class='sitetable nestedlisting']/div[@data-type='comment']")
        if not roots:
            roots = sel.xpath("//div[@id='siteTable']/div[@data-type='comment'] | //div[contains(@class,'sitetable')]/div[@data-type='comment']")
        if not roots:
            log.warning("Aucun commentaire racine trouvé (old.reddit) - DOM alternatif ou interstitiel.")
            return []
        
        out: List[Dict] = []
        for node in roots:
            out.append(self._parse_comment(node))
        
        log.info(f"parse_post_comments_html: {len(out)} commentaires racine parsés (HTML).")
        return out
    
    def _parse_comment(self, node: Selector) -> Dict:
        """Parse un commentaire individuel"""
        author = node.xpath("./@data-author").get()
        link = node.xpath("./@data-permalink").get()
        body = self._text_join(node.xpath(".//div[@class='md']//text()").getall())
        
        def _safe_int(s: Optional[str]) -> Optional[int]:
            return int(s) if (s and s.isdigit()) else None
        
        dislikes = node.xpath(".//span[contains(@class, 'dislikes')]/@title").get()
        upvotes = node.xpath(".//span[contains(@class, 'likes')]/@title").get()
        unvoted = node.xpath(".//span[contains(@class, 'unvoted')]/@title").get()
        
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
                "commentBody": self._text_join(child.xpath(".//div[@class='md']//text()").getall()),
                "upvotes": _safe_int(child.xpath(".//span[contains(@class, 'likes')]/@title").get()),
                "dislikes": _safe_int(child.xpath(".//span[contains(@class, 'dislikes')]/@title").get()),
                "downvotes": _safe_int(child.xpath(".//span[contains(@class, 'unvoted')]/@title").get()),
            }
            replies.append(c)
        if replies:
            item["replies"] = replies
        return item
    
    def _text_join(self, parts: List[str]) -> str:
        """Joint les parties de texte"""
        return " ".join(t.strip() for t in parts if isinstance(t, str) and t.strip())


class JsonCommentParser(CommentParser):
    """Parser pour les commentaires via API JSON"""
    
    def __init__(self, http_client: RedditHttpClient):
        self.http_client = http_client
    
    async def parse(self, post_id: str) -> List[Dict]:
        """Parse les commentaires via JSON"""
        url = f"https://www.reddit.com/comments/{post_id}.json"
        resp = await self.http_client.get(url)
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
        
        out: List[Dict] = []
        children = data[1].get("data", {}).get("children", [])
        for c in children:
            if c.get("kind") == "t1":
                out.append(self._map_comment(c.get("data", {})))
        
        log.info(f"parse_post_comments_json: {len(out)} commentaires racine via JSON.")
        return out
    
    def _map_comment(self, d: Dict) -> Dict:
        """Mappe un commentaire JSON vers le format standard"""
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
                    reps.append(self._map_comment(c.get("data", {})))
            if reps:
                item["replies"] = reps
        return item


# =============================
# Error Tracker
# =============================
class ErrorTracker:
    """Système de tracking des erreurs de scraping"""
    
    def __init__(self):
        self.errors = []
    
    async def log_error(self, url: str, reason: str, output_folder: Optional[str] = None):
        """Enregistre une erreur de scraping"""
        
        
        error_entry = {
            "url": url,
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat(),
            "post_id": self._extract_post_id(url)
        }
        
        self.errors.append(error_entry)
        
        # Sauvegarder dans error.json si output_folder est spécifié
        if output_folder:
            # Déterminer le dossier parent (même niveau que post_search.json)
            error_folder = os.path.dirname(output_folder) if output_folder != "results" else "results/posts"
            error_file = os.path.join(error_folder, "error.json")
            
            # Créer le dossier s'il n'existe pas
            os.makedirs(error_folder, exist_ok=True)
            
            # Charger les erreurs existantes
            existing_errors = []
            if os.path.exists(error_file):
                try:
                    with open(error_file, "r", encoding="utf-8") as f:
                        existing_errors = json.load(f)
                except:
                    existing_errors = []
            
            # Ajouter la nouvelle erreur
            existing_errors.append(error_entry)
            
            # Sauvegarder
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(existing_errors, f, indent=2, ensure_ascii=False)
            
            log.warning(f"Erreur enregistrée dans {error_file}: {url} - {reason}")
    
    def _extract_post_id(self, url: str) -> Optional[str]:
        """Extrait l'ID du post depuis l'URL"""
        import re
        m = re.search(r"/comments/([a-z0-9]+)/", url)
        return m.group(1) if m else None


# =============================
# Post Scraper
# =============================
class PostScraper:
    """Scraper principal pour les posts Reddit"""
    
    def __init__(self):
        self.http_client = RedditHttpClient()
        self.url_helper = UrlHelper()
        self.modern_parser = ModernRedditPostParser()
        self.old_parser = OldRedditPostParser()
        self.json_parser = JsonPostParser(self.http_client)
        self.html_comment_parser = HtmlCommentParser()
        self.json_comment_parser = JsonCommentParser(self.http_client)
        self.result_exporter = ResultExporter()
        self.error_tracker = ErrorTracker()
    
    async def scrape_post(
        self,
        url: str,
        output_folder: Optional[str] = None,
        require_comments: bool = True,
    ) -> Optional[Dict]:
        """
        Scrape un post Reddit complet avec ses commentaires
        
        Args:
            url: URL du post Reddit
            output_folder: Dossier où sauvegarder le résultat (optionnel)
            require_comments: Si True, retourne None si aucun commentaire
            
        Returns:
            Dict avec les infos du post et ses commentaires, ou None
        """
        # Récupérer les infos du post
        info = await self._get_post_info(url)
        
        # Early skip si compteur fiable et = 0
        if require_comments and isinstance(info.get("commentCount"), int) and info["commentCount"] == 0:
            log.info(f"skip (commentCount=0) → {url}")
            await self.error_tracker.log_error(url, "commentCount=0", output_folder)
            return None
        
        # Récupérer les commentaires
        comments = await self._get_comments(info, url)
        
        # Filtre final
        if require_comments and not comments:
            log.info(f"skip (no comments after fallbacks) → {url}")
            await self.error_tracker.log_error(url, "no comments after fallbacks", output_folder)
            return None
        
        log.success(f"scraped {len(comments)} comments from {url}")
        results = {"info": info, "comments": comments}
        await self.result_exporter.export(output_folder, results, url, info.get("postId"))
        return results
    
    
    async def _get_post_info(self, url: str) -> Dict:
        """Récupère les infos du post via plusieurs méthodes"""
        # 1) new reddit
        resp = await self.http_client.get(self.url_helper.to_www_url(url))
        info = self.modern_parser.parse(Selector(resp.text))
        if info:
            return info
        
        # 2) old reddit
        resp2 = await self.http_client.get(self.url_helper.to_old_url(url))
        info = self.old_parser.parse(Selector(resp2.text))
        if info:
            return info
        
        # 3) JSON
        pid = self.url_helper.extract_post_id(url)
        if pid:
            info = await self.json_parser.parse(pid)
            if info:
                return info
        
        # 4) défaut
        log.warning(f"Aucune info de post trouvée via new/old/json pour {url}")
        return {
            "authorId": None, "author": None, "authorProfile": None, "subreddit": None,
            "postId": None, "postLabel": None, "publishingDate": None, "postTitle": None,
            "postMessage": "", "postLink": url, "commentCount": None, "upvoteCount": None,
            "attachmentType": None, "attachmentLink": None
        }
    
    async def _get_comments(self, info: Dict, url: str) -> List[Dict]:
        """Récupère les commentaires du post"""
        # old.reddit comments
        comments_url = self.url_helper.to_old_url(info.get("postLink") or url) + ""
        resp = await self.http_client.get(comments_url)
        comments = self.html_comment_parser.parse(resp)
        
        # Fallback JSON si HTML vide mais compteur > 0
        if (not comments) and isinstance(info.get("commentCount"), int) and info["commentCount"] > 0:
            pid = self.url_helper.extract_post_id(info.get("postLink") or url)
            if pid:
                log.info(f"HTML comments empty; trying JSON fallback for {pid}")
                comments = await self.json_comment_parser.parse(pid)
        
        return comments


# =============================
# Interface simplifiée pour compatibilité
# =============================
# Instance globale pour la compatibilité avec l'ancien code


# async def scrape_post(
#     url: str,
#     sort: Literal["old", "new", "top"] = "new",
#     require_comments: bool = True,
#     output_folder: Optional[str] = None,
# ) -> Optional[Dict]:
#     """
#     Interface simplifiée pour la compatibilité avec l'ancien code
    
#     Args:
#         url: URL du post Reddit
#         sort: Ordre de tri des commentaires
#         require_comments: Si True, retourne None si aucun commentaire
#         output_folder: Dossier où sauvegarder le résultat (optionnel)
#     """

    
#     print("url", url)
#     result = await _scraper.scrape_post(url, sort, require_comments)
    
class ResultExporter:
    async def export(self, output_folder: Optional[str], result: Optional[Dict], url: str, post_id: str):
        # Sauvegarder automatiquement si un dossier est spécifié
        if output_folder and result:
            # Extraire l'ID du post pour le nom de fichier
            if not post_id:
                post_id = str(random.randint(10000, 99999))
                # Fallback: utiliser une partie de l'URL
                post_id = urlparse(url).path.split('/')[-2] if urlparse(url).path.split('/') else "unknown"
            
            # Créer le dossier s'il n'existe pas
            os.makedirs(output_folder, exist_ok=True)
            
            # Nom du fichier
            filename = f"post_{post_id}_comments.json"
            filepath = os.path.join(output_folder, filename)
            
            # Sauvegarder
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2, ensure_ascii=False)
            
            print(f"Résultat sauvegardé : {filepath}")
        
        return result


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
