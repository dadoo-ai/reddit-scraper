from typing import List, Dict, Literal
from httpx import AsyncClient, Response
from parsel import Selector
from loguru import logger as log

# initialize an async httpx client
client = AsyncClient(
    # enable http2
    http2=True,
    # add basic browser like headers to prevent getting blocked
    headers={
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Cookie": "intl_splash=false"
    },
    follow_redirects=True
)

def parse_subreddit(response: Response, from_search: bool = False) -> List[Dict]:
    """parse article data from HTML"""
    
    selector = Selector(response.text)
    url = str(response.url)
    info = {}
    info["id"] = url.split("/r")[-1]
    # info["description"] = selector.xpath("//shreddit-subreddit-header/@description").get()
    members = selector.xpath("//shreddit-subreddit-header/@subscribers").get()
    rank = selector.xpath("//strong[@id='position']/*/@number").get()    
    info["members"] = int(members) if members else None
    info["rank"] = int(rank) if rank else None
    info["bookmarks"] = {}

    for item in selector.xpath("//div[faceplate-tracker[@source='community_menu']]/faceplate-tracker"):
        name = item.xpath(".//a/span/span/span/text()").get()
        link = item.xpath(".//a/@href").get()
        info["bookmarks"][name] = link
    
    info["url"] = url
    post_data:List[Dict] = []
    for box in selector.xpath("//article"):
        link = box.xpath(".//a/@href").get()
        author = box.xpath(".//shreddit-post/@author").get()
        post_label = box.xpath(".//faceplate-tracker[@source='post']/a/span/div/text()").get()
        upvotes = box.xpath(".//shreddit-post/@score").get()
        comment_count = box.xpath(".//shreddit-post/@comment-count").get()
        attachment_type = box.xpath(".//shreddit-post/@post-type").get()
        if attachment_type and attachment_type == "image":
            attachment_link = box.xpath(".//div[@slot='thumbnail']/*/*/@src").get()
        elif attachment_type == "video":
            attachment_link = box.xpath(".//shreddit-player/@preview").get()
        else:
            attachment_link = box.xpath(".//div[@slot='thumbnail']/a/@href").get()
        post_data.append({
            "authorProfile": "https://www.reddit.com/user/" + author if author else None,
            "authorId": box.xpath(".//shreddit-post/@author-id").get(),            
            "title": box.xpath("./@aria-label").get(),
            "link": "https://www.reddit.com" + link if link else None,
            "publishingDate": box.xpath(".//shreddit-post/@created-timestamp").get(),
            "postId": box.xpath(".//shreddit-post/@id").get(),
            "postLabel": post_label.strip() if post_label else None,
            "postUpvotes": int(upvotes) if upvotes else None,
            "commentCount": int(comment_count) if comment_count else None,
            "attachmentType": attachment_type,
            "attachmentLink": attachment_link,
        })
   
    # id for the next posts batch
    cursor_id = selector.xpath("//shreddit-post/@more-posts-cursor").get()
    return {"post_data": post_data, "info": info, "cursor": cursor_id}
 

def parse_subreddit_from_search(response: Response) -> List[Dict]:
    selector = Selector(response.text)
    url = str(response.url)
    info = {}
    info["id"] = url.split("/r")[-1]
    # info["description"] = selector.xpath("//shreddit-subreddit-header/@description").get()
    members = selector.xpath('//span[@class="subscribers"]/span[@class="number"]/text()').get()
    # rank = selector.xpath("//strong[@id='position']/*/@number").get()    
    info["members"] = int(members.replace(",", "")) if members else None
    # info["rank"] = int(rank) if rank else None
    # info["bookmarks"] = {}

    # for item in selector.xpath("//div[faceplate-tracker[@source='community_menu']]/faceplate-tracker"):
    #     name = item.xpath(".//a/span/span/span/text()").get()
    #     link = item.xpath(".//a/@href").get()
    #     info["bookmarks"][name] = link
    
    info["url"] = url
    post_data:List[Dict] = []
    
    for box in selector.xpath('//div[contains(@class,"has-linkflair")]'):
        link = box.xpath("./a/@href").get()
        author = box.xpath('.//div[@class="search-result-meta"]//span[@class="search-author"]/a/text()').get()
        post_label = box.xpath('//span[contains(@class,"linkflairlabel")]/span/text()').get()
        upvotes = box.xpath(".//div[@class='search-result-meta']/span[@class='search-score']/text()").get()
        comment_count = box.xpath('//a[contains(@class,"search-comments")]/text()').re_first(r'\d+')
        attachment_type = box.xpath(".//shreddit-post/@post-type").get()
        if attachment_type and attachment_type == "image":
            attachment_link = box.xpath(".//div[@slot='thumbnail']/*/*/@src").get()
        elif attachment_type == "video":
            attachment_link = box.xpath(".//shreddit-player/@preview").get()
        else:
            attachment_link = box.xpath(".//div[@slot='thumbnail']/a/@href").get()
        post_data.append({
            "authorProfile": "https://www.reddit.com/user/" + author if author else None,
            "authorId": box.xpath('//a[contains(@class,"id-")]/@class').re_first(r'id-([^\s]+)'),            
            "title": box.xpath(".//a[contains(@class,'may-blank')]/text()").get(),
            "link": "https://www.reddit.com" + link if link else None,
            "publishingDate": box.xpath(".//shreddit-post/@created-timestamp").get(),
            "postId": box.xpath(".//shreddit-post/@id").get(),
            "postLabel": post_label.strip() if post_label else None,
            "postUpvotes": int(upvotes.split(" ")[0]) if upvotes else None,
            "commentCount": int(comment_count) if comment_count else None,
            "attachmentType": attachment_type,
            "attachmentLink": attachment_link,
        })
    # id for the next posts batch
    cursor_id = selector.xpath("//shreddit-post/@more-posts-cursor").get()
    return {"post_data": post_data, "info": info, "cursor": cursor_id}
    

# async def scrape_subreddit(subreddit_id: str, sort: Literal["new", "hot", "old"], query: str = None, max_pages: int = None, limit: int = 10):
#     """scrape articles on a subreddit"""
#     base_url = f"https://www.reddit.com/r/{subreddit_id}"
    
#     if query:
#         base_url = base_url.replace("www", "old") + f"/search?q={query}&sort={sort}&limit={limit}"
#         # base_url += f"/search?q={query}"
    
#     response = await client.get(base_url)
#     subreddit_data = {}
#     if query:
#         data = parse_subreddit_from_search(response)
#     else:
#         data = parse_subreddit(response)
#     subreddit_data["info"] = data["info"]
#     subreddit_data["posts"] = data["post_data"]
#     cursor = data["cursor"]

#     def make_pagination_url(cursor_id: str):
#         return f"https://www.reddit.com/svc/shreddit/community-more-posts/hot/?after={cursor_id}%3D%3D&t=DAY&name={subreddit_id}&feedLength=3&sort={sort}" 
        
#     while cursor and (max_pages is None or max_pages > 0):
#         url = make_pagination_url(cursor)
#         response = await client.get(url)
#         data = parse_subreddit(response)
#         cursor = data["cursor"]
#         post_data = data["post_data"]
#         subreddit_data["posts"].extend(post_data)
#         if max_pages is not None:
#             max_pages -= 1
#     log.success(f"scraped {len(subreddit_data['posts'])} posts from the rubreddit: r/{subreddit_id}")
#     return subreddit_data

async def scrape_subreddit(
    subreddit_id: str,
    sort: Literal["new", "hot", "old"],
    query: str = None,
    max_posts: int = 50
):
    """scrape up to `max_posts` posts from a subreddit"""

    base_url = f"https://www.reddit.com/r/{subreddit_id}"
    if query:
        base_url = base_url.replace("www", "old") + f"/search?q={query}&sort={sort}&limit={max_posts}"

    response = await client.get(base_url)
    subreddit_data = {}

    # --- Parse first batch ---
    if query:
        data = parse_subreddit_from_search(response)
    else:
        data = parse_subreddit(response)

    subreddit_data["info"] = data["info"]
    subreddit_data["posts"] = data["post_data"]

    cursor = data["cursor"]

    def make_pagination_url(cursor_id: str):
        return (
            f"https://www.reddit.com/svc/shreddit/community-more-posts/"
            f"{sort}/?after={cursor_id}%3D%3D&t=DAY&name={subreddit_id}&feedLength=3&sort={sort}"
        )

    # --- Keep fetching until enough posts ---
    while cursor and len(subreddit_data["posts"]) < max_posts:
        url = make_pagination_url(cursor)
        response = await client.get(url)
        data = parse_subreddit(response)

        cursor = data["cursor"]
        subreddit_data["posts"].extend(data["post_data"])

    # Tronquer si on a dépassé
    subreddit_data["posts"] = subreddit_data["posts"][:max_posts]

    log.success(f"scraped {len(subreddit_data['posts'])} posts from r/{subreddit_id}")
    return subreddit_data
