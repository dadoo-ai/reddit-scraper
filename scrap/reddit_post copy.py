from typing import List, Dict, Literal
from httpx import AsyncClient, Response
from parsel import Selector
from loguru import logger as log

client = AsyncClient(
    # previous client configuration
)

def parse_post_info(response: Response) -> Dict:
    """parse post data from a subreddit post"""
    selector = Selector(response.text)
    info = {}
    label = selector.xpath("//faceplate-tracker[@source='post']/a/span/div/text()").get()
    comments = selector.xpath("//shreddit-post/@comment-count").get()
    upvotes = selector.xpath("//shreddit-post/@score").get()
    extracted_text = selector.xpath('//div[@property="schema:articleBody"]//p/text()').getall()
    post_message = " ".join(t.strip() for t in extracted_text if t.strip())

    info["authorId"] = selector.xpath("//shreddit-post/@author-id").get()
    info["author"] = selector.xpath("//shreddit-post/@author").get()
    info["authorProfile"] = "https://www.reddit.com/user/" + info["author"] if info["author"] else None
    info["subreddit"] = selector.xpath("//shreddit-post/@subreddit-prefixed-name").get()
    info["postId"] = selector.xpath("//shreddit-post/@id").get()
    info["postLabel"] = label.strip() if label else None
    info["publishingDate"] = selector.xpath("//shreddit-post/@created-timestamp").get()
    info["postTitle"] = selector.xpath("//shreddit-post/@post-title").get()
    info["postMessage"] = post_message
    info["postLink"] = selector.xpath("//shreddit-canonical-url-updater/@value").get()
    info["commentCount"] = int(comments) if comments else None
    info["upvoteCount"] = int(upvotes) if upvotes else None
    info["attachmentType"] = selector.xpath("//shreddit-post/@post-type").get()
    info["attachmentLink"] = selector.xpath("//shreddit-post/@content-href").get()
    return info


def parse_post_comments(response: Response) -> List[Dict]:
    """parse post comments"""

    def parse_comment(parent_selector:Selector) -> Dict:
        """parse a comment object"""
        author = parent_selector.xpath("./@data-author").get()
        link = parent_selector.xpath("./@data-permalink").get()
        dislikes = parent_selector.xpath(".//span[contains(@class, 'dislikes')]/@title").get()
        upvotes = parent_selector.xpath(".//span[contains(@class, 'likes')]/@title").get()
        downvotes = parent_selector.xpath(".//span[contains(@class, 'unvoted')]/@title").get()        
        return {
            "authorId": parent_selector.xpath("./@data-author-fullname").get(),
            "author": author,
            "authorProfile": "https://www.reddit.com/user/" + author if author else None,
            "commentId": parent_selector.xpath("./@data-fullname").get(),
            "link": "https://www.reddit.com" + link if link else None,
            "publishingDate": parent_selector.xpath(".//time/@datetime").get(),
            "commentBody": parent_selector.xpath(".//div[@class='md']/p/text()").get(),
            "upvotes": int(upvotes) if upvotes else None,
            "dislikes": int(dislikes) if dislikes else None,
            "downvotes": int(downvotes) if downvotes else None,            
        }

    def parse_replies(what) -> List[Dict]:
        """recursively parse replies"""
        replies = []
        for reply_box in what.xpath(".//div[@data-type='comment']"):
            reply_comment = parse_comment(reply_box)
            child_replies = parse_replies(reply_box)
            if child_replies:
                reply_comment["replies"] = child_replies
            replies.append(reply_comment)
        return replies

    selector = Selector(response.text)
    data = []
    for item in selector.xpath("//div[@class='sitetable nestedlisting']/div[@data-type='comment']"):
        comment_data = parse_comment(item)
        replies = parse_replies(item)
        if replies:
            comment_data["replies"] = replies
        data.append(comment_data)            
    return data


async def scrape_post(url: str, sort: Literal["old", "new", "top"]) -> Dict:
    """scrape subreddit post and comment data"""
    response = await client.get(url)  
    post_data = {}
    post_data["info"] = parse_post_info(response)
    
    # scrape the comments from the old.reddit version, with the same post URL 
    # bulk_comments_page_url = post_data["info"]["postLink"].replace("www", "old") + f"?sort={sort}&limit=500"
    bulk_comments_page_url = url.replace("www", "old") + f"?sort={sort}&limit=500"
    response = await client.get(bulk_comments_page_url)
    post_data["comments"] = parse_post_comments(response) 
    log.success(f"scraped {len(post_data['comments'])} comments from the post {url}")
    return post_data