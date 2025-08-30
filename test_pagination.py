import asyncio
from scrap.reddit_subreddits import scrape_subreddit
from loguru import logger

async def test_pagination():
    """Test de la pagination"""
    logger.info("Testing pagination...")
    
    result = await scrape_subreddit(
        max_posts=50,  # Demander plus de posts pour tester la pagination
        subreddit_id="python",
        query="django",
        sort="relevance",
        delay_s=1.0,
        old_reddit=True
    )
    
    logger.info(f"Total posts scraped: {len(result['posts'])}")
    logger.info(f"Subreddit info: {result['info']}")
    
    # Afficher les premiers posts
    for i, post in enumerate(result['posts'][:5]):
        logger.info(f"Post {i+1}: {post['title']} - {post['commentCount']} comments")

if __name__ == "__main__":
    asyncio.run(test_pagination())
