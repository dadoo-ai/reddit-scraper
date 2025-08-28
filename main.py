from scrap.reddit_subreddits import scrape_subreddit
from scrap.reddit_post import scrape_post
import json
import asyncio
from analyze.run import run_pipeline


# liste des subreddits
async def list_posts():
    data = await scrape_subreddit(subreddit_id="salesforce", sort="new", max_pages=2)
    with open("results/subreddit_posts.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def search_posts(theme: str, filter:str):
    data = await scrape_subreddit(
        subreddit_id=theme, sort="new", query=filter, max_pages=2
    )
    with open("results/subreddit_search.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def list_post_comments(url: str):
    post_data = await scrape_post(url=url, sort="")
    with open("results/post_comments.json", "w", encoding="utf-8") as file:
        json.dump(post_data, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    theme: str = "salesforce"
    filter: str = "agentforce"

    url = "https://www.reddit.com/r/salesforce/comments/1mo8oni/how_effective_is_salesforce_agentforce_for/"
    model = "gpt-4.1"

    # permet de scrappper tous les subreddits du theme avec le filtre
    # asyncio.run(search_posts(theme, filter)) 

    # permet de scrappper tous les commentaires d'un post a partir de son url
    # asyncio.run(list_post_comments(url))
    
    # pipeline permettant d'analyser les commentaires d'un post
    # run_pipeline(
    #     input_json_path="results/post_comments.json",
    #     output_users_csv="results/users_agentforce.csv",
    #     model=model,
    #     max_workers=1
    # )
