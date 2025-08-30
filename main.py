# from __future__ import annotations
import json
import asyncio
import os
from typing import List, Literal, Dict, Optional
from scrap.user_profile import ProfileBatchRunner
from scrap.reddit_subreddits import scrape_subreddit
from scrap.reddit_post import scrape_post
from scrap.user_extractor import RedditUserExtractor
from loguru import logger as log
from analyze.run import run_pipeline
from analyze.utils import group_csv_files
from analyze.post_analyse_by_user import CSVUserAnalyzer
import pandas as pd


async def search_posts(theme: str, filter: str, max_posts):
    data = await scrape_subreddit(
        subreddit_id=theme, sort="relevance", query=filter, max_posts=max_posts
    )
    with open("results/posts/post_search.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

async def list_post_comments_from_url(url: str):
    post_data = await scrape_post(url=url, sort="")
    with open("results/post_comments.json", "w", encoding="utf-8") as file:
        json.dump(post_data, file, indent=2, ensure_ascii=False)

async def list_post_comments_from_file(path_file: str, limit: int = 5):
    with open(path_file, "r", encoding="utf-8") as file:
        data = json.load(file)
        urls: List[str] = []
        tasks: List[asyncio.Task] = []
        count: int = 0
        for post in data["posts"]:
            if post["link"] not in urls:
                urls.append(post["link"])
                tasks.append(
                    asyncio.create_task(
                        scrape_post(url=post["link"], sort="", require_comments=True)
                    )
                )
        comments = await asyncio.gather(*tasks[:limit])

    for comment in comments:
        count += 1
        with open(
            f"results/comments/post_{count}_comments.json", "w", encoding="utf-8"
        ) as file:
            json.dump(comment, file, indent=2, ensure_ascii=False)

def scrap_user_from_post(
    folder_path: str,
    output_format: Literal["csv", "json"],
    output_dir: str = "results/users",
):
    """
    Extrait les utilisateurs des posts dans le dossier et les sauvegarde dans le dossier output_dir
    """
    file_name = folder_path.split("/")[-1]
    extractor = RedditUserExtractor()
    files = os.listdir(folder_path)

    for file in files[:3]:
        if file.endswith(".json"):
            continue
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        extractor.ingest_json(raw)

    if output_format == "csv":
        output = extractor.to_csv(f"{output_dir}/{file_name}_users.csv")
    elif output_format == "json":
        output = extractor.to_json(f"{output_dir}/{file_name}_users.json")

    print("Extraction terminée :", extractor.to_dataframe().shape[0], "utilisateurs")
    return output

async def scrap_user_profile_from_csv_file(
    csv_path: str,
    concurrency: int = 10,     # paramétrable
    threshold: int = 100,      # paramétrable
    max_pages: Optional[int] = None, # None = paginer jusqu'au bout ; sinon ex: 50
):
    await ProfileBatchRunner(
        csv_path=csv_path,
        delimiter=",",
        column="author_profile",
        concurrency=concurrency,
        threshold=threshold,
        max_pages=max_pages,
        pause_between_batches=10,
    ).run()


def analyze_posts_by_users(
    model: str, input_dir: str, output_dir: str, max_workers: int = 1, limit: int = 1
):
    filenames: List[str] = [
        file for file in os.listdir(input_dir) if file.endswith(".json")
    ]
    for file in filenames[75:limit]:
        input_json_path = os.path.join(input_dir, file)

        print(f"Processing file: {input_json_path}")
        with open(input_json_path, "r", encoding="utf-8") as f:
            output_csv_path = os.path.join(
                output_dir, file.split(".")[0] + "_analyze.csv"
            )

            run_pipeline(
                input_json_path=input_json_path,
                output_users_csv=output_csv_path,
                model=model,
                max_workers=max_workers,
            )

        # Exécuter le pipeline d'analyse qui génère automatiquement le CSV
        # run_pipeline(input_json_path, output_csv_path, model)
        print(f"Résultats sauvegardés dans: {output_csv_path}")

def analyze_posts_by_users_grouped(input_file_path: str, output_file_path: str):
    """
    Analyse les posts groupés par utilisateur et sauvegarde le résultat dans un fichier CSV.
    """
    analyzer = CSVUserAnalyzer()
    analyzer.load(input_file_path)
    analyzer.aggregate()
    out_path = analyzer.save(output_file_path)
    print(f"OK → {out_path}")


if __name__ == "__main__":
    theme: str = "salesforce"
    filter: str = "agentforce"
    model = "gpt-4.1"

    # permet de scrappper tous les subreddits du theme avec le filtre
    asyncio.run(search_posts(theme, filter, max_posts=100))

    # permet de scrappper tous les commentaires d'un post a partir de la liste des posts dans le fichier results/posts/post_search.json
    # asyncio.run(list_post_comments_from_file(path_file="results/posts/post_search.json", limit=100))

    # extraction des users des posts
    # scrap_user_from_post(folder_path="results/comments", output_format="json")

    # extraction des posts et comments d'un utilisateur
    
    # runner = ProfileBatchRunner(
    #         csv_path="results/analyze/reporting/user_aggregated.csv",
    #         output_dir="results/analyze/reporting/users",
    #         column="author_profile",
    #         concurrency=2, threshold=100, max_pages=None,
    #         pause_between_batches=10, request_gap_s=1.8, page_gap_s=1.5,
    #         post_fetch_concurrency=1, stop_on_first_valid=True
    # )
    # asyncio.run(runner.run())
        
    
    # pipeline permettant d'analyser les commentaires d'un post
    # analyze_posts_by_users(
    #     model,
    #     input_dir="results/comments",
    #     output_dir="results/analyze/posts",
    #     max_workers=6,
    #     limit=100,
    # )

    # regrouement 
    # group_csv_files(folder_path="results/analyze/posts", output_folder="results/analyze/posts", file_name="posts_aggregated")
    # analyze_posts_by_users_grouped(
    #     input_file_path="results/analyze/posts/posts_aggregated.csv",
    #     output_file_path="results/analyze/posts/user_aggregated.csv",
    # )

    