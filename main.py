from __future__ import annotations
import json
import asyncio
import os
from pathlib import Path
from itertools import islice
from urllib.parse import urlparse
from typing import List, Literal, Dict, Optional, Tuple, Any
from scrap.user_profile import ProfileBatchRunner
from scrap.reddit_subreddits import scrape_subreddit
from scrap.user_extractor import RedditUserExtractor
from loguru import logger as log
from analyze.run import run_pipeline
from analyze.utils import group_csv_files
from analyze.post_analyse_by_user import CSVUserAnalyzer
import pandas as pd
from analyze.apify_user_analyze import run
from scrap.reddit_post import PostScraper
from apify.apify_scrap_profile import ApifyProfileScraper  
from apify.apify_to_csv import FolderExtractor
from analyze.prompt_all_posts import PromptCSVAnalyzer, RunnerConfig
import argparse

async def scrap_subreddit(theme: str, filter: str, max_posts):
	data = await scrape_subreddit(
		subreddit_id=theme, sort="relevance", query=filter, max_posts=max_posts
	)
	with open("results/posts/post_search.json", "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


# async def list_post_comments_from_url(url: str):
#     post_data = await scrape_post(url=url, sort="")
#     with open("results/post_comments.json", "w", encoding="utf-8") as file:
#         json.dump(post_data, file, indent=2, ensure_ascii=False)


async def scrape_comments_from_post(path_file: str, limit: int = 5, batch_size: int = 5, pause_seconds: int = 5):
	"""
	Scrape les commentaires d'un post depuis le fichier results/posts/post_search.json
	Traite les URLs par groupes avec des pauses entre chaque groupe
	"""
	with open(path_file, "r", encoding="utf-8") as file:
		data = json.load(file)


	if posts := data.get("posts"):
		urls: List[str] = list(
			set(post["link"] for post in posts)
		)
	elif posts := data.get("errors"):
		urls: List[str] = list(
			set(post.get("url") for post in posts)
		)         
	
	urls = urls[:limit]
	scraper = PostScraper()
	# Traiter par groupes
	for i in range(0, len(urls), batch_size):
		batch = urls[i:i + batch_size]
		print(f"Traitement du groupe {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size} ({len(batch)} URLs)")
		
		await asyncio.gather(*[
			scraper.scrape_post(
				url=post, output_folder="results/comments", require_comments=True
			)
			for post in batch
		])
		
		# Pause entre les groupes (sauf pour le dernier)
		if i + batch_size < len(urls):
			print(f"Pause de {pause_seconds} secondes...")
			await asyncio.sleep(pause_seconds)
	


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
	concurrency: int = 10,  # paramétrable
	threshold: int = 100,  # paramétrable
	max_pages: Optional[int] = None,  # None = paginer jusqu'au bout ; sinon ex: 50
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
	for file in filenames[:limit]:
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


def convert_json_to_csv(input_file_path: str, output_file_path: str, csv_file_name: str):

	extractor = FolderExtractor(input_file_path, output_file_path, csv_file_name)
	extractor.run()





if __name__ == "__main__":
	theme: str = "salesforce"
	filter: str = "agentforce"
	model = "gpt-4.1"

	# permet de scrappper tous les subreddits du theme avec le filtre
	# asyncio.run(scrap_subreddit(theme, filter, max_posts=100))

	# scrap les commentaires d'un post depuis le fichier results/posts/post_search.json
	# asyncio.run(
	#     scrape_comments_from_post(path_file="results/comments/parse_error.json", limit=100)
	# )

	# extraction des users des posts
	# scrap_user_from_post(folder_path="results/comments", output_format="json")


	# pipeline permettant d'analyser les commentaires d'un post
	# analyze_posts_by_users(
	#     model,
	#     input_dir="results/comments",
	#     output_dir="results/analyze/posts",
	#     max_workers=3,
	#     limit=100,
	# )

	# regrouement des csv des posts
	# group_csv_files(folder_path="results/analyze/posts", output_folder="results/analyze/aggregated", file_name="posts_aggregated")
	
	# analyse des users de chaque post et regroupe les users dans un seul csv
	# analyze_posts_by_users_grouped(
	#     input_file_path="results/analyze/aggregated/aggregated_posts.csv",
	#     output_file_path="results/analyze/aggregated/aggregated_users.csv",
	# )

	# extraction des posts d'un utilisateur ayant des posts dans le csv results/analyze/reporting/user_aggregated.csv
	# runner = ProfileBatchRunner(
	#         csv_path="results/analyze/reporting/user_aggregated.csv",
	#         output_dir="results/analyze/reporting/users",
	#         column="author_profile",
	#         concurrency=2, threshold=100, max_pages=None,
	#         pause_between_batches=10, request_gap_s=1.8, page_gap_s=1.5,
	#         post_fetch_concurrency=1, stop_on_first_valid=True
	# )
	# asyncio.run(runner.run())

	# scrap le profil d'un utilisateur avec Apify
	# Méthode 1 : Avec pandas (recommandée)
	# def extract_username(url: str) -> str:
	#     # path = urlparse(url).path  # ex: "/user/Delicious-Ad9511"
	#     profile_id=url.strip("/").split("/")[-1]
	#     print(profile_id)
	#     # return path.strip("/").split("/")[-1]
	
	# df = pd.read_csv("results/analyze/aggregated/aggregated_users.csv")
	# author_profiles:List[str] = df['user'].tolist()  # Liste de toutes les valeurs
	
	# with open("results/users/profiles/done.txt", "r", encoding="utf-8") as f:
	#     done_profiles:List[str] = f.read().splitlines()
	
	# limited_profiles = [f"https://www.reddit.com/user/{profile_id}" for profile_id in author_profiles if profile_id not in done_profiles] 
	
	# apify_scraper = ApifyProfileScraper()
	# # for profile in limited_profiles:
	# #     apify_scraper.scrape_profile(url=profile)
	
	# CONCURRENCY = 8
	# BATCH_SIZE = 20
	# SLEEP_BETWEEN_BATCHES = 10

	# async def _run_one(scraper: "ApifyProfileScraper", url: str, sem: asyncio.Semaphore) -> Tuple[str, Optional[Any], Optional[Exception]]:
	#     async with sem:
	#         try:
	#             result = await asyncio.to_thread(scraper.scrape_profile, url=url)
	#             return (url, result, None)
	#         except Exception as e:
	#             return (url, None, e)

	# def chunked(iterable, size):
	#     """Découpe iterable en morceaux de taille max `size`."""
	#     it = iter(iterable)
	#     while True:
	#         batch = list(islice(it, size))
	#         if not batch:
	#             break
	#         yield batch

	# async def run_profiles_async(limited_profiles: List[str]) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Exception]]]:
	#     sem = asyncio.Semaphore(CONCURRENCY)
	#     apify_scraper = ApifyProfileScraper()

	#     oks, errs = [], []

	#     for i, batch in enumerate(chunked(limited_profiles, BATCH_SIZE), 1):
	#         print(f"[INFO] Lancement batch {i} ({len(batch)} URLs)")

	#         tasks = [_run_one(apify_scraper, url, sem) for url in batch]
	#         done = await asyncio.gather(*tasks, return_exceptions=False)

	#         for url, result, err in done:
	#             if err is None:
	#                 oks.append((url, result))
	#             else:
	#                 errs.append((url, err))

	#         if i * BATCH_SIZE < len(limited_profiles):  # pause si pas le dernier batch
	#             print(f"[INFO] Pause {SLEEP_BETWEEN_BATCHES}s avant prochain batch…")
	#             await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

	#     return oks, errs
	
	# print(len(limited_profiles))
	# asyncio.run(run_profiles_async(limited_profiles))


	# ----------- apify json to csv -----------
	# convert_json_to_csv(input_file_path="results/users/profiles", output_file_path="results/users", csv_file_name="merged_profiles.csv")


	# -------------- analyze users -----------
	
	# INPUT = Path("results/users/merged_profiles.csv")      # adapte le chemin
	# OUTPUT = Path("results/analyze/aggregated/merged_profiles_aggregated.csv")   # adapte le chemin

	# run(INPUT, OUTPUT, log_level="DEBUG")  # écrit directement le CSV de sortie



