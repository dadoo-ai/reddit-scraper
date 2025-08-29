from .reddit_thread_flattener import RedditThreadFlattener
from .thread_analyzer import CommentAnalyzer
import pprint

def run_pipeline(input_json_path: str, output_users_csv: str, model: str, max_workers: int = 1):
    flattener = RedditThreadFlattener(max_ancestor_hops=2, dedupe=True)
    data = flattener.load(input_json_path)
    flat = flattener.flatten(data)
 
    analyzer = CommentAnalyzer(model=model, max_workers=max_workers)
    results = analyzer.analyze(flat)

    attachment = data.get("info",{}).get("postLink", "")
    
    users_df = analyzer.aggregate_by_user(results, post_link=attachment)
    users_df.to_csv(output_users_csv, index=False, encoding="utf-8")
    print(f"✔ Users CSV: {output_users_csv}")


def repare_empy_data(input_json_path: str, output_users_csv: str, model: str, max_workers: int = 1):
    flattener = RedditThreadFlattener(max_ancestor_hops=2, dedupe=True)
    data = flattener.load(input_json_path)
    flat = flattener.flatten(data)
 
   
