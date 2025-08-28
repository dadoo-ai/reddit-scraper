from .reddit_thread_flattener import RedditThreadFlattener
from .thread_analyzer import CommentAnalyzer

def run_pipeline(
    input_json_path: str,
    output_users_csv: str ,
    model: str,
    max_workers: int = 1
):
    # 1) Flatten
    flattener = RedditThreadFlattener(max_ancestor_hops=2, dedupe=True)
    data = flattener.load(input_json_path)
    flat = flattener.flatten(data)

    # 2) Analyze (une seule passe)
    analyzer = CommentAnalyzer(model=model, max_workers=max_workers)
    results = analyzer.analyze(flat)

    # 3) Aggregate by user
    users_df = analyzer.aggregate_by_user(results)
    users_df.to_csv(output_users_csv, index=False, encoding="utf-8")
    print(f"✔ Users CSV: {output_users_csv}")
