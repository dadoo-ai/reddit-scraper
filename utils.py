# from __future__ import annotations
import os
from typing import List
from analyze.utils import group_csv_files


if __name__ == "__main__":
    # pipeline permettant d'analyser les commentaires d'un post
    group_csv_files(folder_path="results/analyze/posts", output_folder="results/analyze/posts", file_name="repared_posts_aggregated")
    

    