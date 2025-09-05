from __future__ import annotations

import time
import random
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -------- LOGGING --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("csv_global_analyze.log", encoding="utf-8")]
)
log = logging.getLogger("csv_global_analyze")

# -------- CONFIG --------
class Config:
    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.max_retries = 2
        self.backoff_initial = 0.8
        self.max_items = 2000         # nombre max de posts/commentaires
        self.max_chars_item = 500     # tronque chaque item
        self.max_total_chars = 120_000

SYSTEM_PROMPT = """You are a precise analyst.
You will receive:
1) A corpus of Reddit posts & comments (mixed, not grouped).
2) A task question.

Rules:
- Base your ANALYZE only on the provided corpus.
- Then produce a JUSTIFICATION section: explain why you gave this analysis, and quote or paraphrase relevant posts/comments.
- If the corpus is insufficient, state so explicitly.
Return **plain text** with exactly these headers:

ANALYZE: ...
JUSTIFICATION: ...
"""

class GlobalCSVAnalyzer:
    def __init__(self, cfg: Config = Config()):
        load_dotenv()
        self.cfg = cfg
        self.client = OpenAI()
        log.info(f"Initialized with model={cfg.model}, temperature={cfg.temperature}")

    def run(
        self,
        posts_csv: Path,
        prompts_csv: Path,
        out_txt: Path,
        prompt_row: Optional[int] = None
    ) -> Path:
        df_posts = self._load_posts(posts_csv)
        df_prompts = self._load_prompts(prompts_csv)

        questions = self._select_prompts(df_prompts, prompt_row)
        corpus = self._build_corpus(df_posts)

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w", encoding="utf-8") as f:
            for i, question in enumerate(questions, start=1):
                log.info(f"Running question {i}/{len(questions)}")
                analysis_block = self._ask_llm(corpus=corpus, question=question)

                f.write("QUESTION:\n")
                f.write(question.strip() + "\n\n")
                f.write(analysis_block.strip() + "\n")
                f.write("\n" + "-" * 80 + "\n\n")

        log.info(f"Output written to {out_txt}")
        return out_txt

    # --- Loading ---
    @staticmethod
    def _load_posts(path: Path) -> pd.DataFrame:
        log.info(f"Loading posts/comments from {path}")
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8", engine="python")
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        for col in ("title", "body"):
            if col not in df.columns:
                df[col] = ""
        df["title"] = df["title"].astype(str).fillna("")
        df["body"] = df["body"].astype(str).fillna("")
        df = df[(df["title"] != "") | (df["body"] != "")]
        log.info(f"Posts loaded: {len(df)} rows")
        return df

    @staticmethod
    def _load_prompts(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8", engine="python")
        df.columns = [c.lower().strip() for c in df.columns]
        if "prompt" not in df.columns:
            raise ValueError("Le CSV des questions doit avoir une colonne 'prompt'.")
        df["prompt"] = df["prompt"].astype(str).fillna("").str.strip()
        df = df[df["prompt"] != ""]
        if df.empty:
            raise ValueError("Aucune question valide détectée dans le CSV.")
        return df

    @staticmethod
    def _select_prompts(df: pd.DataFrame, row: Optional[int]) -> List[str]:
        if row is None:
            return df["prompt"].tolist()
        if row < 1 or row > len(df):
            raise IndexError(f"--prompt-row={row} hors limites (1..{len(df)})")
        return [df.iloc[row - 1]["prompt"]]

    def _build_corpus(self, df: pd.DataFrame) -> str:
        cfg = self.cfg
        lines: List[str] = []
        take = df.head(cfg.max_items)
        for _, r in take.iterrows():
            title = self._clip(r.get("title", ""), cfg.max_chars_item)
            body = self._clip(r.get("body", ""), cfg.max_chars_item)
            if title and body:
                lines.append(f"{title} — {body}")
            elif title:
                lines.append(title)
            elif body:
                lines.append(body)

        corpus = "\n".join(lines)
        if len(corpus) > cfg.max_total_chars:
            corpus = corpus[: cfg.max_total_chars] + "\n...[truncated]"
        log.info(f"Corpus built with {len(lines)} items (chars={len(corpus)})")
        return corpus

    def _ask_llm(self, corpus: str, question: str) -> str:
        cfg = self.cfg
        user_message = f"CORPUS:\n{corpus}\n\nQUESTION:\n{question}"

        attempts, backoff = 0, cfg.backoff_initial
        while True:
            attempts += 1
            try:
                chat = self.client.chat.completions.create(
                    model=cfg.model,
                    temperature=cfg.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                )
                return (chat.choices[0].message.content or "").strip()
            except Exception as e:
                if attempts > cfg.max_retries:
                    log.error(f"LLM failed after {attempts} attempts: {e}")
                    return "ANALYZE: Model error\nJUSTIFICATION: None"
                wait = backoff + random.uniform(0, 0.2)
                log.warning(f"Retrying in {wait:.1f}s ({e})")
                time.sleep(wait)
                backoff *= 1.6

    @staticmethod
    def _clip(s: str, n: int) -> str:
        s = str(s).strip()
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


# -------- CLI --------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Analyse globale (posts+commentaires) à partir de questions dans un CSV."
    )
    p.add_argument("--posts-csv", required=True, type=Path, help="CSV des posts/commentaires.")
    p.add_argument("--prompts-csv", required=True, type=Path, help="CSV avec une colonne 'prompt' (les questions).")
    p.add_argument("--output", required=True, type=Path, help="Fichier texte en sortie.")
    p.add_argument("--prompt-row", type=int, default=None, help="Numéro (1-based) de la question à utiliser. Par défaut: toutes.")
    p.add_argument("--model", type=str, default="gpt-4.1", help="Modèle OpenAI à utiliser (défaut: gpt-4.1).")
    p.add_argument("--temperature", type=float, default=0.0, help="Température du modèle (défaut: 0.0).")
    args = p.parse_args()

    cfg = Config(model=args.model, temperature=args.temperature)
    runner = GlobalCSVAnalyzer(cfg)
    runner.run(args.posts_csv, args.prompts_csv, args.output, args.prompt_row)


# explication Toutes les questions du CSV :

"""
python analyze/prompt_all_posts.py \
  --posts-csv results/reddit_posts.csv \
  --prompts-csv prompt/questions.csv \
  --output results/analyze/analysis.txt


Une seule question (ligne 3 du CSV) :

python analyze/prompt_all_posts.py \
  --posts-csv results/reddit_posts.csv \
  --prompts-csv prompt/questions.csv \
  --prompt-row 3 \
  --output results/analyze/analysis.txt

  python analyze/prompt_all_posts.py \
  --posts-csv results/reddit_posts.csv \
  --prompts-csv prompt/questions.csv \
  --output results/analyze/analysis.txt

# En forçant modèle et température
python analyze/prompt_all_posts.py \
  --posts-csv results/reddit_posts.csv \
  --prompts-csv prompt/questions.csv \
  --output results/analyze/analysis.txt \
  --model gpt-4o-mini \
  --temperature 0.7
  """