from __future__ import annotations

import ast
import os
import re
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ---------------- Config ----------------

@dataclass
class CSVUserAnalyzerConfig:
    # Règles d'inférence de rôle (inchangées)
    strong_sources: frozenset = frozenset({"self_identified", "profile_flair"})
    min_role_confidence: float = 0.6
    min_confirmations: int = 2
    min_cumulative_confidence: float = 1.5

    # Parsing / sortie
    max_quotes: int = 3
    max_justifs: int = 3

    # OpenAI
    model: str = "gpt-4o-mini"   # ← conseillé
    temperature: float = 0.0
    max_retries: int = 2
    backoff_initial: float = 0.7


# --- Schéma : 2 scores (Salesforce + Agentforce) ---
USER_JSON_SCHEMA = {
    "name": "user_sentiment_from_justifications",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment_salesforce": {"type": "number", "minimum": -1, "maximum": 1},
            "sentiment_agentforce": {"type": "number", "minimum": -1, "maximum": 1},
            "justifications": {"type": "string"},
            "evidence_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4
            }
        },
        "required": ["sentiment_salesforce", "sentiment_agentforce", "justifications", "evidence_quotes"],
        "additionalProperties": False
    }
}

# --- SYSTEM PROMPT : on évalue à partir des justifications agrégées ---
USER_SYSTEM_PROMPT = """You evaluate a Reddit user's overall stance toward Salesforce and toward Agentforce,
based SOLELY on previously generated 'justifications' text (summaries of their comments).

Return ONLY JSON fitting the schema.

Guidelines:
- Output both sentiment_salesforce and sentiment_agentforce in [-1..+1].
  -1 = very negative; 0 = neutral/mixed/unclear; +1 = very positive.
- Use the provided 'justifications' snippets as your evidence; do not invent content.
- If a target (Salesforce or Agentforce) is barely mentioned or unclear, keep its score near 0.
- 'justifications': 2–4 crisp sentences explaining your reasoning and the difference (if any) between Salesforce vs Agentforce stance.
- 'evidence_quotes': 1–4 verbatim snippets (~10–20 words) from the supplied text that back your assessment.
Be concise and do not exceed the schema.
"""


class CSVUserAnalyzer:
    """
    Agrège par utilisateur puis évalue DEUX sentiments globaux
    (Salesforce et Agentforce) en s'appuyant uniquement sur la
    colonne 'justifications' présente dans le CSV.

    Sortie par ligne :
      - user
      - author_id
      - author_profile
      - role_inferred
      - sentiment_salesforce
      - sentiment_agentforce
      - justifications
      - evidence_quotes
      - total_comments
    """

    def __init__(self, cfg: Optional[CSVUserAnalyzerConfig] = None):
        load_dotenv()
        self.cfg = cfg or CSVUserAnalyzerConfig()
        self.client = OpenAI()  # OPENAI_API_KEY requis
        self.df: Optional[pd.DataFrame] = None
        self.agg: Optional[pd.DataFrame] = None

    # ---------- Public API ----------

    def load(self, csv_path: str | Path) -> "CSVUserAnalyzer":
        df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8", engine="python")
        df.columns = [str(c).strip() for c in df.columns]

        # Sanitize simple pour evidence_quotes mal typées (évite SyntaxWarning avec ast)
        if "evidence_quotes" in df.columns:
            mask_bad = df["evidence_quotes"].astype(str).str.match(r"^\s*[\d\.\,]+\s*$", na=False)
            if mask_bad.any():
                df.loc[mask_bad, "evidence_quotes"] = ""

        self.df = self._normalize_columns(df)
        self._ensure_expected_columns(self.df)
        self._coerce_types(self.df)

        # garder uniquement les lignes avec user non vide
        self.df = self.df[~self.df["user"].isna() & (self.df["user"].astype(str).str.strip() != "")]
        self.df = self.df.copy()
        return self

    def aggregate(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Aucun dataframe chargé. Appelle .load(csv_path) d'abord.")

        rows: List[Dict[str, Any]] = []

        for user, g in self.df.groupby("user", dropna=True):
            author_id = self._first_non_null(g, "author_id")
            author_profile = self._first_non_null(g, "author_profile")
            role_inferred = self._infer_role_group(g)

            # Matière à analyser : JUSTIFICATIONS (et non les commentaires)
            justifs_src = self._gather_justifications(g)

            if not justifs_src:
                # Pas de matière → neutre
                total_comments = self._compute_total_comments(g)
                rows.append({
                    "user": user,
                    "author_id": author_id,
                    "author_profile": author_profile,
                    "role_inferred": role_inferred,
                    "sentiment_salesforce": 0.0,
                    "sentiment_agentforce": 0.0,
                    "justifications": "Insufficient justification text; defaulting to neutral.",
                    "evidence_quotes": "",
                    "total_comments": total_comments,
                })
                continue

            # Passage LLM basé sur justifications (2 scores + texte)
            sent_sf, sent_af, justifs_llm, quotes_llm = self._analyze_user_from_justifs(user, justifs_src)

            total_comments = self._compute_total_comments(g)
            rows.append({
                "user": user,
                "author_id": author_id,
                "author_profile": author_profile,
                "role_inferred": role_inferred,
                "sentiment_salesforce": round(sent_sf, 3),
                "sentiment_agentforce": round(sent_af, 3),
                "justifications": justifs_llm,
                "evidence_quotes": " | ".join(quotes_llm[: self.cfg.max_quotes]),
                "total_comments": total_comments,
            })

        self.agg = (
            pd.DataFrame(rows)
            .sort_values(["role_inferred", "sentiment_agentforce"], ascending=[True, False])
            .reset_index(drop=True)
        )
        return self.agg

    def save(self, out_csv: str | Path) -> Path:
        if self.agg is None:
            raise RuntimeError("Rien à sauvegarder. Appelle .aggregate() d'abord.")
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.agg.to_csv(out, index=False)
        return out

    # ---------- Internals ----------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rename_map = {
            "author": "user",
            "username": "user",
            "author_name": "user",
            "authorid": "author_id",
            "authorid_": "author_id",
            "authorprofile": "author_profile",
            "profile": "author_profile",
            "justification": "justifications",
            "evidence": "evidence_quotes",
            "evidence_quote": "evidence_quotes",
            "n_comments": "total_comments",
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df.rename(columns={k: v}, inplace=True)
        return df

    @staticmethod
    def _ensure_expected_columns(df: pd.DataFrame) -> None:
        for col in [
            "user", "author_id", "author_profile",
            "role", "role_source", "role_confidence",
            "justifications", "evidence_quotes", "total_comments",
        ]:
            if col not in df.columns:
                df[col] = np.nan

    def _coerce_types(self, df: pd.DataFrame) -> None:
        df["role"] = df["role"].apply(self._coerce_role)
        df["role_source"] = df["role_source"].astype(str).str.strip().str.lower()
        df["role_confidence"] = df["role_confidence"].apply(self._coerce_float)
        df["justifications"] = df["justifications"].astype(str).fillna("").str.strip()
        df["evidence_quotes"] = df["evidence_quotes"].apply(self._parse_quotes)

    @staticmethod
    def _coerce_role(x: Any) -> str:
        x = (str(x) if x is not None else "").strip().lower()
        return x if x in {"employee", "partner"} else "user"

    @staticmethod
    def _coerce_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            if isinstance(x, str) and not x.strip():
                return default
            if pd.isna(x):
                return default
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _parse_quotes(val: Any) -> List[str]:
        # évite ast.literal_eval sur "3.0," etc.
        if isinstance(val, (list, tuple, set)):
            return [str(x).strip() for x in val if str(x).strip()]
        try:
            if pd.isna(val):  # type: ignore
                return []
        except Exception:
            pass
        s = str(val).strip()
        if not s:
            return []
        looks_like_literal = s.startswith("[") or s.startswith("(")
        if looks_like_literal:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [x.strip() for x in s.split(" | ") if x.strip()]

    @staticmethod
    def _first_non_null(g: pd.DataFrame, col: str) -> str:
        if col not in g.columns:
            return ""
        vals = g[col].dropna().astype(str)
        return vals.iloc[0] if len(vals) else ""

    @staticmethod
    def _to_int_safe(x, default: Optional[int] = 0) -> int:
        try:
            if x is None:
                return default if default is not None else 0
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return default if default is not None else 0
            return int(float(s))  # accepte "1", "1.0", " 2 "
        except Exception:
            return default if default is not None else 0

    def _compute_total_comments(self, g: pd.DataFrame) -> int:
        if "total_comments" in g.columns:
            totals = g["total_comments"].apply(self._to_int_safe)
            if totals.notna().any():
                return int(totals.sum())
        return len(g)

    def _infer_role_group(self, g: pd.DataFrame) -> str:
        cfg = self.cfg
        role_scores = {"employee": 0.0, "partner": 0.0}
        role_counts = {"employee": 0, "partner": 0}
        for _, r in g.iterrows():
            role = self._coerce_role(r.get("role", "user"))
            src = str(r.get("role_source", "")).lower()
            conf = self._coerce_float(r.get("role_confidence", 0.0))
            if role in role_scores and (src in cfg.strong_sources) and conf >= cfg.min_role_confidence:
                role_scores[role] += conf
                role_counts[role] += 1
        if any(role_counts.values()):
            best = max(role_scores, key=lambda k: role_scores[k])
            if role_counts[best] >= cfg.min_confirmations or role_scores[best] >= cfg.min_cumulative_confidence:
                return best
        return "user"

    # -------- Utilise JUSTIFICATIONS comme source --------

    def _gather_justifications(self, g: pd.DataFrame) -> List[str]:
        """
        Récupère toutes les justifications du groupe user et les découpe sur ' | '.
        """
        items: List[str] = []
        for s in g["justifications"].astype(str).tolist():
            s = s.strip()
            if not s:
                continue
            parts = [x.strip() for x in s.split(" | ") if x.strip()]
            items.extend(parts)
        # Déduplication en gardant l'ordre
        seen, out = set(), []
        for it in items:
            if it not in seen:
                out.append(it)
                seen.add(it)
        return out

    # ---------------- LLM depuis JUSTIFICATIONS ----------------

    def _analyze_user_from_justifs(self, user: str, justifs: List[str]) -> Tuple[float, float, str, List[str]]:
        cfg = self.cfg

        # Construit un prompt compact avec les justifs
        blocks = [f"[{i+1}] {j}" for i, j in enumerate(justifs)]
        context = " | ".join(blocks)

        user_prompt = f"""
You are given multiple 'justifications' snippets previously generated for the SAME Reddit user.
Use them to infer the user's stance.

User: u/{user}

Justifications ({len(blocks)} items):
{context}
"""

        # Try Responses API + json_schema
        attempts, backoff = 0, cfg.backoff_initial
        while True:
            attempts += 1
            try:
                return self._call_openai_json_schema(user_prompt, USER_JSON_SCHEMA)
            except TypeError as e:
                if "response_format" in str(e):
                    break
            except Exception:
                pass
            if attempts > cfg.max_retries:
                break
            time.sleep(backoff + random.uniform(0, 0.2))
            backoff *= 1.6

        # Fallback Chat Completions (json_object)
        attempts, backoff = 0, cfg.backoff_initial
        while True:
            attempts += 1
            try:
                data = self._chat_json_object(USER_SYSTEM_PROMPT, user_prompt)
                sent_sf = self._clamp_float(data.get("sentiment_salesforce", 0.0), -1.0, 1.0)
                sent_af = self._clamp_float(data.get("sentiment_agentforce", 0.0), -1.0, 1.0)
                justifs_text = str(data.get("justifications", "")).strip() or "Insufficient direct signal; leaning neutral."
                quotes = [q.strip() for q in (data.get("evidence_quotes") or []) if q and q.strip()]
                quotes = [self._trim_to_words(q, 20) for q in quotes]
                quotes = self._dedup_keep_order(quotes)[: self.cfg.max_quotes]
                return sent_sf, sent_af, justifs_text, quotes
            except Exception:
                if attempts > cfg.max_retries:
                    break
                time.sleep(backoff + random.uniform(0, 0.2))
                backoff *= 1.6

        # Fallback final
        return 0.0, 0.0, "Model error; defaulting to neutral.", []

    # ----- OpenAI helpers -----

    def _call_openai_json_schema(self, user_prompt: str, schema: Dict[str, Any]) -> Tuple[float, float, str, List[str]]:
        r = self.client.responses.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            response_format={"type": "json_schema", "json_schema": schema},
            input=[
                {"role": "system", "content": [{"type": "text", "text": USER_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
        )
        txt = getattr(r, "output_text", None) or self._extract_text_from_responses(r)
        if not txt or not txt.strip():
            raise ValueError("Empty response from Responses API")
        data = self._safe_json_loads(txt)
        sent_sf = self._clamp_float(data.get("sentiment_salesforce", 0.0), -1.0, 1.0)
        sent_af = self._clamp_float(data.get("sentiment_agentforce", 0.0), -1.0, 1.0)
        justifs = str(data.get("justifications", "")).strip() or "Insufficient direct signal; leaning neutral."
        quotes = [q.strip() for q in (data.get("evidence_quotes") or []) if q and q.strip()]
        quotes = [self._trim_to_words(q, 20) for q in quotes]
        quotes = self._dedup_keep_order(quotes)[: self.cfg.max_quotes]
        return sent_sf, sent_af, justifs, quotes

    def _chat_json_object(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        chat = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        txt = chat.choices[0].message.content
        return self._safe_json_loads(txt)

    # ----- utils -----

    @staticmethod
    def _safe_json_loads(s: str) -> Dict[str, Any]:
        import orjson as _json
        try:
            return _json.loads(s)
        except Exception:
            s2 = s.strip().strip("`").strip()
            try:
                return _json.loads(s2)
            except Exception:
                return {}

    @staticmethod
    def _extract_text_from_responses(r) -> str:
        try:
            return "".join(
                block.text.value
                for item in r.output
                for block in getattr(item, "content", [])
                if getattr(block, "type", "") == "output_text" or hasattr(block, "text")
            )
        except Exception:
            return ""

    @staticmethod
    def _trim_to_words(s: str, max_words: int) -> str:
        return " ".join(str(s).split()[:max_words])

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        seen, out = set(), []
        for it in items:
            if it not in seen:
                out.append(it)
                seen.add(it)
        return out

    @staticmethod
    def _clamp_float(x: Any, lo: float, hi: float) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(lo, min(hi, v))


# --------- Exemple d'utilisation ---------
if __name__ == "__main__":
    analyzer = CSVUserAnalyzer()
    analyzer.load("results/analyze/posts/posts_aggregated.csv")
    analyzer.aggregate()
    analyzer.save("results/analyze/posts/user_aggregated.csv")
    print("OK")
