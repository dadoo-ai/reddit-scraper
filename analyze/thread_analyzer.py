from __future__ import annotations

import orjson as json
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import time
import random
import re

from .reddit_thread_flattener import FlatComment
from dotenv import load_dotenv

load_dotenv()  # charge OPENAI_API_KEY si présent

# ----- Schéma strict pour forcer la forme de sortie -----
JSON_SCHEMA = {
    "name": "reddit_comment_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": ["employee", "partner", "user"] 
            },
            "role_source": {
                "type": "string",
                "enum": ["self_identified", "profile_flair", "third_party_claim", "inference", "none"]
            },
            "role_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "agentforce_sentiment": {"type": "number", "minimum": -1, "maximum": 1},
            "agentforce_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "targets_salesforce": {"type": "boolean"},
            "is_reply_to_salesforce_question": {"type": "boolean"},
            "justification": {"type": "string"},
            "evidence_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4
            }
        },
        "required": [
            "role",
            "role_source",
            "role_confidence",
            "agentforce_sentiment",
            "agentforce_confidence",
            "targets_salesforce",
            "is_reply_to_salesforce_question",
            "justification",
            "evidence_quotes"
        ],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = """You analyze Reddit threads about Salesforce and Agentforce.
Return ONLY JSON matching the provided JSON schema.

STRICT Guidelines:

- role:
  - MUST be "user" by default.
  - Use "employee" or "former_employee" ONLY if there is explicit first-person evidence
    (e.g., "I work at Salesforce", "I'm an AE at Salesforce", "at my Salesforce team").
  - Use "customer" ONLY if the comment clearly states personal usage or purchase
    (e.g., "we use Salesforce at my company", "as a Salesforce customer").
  - Inference, insider tone, or hearsay is NOT sufficient.
  - If you cannot quote explicit evidence, role MUST be "user".

- role_source:
  - "self_identified" → user clearly states it (e.g., "I work at Salesforce").
  - "profile_flair" → evidence comes from profile flair/metadata.
  - "third_party_claim" → another user claims this about them.
  - "inference" → weak guess without explicit quote.
  - "none" → default when role="user".

- role_confidence:
  - ≤ 0.2 when role="user".
  - ≥ 0.6 only when explicit evidence is present.
  - Never set high confidence if role is inferred without proof.

- agentforce_sentiment: ONLY about Salesforce/Agentforce adoption/impact (−1 very negative … +1 very positive).
- agentforce_confidence: certainty of that sentiment classification.

- targets_salesforce = true ONLY if praise/critique explicitly targets Salesforce or Agentforce.
- is_reply_to_salesforce_question = true ONLY if parent explicitly mentions Salesforce/Agentforce.

- justification: concise (2–4 sentences) explaining reasoning.
- evidence_quotes:
  - 1–4 verbatim snippets (10–20 words).
  - MUST include at least one explicit snippet proving the role when role ≠ user.

Fallback:
- If off-topic or unclear → role="user", role_source="none", low confidence, sentiment≈0.
- Always prefer "user" over guessing.

"""

# ---------------- Helpers ----------------
def _default_analysis(error_msg: str | None = None) -> Dict[str, Any]:
    return {
        "role": "user",  # CHANGED (avant: "unknown")
        "role_source": "none",
        "role_confidence": 0.0,
        "agentforce_sentiment": 0.0,
        "agentforce_confidence": 0.0,
        "targets_salesforce": False,
        "is_reply_to_salesforce_question": False,
        "justification": f"error: {error_msg}" if error_msg else "",
        "evidence_quotes": []
    }


def extract_text_from_responses(r) -> str:
    """Compat : récupère le texte depuis l'API Responses malgré les variations de SDK."""
    txt = getattr(r, "output_text", None)
    if txt:
        return txt
    try:
        return "".join(
            block.text.value
            for item in r.output
            for block in getattr(item, "content", [])
            if getattr(block, "type", "") == "output_text" or hasattr(block, "text")
        )
    except Exception:
        return ""

def _trim_to_words(s: str, n: int = 20) -> str:
    """Coupe une quote à ~n mots (guideline: 10–20 words)."""
    return " ".join(str(s).split()[:n])

def _has_explicit_role_evidence(role: str, quotes: list[str]) -> bool:
    """Vérifie qu'au moins une quote prouve explicitement le rôle."""
    # CHANGED: "user" = baseline, pas de preuve requise
    if role == "user":  # CHANGED
        return True
    if not quotes:
        return False

    # CHANGED: patterns adaptés à employee / partner (EN + FR)
    PATTERNS = {
        "employee": [
            r"\bI work at Salesforce\b",
            r"\bI'm (an?|the) .* at Salesforce\b",
            r"\bmy (team|job|role) at Salesforce\b",
            r"\b(employee|engineer|AE|rep) at Salesforce\b",
            r"\bje travaille chez Salesforce\b",
            r"\bje suis (employ[ée]?) chez Salesforce\b",
            r"\bmon (poste|équipe) chez Salesforce\b",
        ],
        "partner": [
            r"\bI work (for|at) (a|the|my)?\s*Salesforce (consulting|implementation)?\s*partner\b",
            r"\bwe are (an?|the) Salesforce (consulting|implementation)? partner\b",
            r"\b(AppExchange|ISV)\s*partner\b",
            r"\bSalesforce (SI|systems integrator)\b",
            r"\bpartenaire Salesforce\b",
            r"\bint[ée]grateur Salesforce\b",
            r"\bnous sommes partenaire Salesforce\b",
            r"\bESN partenaire Salesforce\b",
        ],
    }

    for q in quotes:
        for p in PATTERNS.get(role, []):
            if re.search(p, q, flags=re.IGNORECASE):
                return True
    return False


def _coerce_analysis(obj: Dict[str, Any]) -> Dict[str, Any]:
    ...
    # Normalisation rôle
    # CHANGED: seuls "employee", "partner", "user" sont valides ; tout le reste → "user"
    if obj.get("role") not in {"employee", "partner", "user"}:  # CHANGED
        obj["role"] = "user"  # CHANGED

    if obj.get("role_source") not in {"self_identified","profile_flair","third_party_claim","inference","none"}:
        obj["role_source"] = "none"

    ...
    role = obj.get("role", "user")         # CHANGED (baseline "user")
    role_source = obj.get("role_source", "none")
    quotes = obj.get("evidence_quotes") or []

    # 1) Baseline "user": confiance basse et source none
    if role == "user":                     # CHANGED
        obj["role_confidence"] = min(float(obj.get("role_confidence", 0.0)), 0.2)  # CHANGED
        obj["role_source"] = "none"
        obj["role"] = "user"               # idempotent
        return obj

    # 2) Rôle spécifique → exige source forte + quote probante
    strong_sources = {"self_identified","profile_flair"}
    if (role_source not in strong_sources) or (not _has_explicit_role_evidence(role, quotes)):
        obj["role"] = "user"               # CHANGED (retombe sur user)
        obj["role_confidence"] = 0.2       # CHANGED
        obj["role_source"] = "none"
        return obj

    # 3) Confiance minimale
    if float(obj.get("role_confidence", 0.0)) < 0.6:
        obj["role"] = "user"               # CHANGED
        obj["role_confidence"] = 0.2       # CHANGED
        obj["role_source"] = "none"

    return obj


# ---------------- Classe principale ----------------
class CommentAnalyzer:
    """Une seule passe (avec traces et retries), puis agrégation utilisateur."""

    def __init__(self, model: str = "gpt-4.1-mini", max_workers: int = 4):
        self.client = OpenAI()  # nécessite OPENAI_API_KEY
        self.model = model
        self.max_workers = max_workers

    # ---------- PUBLIC ----------
    def analyze(self, comments: List[FlatComment]) -> Dict[str, Dict[str, Any]]:
        print(f"[INFO] Starting analysis | model={self.model} | n_comments={len(comments)}")

        # Skip propre si aucun commentaire
        if not comments:
            print("[INFO] No comments to analyze. Skipping.")
            return {}
        
        out: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_map = {ex.submit(self._analyze_one, c): c for c in comments}
            for fut in tqdm(fut_map, total=len(fut_map), desc=f"LLM: {self.model}"):
                c = fut_map[fut]
                try:
                    analysis = fut.result()
                except Exception as e:
                    print(f"[ERROR] Comment {c.id} by {c.author}: {e}")
                    analysis = _default_analysis(str(e))
                out[c.id] = {"comment": c, "analysis": analysis}
        print("[INFO] Analysis done.")
        return out

    
    def aggregate_by_user(self, results: Dict[str, Dict[str, Any]], post_link: str | None = None, post_title: str | None = None) -> pd.DataFrame:
        print("[INFO] Aggregating per user…")
        by_user: Dict[str, List[Dict[str, Any]]] = {}
        meta: Dict[str, Tuple[str, str]] = {}

        for _, payload in results.items():
            c: FlatComment = payload["comment"]
            a: Dict[str, Any] = payload["analysis"]
            by_user.setdefault(c.author, []).append(a)
            if c.author not in meta:
                meta[c.author] = (c.author_id, c.author_profile)

        rows = []
        for user, analyses in by_user.items():
            # CHANGED: rôles cibles = employee / partner
            role_scores = {"employee": 0.0, "partner": 0.0}  # CHANGED
            role_counts = {"employee": 0, "partner": 0}      # CHANGED
            STRONG = {"self_identified", "profile_flair"}

            for a in analyses:
                r = a.get("role", "user")                   # CHANGED
                src = a.get("role_source", "none")
                cconf = float(a.get("role_confidence", 0.0))
                if r in role_scores and src in STRONG and cconf >= 0.6:
                    role_scores[r] += cconf
                    role_counts[r] += 1

            # CHANGED: par défaut "user"
            role_inferred = "user"                          # CHANGED
            if any(role_counts.values()):
                best = max(role_scores, key=lambda k: role_scores[k])
                # garde les mêmes seuils d’inférence “forte”
                if role_counts[best] >= 2 or role_scores[best] >= 1.5:
                    role_inferred = best

            # Sentiment (inchangé)
            num = den = 0.0
            quotes, justifs = [], []
            for a in analyses:
                w = max(0.05, float(a.get("agentforce_confidence", 0.0)))
                num += float(a.get("agentforce_sentiment", 0.0)) * w
                den += w
                quotes += a.get("evidence_quotes", []) or []
                if a.get("justification"):
                    justifs.append(a["justification"])
            sentiment = (num / den) if den > 0 else 0.0

            quotes = list(dict.fromkeys(quotes))[:3]
            justifs = list(dict.fromkeys(justifs))[:3]
            author_id, author_profile = meta[user]

            rows.append({
                "user": user,
                "author_profile": author_profile,
                "role_inferred": role_inferred,              # CHANGED: peut valoir "user"
                "sentiment_agentforce": round(sentiment, 3),
                "justifications": " | ".join(justifs),
                "evidence_quotes": " | ".join(quotes),
                "n_comments": len(analyses),
            })

        df = pd.DataFrame(rows).sort_values(
            ["role_inferred", "sentiment_agentforce"], ascending=[True, False]
        )

        # ➜ Ajoute les colonnes en tête SANS impacter l'analyse
        if post_title is not None:
            df.insert(0, "post_title", post_title)
        if post_link is not None:
            df.insert(0, "post_link", post_link)

        print(f"[INFO] Aggregation complete. n_users={len(df)}")
        return df



    # ---------- INTERNALS ----------
    def _analyze_one(self, c: FlatComment) -> Dict[str, Any]:
        # Trace courte du commentaire
        preview = (c.body or "").replace("\n", " ")[:120]
        print(f"[TRACE] Send id={c.id} user={c.author} | parent={c.parent_id} | text='{preview}...'")

        user_text = f"""PARENT/ANCESTORS (up to 2):
        {c.ancestors_text if c.ancestors_text else "(none)"}

        CURRENT COMMENT by u/{c.author} ({c.permalink}):
        {c.body}
        """

        # Messages structurés (compat Responses API)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

        # Versions "chat" (fallback chat.completions)
        chat_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        # Retries exponentiels (réseaux/429)
        attempts, backoff = 0, 1.0
        while True:
            attempts += 1
            try:
                # ---- Responses API moderne ----
                r = self.client.responses.create(
                    model=self.model,
                    temperature=0.0,
                    response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                    input=messages,
                )
                text = extract_text_from_responses(r)
                if not text or not text.strip():
                    raise ValueError("Empty response from Responses API")
                print(f"[TRACE] OK (responses) id={c.id} | len={len(text)}")
                data = json.loads(text)
                return _coerce_analysis(data)

            except TypeError as e:
                # ex: lib qui n'accepte pas response_format pour responses.create
                if "response_format" in str(e):
                    print(f"[WARN] responses.create response_format unsupported → fallback chat.completions (id={c.id})")
                    chat = self.client.chat.completions.create(
                        model=self.model,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        messages=chat_messages,
                    )
                    txt = chat.choices[0].message.content
                    if not txt or not txt.strip():
                        raise ValueError("Empty response from Chat Completions")
                    print(f"[TRACE] OK (chat) id={c.id} | len={len(txt)}")
                    data = json.loads(txt)
                    return _coerce_analysis(data)
                else:
                    print(f"[ERROR] TypeError (id={c.id}): {e}")
                    raise

            except Exception as e:
                # Erreurs transientes: on retry un peu
                if attempts <= 3:
                    jitter = random.uniform(0, 0.3)
                    print(f"[WARN] attempt={attempts} failed for id={c.id}: {e} → retry in {backoff + jitter:.2f}s")
                    time.sleep(backoff + jitter)
                    backoff *= 2
                    continue
                print(f"[ERROR] Final failure id={c.id}: {e}")
                return _default_analysis(str(e))
