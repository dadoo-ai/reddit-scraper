#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple

import pandas as pd


# ===========================
#  Chargement & Nettoyage
# ===========================

class CSVLoader:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, dtype=str, keep_default_na=False).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        return df


class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text
        t = re.sub(r"<img[^>]*>", "", t, flags=re.IGNORECASE)                # HTML <img>
        t = re.sub(r"!\[.*?\]\(.*?\)", "", t)                             # Markdown images
        t = re.sub(r"http[s]?://\S+\.(?:jpg|jpeg|png|gif|webp|bmp|tiff?)\S*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\bImages?:\b.*", "", t, flags=re.IGNORECASE)           # trailing "Images: ..."
        t = t.replace("[ Removed by Reddit ]", "").replace("[removed]", "").replace("[deleted]", "")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def clean_series(df: pd.DataFrame, *cols: Optional[str]) -> pd.DataFrame:
        for c in cols:
            if c and c in df.columns:
                df[c] = df[c].apply(TextCleaner.clean)
        return df


# ===========================
#  Sentiment & Style
# ===========================

class SentimentAnalyzer:
    POS = {
        "great","good","helpful","love","like","works","working","effective","improve","fast","easy",
        "recommend","reliable","positive","success","win","adopt","adoption","valuable","benefit"
    }
    NEG = {
        "bad","worse","worst","hate","dislike","doesn't","dont","broken","bug","issue","slow","hard",
        "difficult","useless","negative","fail","failure","pain","problem","hype","overrated","cant","can't"
    }

    @staticmethod
    def _mentions(text: str, keys: List[str]) -> bool:
        tl = text.lower()
        return any(k in tl for k in keys)

    @staticmethod
    def score(texts: Iterable[str], topic_keywords: List[str]) -> Optional[float]:
        """Renvoie un score [-1,1] si le sujet est mentionné, sinon None."""
        joined = " ".join(t.lower() for t in texts if isinstance(t, str))
        if not SentimentAnalyzer._mentions(joined, topic_keywords):
            return None

        tokens = re.findall(r"[a-z']+", joined)
        idxs = [i for i,w in enumerate(tokens) if any(w.startswith(k) for k in topic_keywords)]
        scope: List[str] = []
        if idxs:
            for i in idxs:
                lo, hi = max(0,i-12), min(len(tokens), i+13)
                scope.extend(tokens[lo:hi])
        else:
            scope = tokens

        pos = sum(1 for t in scope if t in SentimentAnalyzer.POS)
        neg = sum(1 for t in scope if t in SentimentAnalyzer.NEG)
        if pos == 0 and neg == 0:
            return 0.0
        return (pos - neg) / max(1, pos + neg)


class CommunicationStyleDetector:
    TERMS = [
        "curious","opinionated","critical","supportive","factual",
        "humorous","help-seeking","experiential","speculative","neutral",
    ]

    @staticmethod
    def detect(texts: Iterable[str]) -> List[str]:
        t = " ".join(texts)
        tlow = t.lower()
        styles = set()

        # curious
        if "?" in t or re.search(r"\b(how|why|what|when|where|quel|pourquoi|comment)\b", tlow):
            styles.add("curious")
        # opinionated
        if re.search(r"\b(i think|i believe|imo|imho|je pense|à mon avis)\b", tlow):
            styles.add("opinionated")
        # critical / supportive
        if any(w in tlow for w in ["hate","worst","useless","hype","problem","broken","bug","pain","fail"]):
            styles.add("critical")
        if any(w in tlow for w in ["love","great","works","helpful","recommend","amazing"]):
            styles.add("supportive")
        # factual
        if re.search(r"\b\d{2,}\b", tlow) or "http" in tlow or "code" in tlow:
            styles.add("factual")
        # humorous
        if re.search(r"\b(lol|haha|lmao|mdr)\b", tlow):
            styles.add("humorous")
        # help-seeking
        if re.search(r"\b(can someone|any advice|need help|besoin d'aide|how do i)\b", tlow):
            styles.add("help-seeking")
        # experiential
        if re.search(r"\b(we use|we used|i use|i used|on my team|dans mon équipe)\b", tlow):
            styles.add("experiential")
        # speculative
        if re.search(r"\b(might|could|maybe|peut[- ]?être|pourrait)\b", tlow):
            styles.add("speculative")
        # fallback
        if not styles:
            styles.add("neutral")

        return sorted(styles)


# ===========================
#  Loisirs (Hobbies)
# ===========================

class HobbiesExtractor:
    """Retourne (labels de loisirs, catégories correspondantes, mots-clés correspondants)."""

    # Subreddit / catégorie -> libellé normalisé
    CAT_MAP: Dict[str, str] = {
        "chiweenie": "dogs",
        "dachshund": "dogs",
        "askvet": "pets",
        "seestar": "astrophotography",
        "askastrophotography": "astrophotography",
        "astrophotography": "astrophotography",
        "labdiamond": "jewelry",
        "labgrowndiamonds": "jewelry",
        "engagementrings": "jewelry",
        "engagementringdesigns": "jewelry",
        "weddingring": "jewelry",
        "tattooadvice": "tattoos",
        "antitrump": "politics",
        "lashextensions": "beauty",
        "thegoldengirls": "tv",
        "90dayfianceuncensored": "tv",
        "badmagicproductions": "podcasts",
    }

    # Mots-clés texte -> libellé
    KW_MAP: Dict[str, str] = {
        "dog": "dogs", "dogs": "dogs", "puppy": "dogs", "chiweenie": "dogs", "dachshund": "dogs",
        "vet": "pets", "animal": "pets", "pet": "pets",
        "tattoo": "tattoos", "ink": "tattoos",
        "ring": "jewelry", "diamond": "jewelry", "jewelry": "jewelry", "engagement": "jewelry", "wedding": "jewelry",
        "astrophotography": "astrophotography", "telescope": "astrophotography", "seestar": "astrophotography",
        "rokinon": "astrophotography", "mount": "astrophotography", "polar align": "astrophotography",
        "camera": "photography", "photo": "photography",
        "lash": "beauty", "eyelash": "beauty",
        "podcast": "podcasts",
        "politics": "politics",
    }

    @staticmethod
    def from_category(values: Iterable[str]) -> Tuple[set, set]:
        hobbies = set()
        matched_cats = set()
        for v in values:
            k = str(v).strip().lower()
            if not k:
                continue
            label = HobbiesExtractor.CAT_MAP.get(k)
            if label:
                hobbies.add(label)
                matched_cats.add(k)
        return hobbies, matched_cats

    @staticmethod
    def from_texts(texts: Iterable[str]) -> Tuple[set, set]:
        joined = " ".join(t for t in texts if isinstance(t, str)).lower()
        hobbies = set()
        matched_kws = set()
        for kw, label in HobbiesExtractor.KW_MAP.items():
            if kw in joined:
                hobbies.add(label)
                matched_kws.add(kw)
        return hobbies, matched_kws


# ===========================
#  Démographie / Statut (preuves explicites uniquement)
# ===========================

class DemographicsExtractor:
    AGE_PATTERNS = [
        (re.compile(r"\b(\d{2})\s*(?:yo|y/o|years old|ans)\b", re.I), int),
    ]
    JOB_LEVEL_TERMS: Dict[str, List[str]] = {
        "intern":["intern","stagiaire"],
        "junior":["junior","jr"],
        "mid":["mid","intermediate"],
        "senior":["senior","sr","senior."],
        "lead":["lead","tech lead","staff"],
        "manager":["manager","managing"],
        "director":["director","directeur"],
        "vp":["vp","vice president"],
        "cxo":["cto","ceo","cpo","cfo","coo"],
    }
    EMPLOYMENT_STATUS_TERMS: Dict[str, List[str]] = {
        "employed":["i work at","i'm employed","je travaille chez","salarié"],
        "unemployed":["unemployed","sans emploi"],
        "self-employed":["self-employed","indépendant","freelance"],
        "contractor":["contractor","contract"],
        "student":["student","étudiant","apprentice","alternant"],
    }
    POLITICAL_TERMS: Dict[str, List[str]] = {
        "left":["left-wing","leftist","gauche","socialist","democrat","progressive","liberal"],
        "right":["right-wing","rightist","droite","conservative","republican","libertarian"],
        "green":["green party","écologiste"],
    }
    SELF_ID_PAT = re.compile(r"\b(i am|i'm|je suis)\b", re.I)

    @staticmethod
    def age_range(texts: Iterable[str]) -> Optional[str]:
        joined = " ".join(texts)
        for pat, caster in DemographicsExtractor.AGE_PATTERNS:
            m = pat.search(joined)
            if m:
                try:
                    age = caster(m.group(1))
                    if age < 18:
                        return None
                    if age <= 24: return "18-24"
                    if age <= 34: return "25-34"
                    if age <= 44: return "35-44"
                    if age <= 54: return "45-54"
                    return "55+"
                except:
                    pass
        return None

    @staticmethod
    def job_level_with_evidence(texts: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
        tlow = " ".join(texts).lower()
        for lvl, keys in DemographicsExtractor.JOB_LEVEL_TERMS.items():
            for k in keys:
                if re.search(rf"\b{re.escape(k)}\b", tlow):
                    return lvl, k
        return None, None

    @staticmethod
    def employment_status_with_evidence(texts: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
        tlow = " ".join(texts).lower()
        for st, keys in DemographicsExtractor.EMPLOYMENT_STATUS_TERMS.items():
            for k in keys:
                if k in tlow:
                    return st, k
        return None, None

    @staticmethod
    def political_leaning_with_evidence(texts: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
        tlow = " ".join(texts).lower()
        if not DemographicsExtractor.SELF_ID_PAT.search(tlow):
            return None, None
        for label, keys in DemographicsExtractor.POLITICAL_TERMS.items():
            for k in keys:
                if k in tlow:
                    return label, k
        return None, None

    @staticmethod
    def job_level(texts: Iterable[str]) -> Optional[str]:
        lvl, _ = DemographicsExtractor.job_level_with_evidence(texts)
        return lvl

    @staticmethod
    def employment_status(texts: Iterable[str]) -> Optional[str]:
        st, _ = DemographicsExtractor.employment_status_with_evidence(texts)
        return st

    @staticmethod
    def political_leaning(texts: Iterable[str]) -> Optional[str]:
        pol, _ = DemographicsExtractor.political_leaning_with_evidence(texts)
        return pol


# ===========================
#  Justification (FR)
# ===========================

def _build_justification(hobbies, hob_cats, hob_kws, job_lvl, job_kw, pol, pol_kw, emp, emp_kw, age_rng, age_src=None):
    lines = []
    # Loisirs
    if hobbies:
        if hob_cats:
            lines.append("Loisirs : déduits des catégories " + ", ".join(sorted(hob_cats)) + ".")
        if hob_kws and len(lines) < 2:
            lines.append("Mots-clés repérés pour les loisirs : " + ", ".join(sorted(hob_kws)) + ".")
        if not hob_cats and not hob_kws:
            lines.append("Loisirs : mentions explicites dans les textes.")
    else:
        lines.append("Loisirs : aucune catégorie ou mot-clé explicite détecté.")
    # Niveau de poste
    if job_lvl:
        lines.append(f"Niveau de poste : terme détecté « {job_kw} ». ")
    else:
        lines.append("Niveau de poste : aucune auto‑description explicite.")
    # Politique
    if pol:
        lines.append(f"Politique : auto‑identification près de « {pol_kw} ». ")
    else:
        lines.append("Politique : aucune auto‑identification explicite.")
    # Âge
    if age_rng:
        lines.append("Âge : tranche déduite d’un motif chiffré explicite.")
    # Conserver 3–4 phrases courtes
    return " ".join(lines[:4])


# ===========================
#  Agrégation par utilisateur
# ===========================

@dataclass
class ColumnsMap:
    user: str
    title: Optional[str]
    body: Optional[str]
    kind: Optional[str]
    category: Optional[str]
    flair: Optional[str]


class UserAggregator:
    def __init__(self, cols: ColumnsMap):
        self.cols = cols

    @staticmethod
    def _safe_join(values: List[str], max_len: int = 2000) -> str:
        seen = set()
        uniq: List[str] = []
        for v in values:
            v = TextCleaner.clean(v)
            if not v or v in seen:
                continue
            seen.add(v)
            uniq.append(v)
        out = " | ".join(uniq)
        return (out[:max_len] + " …") if len(out) > max_len else out

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cols

        # split post vs comment
        if c.kind and c.kind in df.columns:
            is_post = df[c.kind].str.lower().str.contains("post")
            is_comment = df[c.kind].str.lower().str.contains("comment")
        else:
            is_post = df[c.title].astype(bool) if c.title else pd.Series([False]*len(df), index=df.index)
            is_comment = ~is_post

        rows: List[Dict[str,str]] = []
        for user, g in df.groupby(c.user):
            titles = g.loc[is_post[g.index], c.title].tolist() if c.title else []
            post_bodies = g.loc[is_post[g.index], c.body].tolist() if c.body else []
            comment_bodies = g.loc[is_comment[g.index], c.body].tolist() if c.body else []

            # remove blanks
            post_bodies = [t for t in post_bodies if t.strip()]
            comment_bodies = [t for t in comment_bodies if t.strip()]

            cats: List[str] = []
            if c.category:
                cats += g.loc[is_post[g.index], c.category].tolist()
            if c.flair:
                cats += g.loc[is_comment[g.index], c.flair].tolist()

            all_texts = (titles or []) + (post_bodies or []) + (comment_bodies or [])

            agentforce = SentimentAnalyzer.score(all_texts, ["agentforce"])  # None si non-mention
            salesforce = SentimentAnalyzer.score(all_texts, ["salesforce"])  # None si non-mention
            styles = CommunicationStyleDetector.detect(all_texts)

            age_rng = DemographicsExtractor.age_range(all_texts) or ""
            job_lvl, job_kw = DemographicsExtractor.job_level_with_evidence(all_texts)
            emp, emp_kw = DemographicsExtractor.employment_status_with_evidence(all_texts)
            pol, pol_kw = DemographicsExtractor.political_leaning_with_evidence(all_texts)

            job_lvl = job_lvl or ""
            emp = emp or ""
            pol = pol or ""

            hob_from_cat, hob_cats = HobbiesExtractor.from_category(cats)
            hob_from_txt, hob_kws = HobbiesExtractor.from_texts(all_texts)
            hobbies = sorted(hob_from_cat | hob_from_txt)

            justification = _build_justification(
                hobbies, hob_cats, hob_kws, job_lvl, job_kw, pol, pol_kw, emp, emp_kw, age_rng
            )

            rows.append({
                "user": user,
                "titre_des_posts": self._safe_join(titles),
                "message_de_post": self._safe_join(post_bodies),
                "message_des_commentaires": self._safe_join(comment_bodies),
                "categorie": self._safe_join(cats),
                "sentiment_agentforce": "" if agentforce is None else f"{agentforce:.3f}",
                "sentiment_salesforce": "" if salesforce is None else f"{salesforce:.3f}",
                "communication_style": ", ".join(styles),
                "age_range": age_rng,
                "job_level": job_lvl,
                "employment_status": emp,
                "political_leaning": pol,
                "hobbies": ", ".join(hobbies),
                "justification": justification,
            })

        return pd.DataFrame(rows).sort_values("user")


# ===========================
#  CLI
# ===========================

def pick_first(df: pd.DataFrame, *names: str) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def run(input_csv: Path, output_csv: Path, log_level: Optional[str] = None) -> None:
    if log_level:
        logging.basicConfig(level=getattr(logging, (log_level or "INFO").upper(), logging.INFO),
                            format='[%(levelname)s] %(message)s')
        print(f"[INFO] Niveau de log fixé à {log_level}")
    df = CSVLoader(input_csv).load()
    # Mapping flexible des colonnes
    user = pick_first(df, "username", "user", "author", "info_username")
    title = pick_first(df, "title", "post_title")
    body = pick_first(df, "body", "message", "post_message", "comment_body")
    kind = pick_first(df, "datatype", "type", "kind", "datatype")
    category = pick_first(df, "category")
    flair = pick_first(df, "flair")

    print(f"[INFO] Colonnes choisies -> user={user}, title={title}, body={body}, kind={kind}, category={category}, flair={flair}")

    if not user:
        raise SystemExit("Aucune colonne utilisateur trouvée (attendu : username, user, author, info_username).")

    # Nettoyage des colonnes utiles
    df = TextCleaner.clean_series(df, title, body, category, flair)

    print("[INFO] Nettoyage du texte effectué")

    agg = UserAggregator(ColumnsMap(
        user=user, title=title, body=body, kind=kind, category=category, flair=flair
    ))
    out = agg.aggregate(df)
    out.to_csv(output_csv, index=False)
    print(f"[OK] {len(out)} lignes écrites → {output_csv}")


