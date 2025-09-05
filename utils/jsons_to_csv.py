from __future__ import annotations
import orjson as json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple
import logging
import pandas as pd

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("flatten_json_to_csv.log", encoding="utf-8")]
)
log = logging.getLogger("flatten_json_to_csv")


# ====== DONNÉES APLATIES ======
@dataclass
class FlatComment:
    id: str
    author: str
    author_id: str
    author_profile: str
    permalink: str
    body: str
    parent_id: Optional[str]
    ancestors_text: str
    path: List[str]


class RedditThreadFlattener:
    """Aplatis un JSON (post + comments imbriqués) en une liste de commentaires
    en conservant le contexte (parent/grand-parent)."""

    def __init__(self, max_ancestor_hops: int = 2, dedupe: bool = True):
        self.max_ancestor_hops = max_ancestor_hops
        self.dedupe = dedupe

    def load(self, path: str | Path) -> Dict[str, Any] | List[Any]:
        return json.loads(Path(path).read_bytes())

    def flatten(self, root: Dict[str, Any]) -> List[FlatComment]:
        parent_map: Dict[str, Optional[str]] = {}
        node_map: Dict[str, Dict[str, Any]] = {}
        path_map: Dict[str, List[str]] = {}
        # On attend root["comments"] dans ton schéma objet
        self._walk(root.get("comments", []), None, [], parent_map, node_map, path_map)

        flat: List[FlatComment] = []
        seen: set[str] = set()
        for cid, node in node_map.items():
            if self.dedupe and cid in seen:
                continue
            seen.add(cid)
            flat.append(
                FlatComment(
                    id=cid,
                    author=node.get("author") or "unknown",
                    author_id=node.get("authorId") or "",
                    author_profile=node.get("authorProfile") or "",
                    permalink=node.get("link") or "",
                    body=(node.get("commentBody") or "").strip(),
                    parent_id=parent_map.get(cid),
                    ancestors_text=self._ancestors_text(node_map, parent_map, parent_map.get(cid)),
                    path=path_map.get(cid, []),
                )
            )
        return flat

    def _walk(self, nodes, parent_id, path, parent_map, node_map, path_map):
        for n in nodes or []:
            cid = n.get("commentId")
            if not cid:
                continue
            node_map[cid] = n
            parent_map[cid] = parent_id
            cur_path = path + [cid]
            path_map[cid] = cur_path
            self._walk(n.get("replies", []), cid, cur_path, parent_map, node_map, path_map)

    def _ancestors_text(self, node_map, parent_map, start_parent) -> str:
        texts: List[str] = []
        cur = start_parent
        hops = 0
        while cur and cur in node_map and hops < self.max_ancestor_hops:
            body = (node_map[cur].get("commentBody") or "").strip()
            if body:
                texts.append(body)
            cur = parent_map.get(cur)
            hops += 1
        texts.reverse()
        return "\n\n---\n\n".join(texts)


# ====== PIPELINE DOSSIER -> CSV ======
REMOVED_MARKERS = {
    "[removed by reddit]", "[removed]", "[deleted]",
    "removed by reddit", "removed", "deleted"
}

def is_removed(text: str) -> bool:
    t = (text or "").strip().lower()
    t = t.strip("[](){}<> ").replace("  ", " ")
    return t in REMOVED_MARKERS


def clean_subreddit(val: Optional[str]) -> str:
    if not val:
        return ""
    s = str(val).strip()
    return s[2:] if s.startswith("r/") else s


def list_json_files(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.json") if p.is_file()])
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        return [input_path]
    return []


def extract_rows_from_object_schema(data: Dict[str, Any], src: Path) -> List[Dict[str, Any]]:
    """
    Schéma principal géré:
    {
      "info": {... post ...},
      "comments": [ {..., replies:[...]}, ... ]
    }
    """
    rows: List[Dict[str, Any]] = []
    info = data.get("info") or {}

    # --- Ligne POST ---
    post_title = str(info.get("postTitle") or "")
    post_body = str(info.get("postMessage") or "")
    if not is_removed(post_body) and (post_title or post_body):
        rows.append({
            "dataType": "post",
            "url": str(info.get("postLink") or info.get("url") or ""),
            "category": clean_subreddit(info.get("subreddit")),
            "title": post_title,
            "body": post_body,
            "flair": str(info.get("postLabel") or ""),
            "author": str(info.get("author") or ""),
            "author_id": str(info.get("authorId") or ""),
            "author_profile": str(info.get("authorProfile") or ""),
            "parent_id": "",
            "ancestors_text": "",
            "path": "",
            "publishingDate": str(info.get("publishingDate") or ""),
            "source_file": str(src),
        })

    # --- Commentaires aplatis via flattener ---
    flattener = RedditThreadFlattener(max_ancestor_hops=2, dedupe=True)
    flat_comments = flattener.flatten(data)

    for fc in flat_comments:
        if not fc.body or is_removed(fc.body):
            continue
        rows.append({
            "dataType": "comment",
            "url": fc.permalink,
            "category": "",           # pas de subreddit au niveau commentaire dans ce schéma
            "title": "",              # pas de titre pour les commentaires
            "body": fc.body,
            "flair": "",
            "author": fc.author,
            "author_id": fc.author_id,
            "author_profile": fc.author_profile,
            "parent_id": fc.parent_id or "",
            "ancestors_text": fc.ancestors_text,
            "path": " > ".join(fc.path),
            "publishingDate": "",     # non fourni au niveau commentaire dans ce bloc, à ajouter si besoin
            "source_file": str(src),
        })

    return rows


def extract_rows_from_list_schema(data: List[Any], src: Path) -> List[Dict[str, Any]]:
    """
    Schéma fallback:
    [ info_dict, item_dict, item_dict, ... ]
    où item peut être un post ou un commentaire aplatissable.
    """
    rows: List[Dict[str, Any]] = []
    info = data[0] if data and isinstance(data[0], dict) else {}

    # On essaie d’extraire un post depuis info si présent
    post_title = str(info.get("postTitle") or info.get("title") or "")
    post_body = str(info.get("postMessage") or info.get("selftext") or info.get("body") or "")
    post_url = str(info.get("postLink") or info.get("url") or "")
    if (post_title or post_body) and not is_removed(post_body):
        rows.append({
            "dataType": "post",
            "url": post_url,
            "category": clean_subreddit(info.get("subreddit") or info.get("category")),
            "title": post_title,
            "body": post_body,
            "flair": str(info.get("postLabel") or info.get("flair") or ""),
            "author": str(info.get("author") or ""),
            "author_id": str(info.get("authorId") or ""),
            "author_profile": str(info.get("authorProfile") or ""),
            "parent_id": "",
            "ancestors_text": "",
            "path": "",
            "publishingDate": str(info.get("publishingDate") or ""),
            "source_file": str(src),
        })

    # Puis tous les items suivants (post/comment)
    for it in data[1:]:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or it.get("postTitle") or "")
        body = str(it.get("body") or it.get("selftext") or it.get("postMessage") or it.get("commentBody") or "")
        if not body and not title:
            continue
        if is_removed(body):
            continue

        url = str(it.get("url") or it.get("postLink") or it.get("link") or "")
        flair = str(it.get("flair") or it.get("link_flair_text") or it.get("postLabel") or "")
        category = clean_subreddit(it.get("category") or it.get("subreddit"))
        data_type = it.get("dataType") or ("comment" if "commentBody" in it else "post")

        rows.append({
            "dataType": str(data_type),
            "url": url,
            "category": category,
            "title": title if data_type == "post" else "",
            "body": body,
            "flair": flair if data_type == "post" else "",
            "author": str(it.get("author") or ""),
            "author_id": str(it.get("authorId") or ""),
            "author_profile": str(it.get("authorProfile") or ""),
            "parent_id": "",
            "ancestors_text": "",
            "path": "",
            "publishingDate": str(it.get("publishingDate") or ""),
            "source_file": str(src),
        })
    return rows


def extract_file(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_bytes())
    try:
        if isinstance(raw, dict) and ("info" in raw or "comments" in raw):
            return extract_rows_from_object_schema(raw, path)
        elif isinstance(raw, list):
            return extract_rows_from_list_schema(raw, path)
        else:
            log.debug(f"Format non reconnu: {path}")
            return []
    except Exception as e:
        log.warning(f"Erreur d’extraction sur {path}: {e}")
        return []


def run_folder_to_csv(input_path: Path, output_csv: Path) -> Path:
    files = list_json_files(input_path)
    log.info(f"{len(files)} JSON détectés sous {input_path}")

    all_rows: List[Dict[str, Any]] = []
    for f in files:
        rows = extract_file(f)
        all_rows.extend(rows)

    # Colonnes stables & ordre
    cols = [
        "dataType", "url", "category", "title", "body", "flair",
        "author", "author_id", "author_profile",
        "parent_id", "ancestors_text", "path",
        "publishingDate", "source_file"
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    # Si vraiment vide, on écrit un CSV vide mais avec en-têtes
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    log.info(f"CSV écrit: {output_csv} ({len(df)} lignes)")
    return output_csv


# ============== CLI ==============
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Aplatis des JSON Reddit (post + commentaires imbriqués) en CSV (post + comments)."
    )
    p.add_argument("--input-path", required=True, type=Path, help="Dossier OU fichier JSON unique.")
    p.add_argument("--output-csv", required=True, type=Path, help="Chemin du CSV de sortie.")
    args = p.parse_args()

    run_folder_to_csv(args.input_path, args.output_csv)


# utilisation
"""
# Un seul fichier
python utils/jsons_to_csv.py \
  --input-path results/comments/exemple.json \
  --output-csv results/reddit_posts.csv

# Un dossier complet (récursif)
python utils/jsons_to_csv.py \
  --input-path results/comments/ \
  --output-csv results/reddit_posts.csv
"""