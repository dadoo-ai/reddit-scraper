#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reddit_json_folder_to_csv.py

Usage:
    python reddit_json_folder_to_csv.py --input /path/to/json_dir --output /path/to/out_dir --csv reddit_export.csv

Sortie: /path/to/out_dir/reddit_export.csv
Colonnes:
    info_url, info_username, url, category, title, body, flair, dataType, source_file
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# -----------------------------
# Utilitaires
# -----------------------------

def normalize_ws(text: Optional[str]) -> str:
    """Compacte les espaces et supprime les retours inutiles."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def coalesce(*values: Any, default: Any = "") -> Any:
    """Retourne le premier value non vide/non None."""
    for v in values:
        if v is not None and v != "":
            return v
    return default


def infer_category(item: Dict[str, Any]) -> str:
    """Essaye de déterminer la catégorie si manquante."""
    cat = item.get("category")
    if cat:
        return str(cat)

    cat = item.get("parsedCommunityName")
    if cat:
        return str(cat)

    comm = item.get("communityName")  # ex: "r/glasgow"
    if isinstance(comm, str) and comm:
        return comm.replace("r/", "").strip()

    return ""


def is_removed_body(text: Optional[str]) -> bool:
    """
    True si le body correspond à des contenus supprimés, notamment:
      - "[ Removed by Reddit ]" (avec/sans espaces/casse)
      - "[removed]"
      - "[deleted]"
    """
    if not isinstance(text, str):
        return False
    t = normalize_ws(text)
    # regex: accepte crochets facultatifs, espaces optionnels, casse insensible
    return bool(re.match(r"^\[?\s*(removed by reddit|removed|deleted)\s*\]?$", t, flags=re.IGNORECASE))


# -----------------------------
# Modèles de données
# -----------------------------

@dataclass
class UserInfo:
    url: str = ""
    username: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserInfo":
        return cls(
            url=str(d.get("url", "")),
            username=str(d.get("username", "")),
        )


@dataclass
class RedditItem:
    url: str
    category: str
    title: str
    body: str
    flair: str
    dataType: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RedditItem":
        data_type = str(d.get("dataType", "")).lower()  # "post" ou "comment"
        title = ""
        flair = ""

        if data_type == "post":
            title = normalize_ws(d.get("title"))
            flair = coalesce(d.get("flair"), default="")
        else:
            title = ""
            flair = ""

        return cls(
            url=str(d.get("url", "")),
            category=normalize_ws(infer_category(d)),
            title=title,
            body=normalize_ws(d.get("body")),
            flair=str(flair) if flair is not None else "",
            dataType=data_type,
        )


# -----------------------------
# Parseur d'un fichier JSON
# -----------------------------

class RedditJsonParser:
    """
    Chaque fichier contient une liste:
      - index 0 : info utilisateur (dataType='user')
      - index 1..n : posts et/ou commentaires
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> List[Dict[str, Any]]:
        raw = self.path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Le fichier ne contient pas une liste JSON: {self.path}")
        return data

    def parse(self) -> Tuple[UserInfo, List[RedditItem]]:
        data = self._load()
        if not data:
            return UserInfo(), []

        user_info_raw = data[0]
        user_info = UserInfo.from_dict(user_info_raw) if isinstance(user_info_raw, dict) else UserInfo()

        items: List[RedditItem] = []
        for raw in data[1:]:
            if not isinstance(raw, dict):
                continue
            dt = str(raw.get("dataType", "")).lower()
            if dt not in {"post", "comment"}:
                continue
            item = RedditItem.from_dict(raw)
            # ---- Filtre "Removed by Reddit" / removed / deleted ----
            if is_removed_body(item.body):
                continue
            items.append(item)

        return user_info, items


# -----------------------------
# Extraction d'un dossier
# -----------------------------

class FolderExtractor:
    def __init__(self, input_dir: Union[str, Path], output_dir: Union[str, Path], out_csv: str) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.out_csv = self.output_dir / out_csv

    def _iter_json_files(self) -> Iterable[Path]:
        yield from sorted(self.input_dir.glob("*.json"))

    def run(self) -> Path:
        rows: List[List[str]] = []
        header = [
            "info_url",
            "info_username",
            "url",
            "category",
            "title",
            "body",
            "flair",
            "dataType",
            "source_file",
        ]

        for path in self._iter_json_files():
            try:
                parser = RedditJsonParser(path)
                info, items = parser.parse()
            except Exception as e:
                print(f"[ERREUR] {path.name}: {e}")
                continue

            for it in items:
                rows.append([
                    info.url,
                    info.username,
                    it.url,
                    it.category,
                    it.title,
                    it.body,
                    it.flair,
                    it.dataType,
                    path.name,
                ])

        with self.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"[OK] CSV écrit: {self.out_csv} ({len(rows)} lignes)")
        return self.out_csv

