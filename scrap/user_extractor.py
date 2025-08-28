from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd


class RedditUserExtractor:
    """Extrait les utilisateurs uniques (authorId, author, authorProfile) d'un post Reddit JSON."""

    def __init__(self) -> None:
        self._users: Dict[str, Dict[str, Optional[str]]] = {}

    def _add_user(self, node: dict) -> None:
        if not node:
            return
        uid = node.get("authorId")
        if not uid:
            return
        # ajoute ou fusionne
        self._users[uid] = {
            "authorId": uid,
            "author": node.get("author"),
            "authorProfile": node.get("authorProfile"),
        }

    def _walk_comments(self, comments: list) -> None:
        for c in comments:
            self._add_user(c)
            replies = c.get("replies", [])
            if isinstance(replies, list):
                self._walk_comments(replies)

    def ingest_json(self, record: dict) -> None:
        """Ingestion d'un enregistrement JSON unique (avec clés info + comments)."""
        self._add_user(record.get("info", {}))
        self._walk_comments(record.get("comments", []))

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self._users.values()))
        if df.empty:
            return pd.DataFrame(columns=["authorId", "author", "authorProfile"])
        return df.drop_duplicates("authorId").sort_values("author").reset_index(drop=True)

    def to_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        data = self.to_dataframe().to_dict(orient="records")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def to_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        self.to_dataframe().to_csv(path, index=False, encoding="utf-8")
        return path


