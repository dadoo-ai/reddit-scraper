from __future__ import annotations
import orjson as json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

# ----- Donnée aplatie -----
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
    """Aplatis ton JSON (post + comments imbriqués) en une liste de commentaires
    en conservant le contexte (parent/grand-parent)."""

    def __init__(self, max_ancestor_hops: int = 2, dedupe: bool = True):
        self.max_ancestor_hops = max_ancestor_hops
        self.dedupe = dedupe

    def load(self, path: str | Path) -> Dict[str, Any]:
        return json.loads(Path(path).read_bytes())

    def flatten(self, root: Dict[str, Any]) -> List[FlatComment]:
        parent_map: Dict[str, Optional[str]] = {}
        node_map: Dict[str, Dict[str, Any]] = {}
        path_map: Dict[str, List[str]] = {}
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
