from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

Point3 = Tuple[int, int, int]                 # (molmo_id, x, y)
Point4 = Tuple[int, int, int, str]            # (molmo_id, x, y, label)
Point = Union[Point3, Point4]


def points_to_jsonable(points: List[Point]) -> List[Dict[str, Any]]:
    """
    Convert:
      [(molmo_id, x, y), ...] OR [(molmo_id, x, y, label), ...]
    into:
      [{"molmo_id":..., "x":..., "y":..., "label":...}, ...]
    """
    out: List[Dict[str, Any]] = []
    for p in points:
        if len(p) == 3:
            molmo_id, x, y = p
            out.append({"molmo_id": int(molmo_id), "x": int(x), "y": int(y)})
        else:
            molmo_id, x, y, label = p
            out.append({"molmo_id": int(molmo_id), "x": int(x), "y": int(y), "label": str(label)})
    return out