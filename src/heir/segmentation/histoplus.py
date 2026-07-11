"""Parser for the locally installed HistoPLUS tiled JSON contract."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class HistoPLUSCell:
    nucleus_id: str
    centroid_native: Tuple[float, float]
    polygon_native: np.ndarray
    cell_type: str
    confidence: float
    tile_x: int
    tile_y: int


def read_histoplus_json(
    path: Union[str, Path], slide_id: Optional[str] = None
) -> List[HistoPLUSCell]:
    """Read tiled coordinates and restore a slide-global native-pixel frame."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tiles = payload.get("cell_masks")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("HistoPLUS JSON has no cell_masks")
    inferred_slide = slide_id or source.stem.replace("_cells", "")
    result: List[HistoPLUSCell] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            raise ValueError("cell_masks entries must be mappings")
        width = int(tile.get("width", 0))
        tile_x = int(tile.get("x", 0))
        tile_y = int(tile.get("y", 0))
        if width <= 0:
            raise ValueError("HistoPLUS tile width must be positive")
        origin = np.asarray((tile_x * width, tile_y * width), dtype=np.float64)
        masks = tile.get("masks", [])
        if not isinstance(masks, list):
            raise ValueError("tile masks must be a list")
        for mask in masks:
            local_polygon = np.asarray(mask.get("coordinates"), dtype=np.float64)
            local_centroid = np.asarray(mask.get("centroid"), dtype=np.float64)
            if local_polygon.ndim != 2 or local_polygon.shape[1] != 2 or len(local_polygon) < 3:
                raise ValueError("each HistoPLUS polygon needs at least three xy vertices")
            if local_centroid.shape != (2,) or not np.isfinite(local_centroid).all():
                raise ValueError("each HistoPLUS centroid needs two finite values")
            confidence = float(mask.get("confidence", mask.get("score", 1.0)))
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("HistoPLUS confidence must be in [0, 1]")
            index = len(result)
            result.append(
                HistoPLUSCell(
                    nucleus_id="%s:%d" % (inferred_slide, index),
                    centroid_native=tuple((local_centroid + origin).tolist()),
                    polygon_native=local_polygon + origin,
                    cell_type=str(mask.get("cell_type", "unknown")),
                    confidence=confidence,
                    tile_x=tile_x,
                    tile_y=tile_y,
                )
            )
    return result
