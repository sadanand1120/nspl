# Implemented fundamental predicates
The following fundamental predicates are implemented in the codebase already. You are allowed to use them. Note that in the following predicates, `loc` refers to a `(x, y)` location in global map frame, while pixel_loc refers to a `(x, y)` location in the pixel coordinate system.

```python
from typing import Optional, Tuple

def project_map_to_pixel(loc: Tuple[float, float]) -> Optional[Tuple[int, int]]:
    """Given a location tuple in the global map frame, returns the corresponding pixel location tuple if in field of view, else None."""

def project_pixel_to_map(pixel_loc: Tuple[int, int]) -> Tuple[float, float]:
    """Given a pixel location tuple, returns the corresponding location in the global map frame."""

def distance_to_nearest_object(loc: Tuple[float, float], object_class: str) -> float:
    """Given a location tuple and an object class, returns the real-world distance of the loc to the nearest object of that class."""

def extend(flat_object: str, alpha: float) -> None:
    """Given a flat object, extends all instances of the object in the normal direction in global map frame by alpha amount."""

def terrain_at(pixel_loc: Tuple[int, int]) -> str:
    """Given a pixel location tuple, returns the terrain class at that location."""

def is_a_terrain(pixel_loc: Tuple[int, int]) -> bool:
    """Given a pixel location tuple, returns whether the location is a terrain."""

def is_traversable(terrain: str) -> bool:
    """Given a terrain class, returns whether the terrain is traversable."""

def slope_at(loc: Tuple[float, float]) -> float:
    """Given a location tuple, returns the slope at that location."""
```