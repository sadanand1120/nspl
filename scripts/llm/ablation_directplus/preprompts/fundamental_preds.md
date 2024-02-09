# Implemented fundamental predicates
The following fundamental predicates are implemented in the codebase already. You are allowed to use them. Note that in the following predicates, `loc` refers to a `(x, y)` location in global map frame, while pixel_loc refers to a `(x, y)` location in the pixel coordinate system.

```python
from typing import Optional, Tuple

def is_on(pixel_loc: Tuple[int, int], terrain: str) -> bool:
    """Returns True if the terrain at the pixel location matches the terrain mentioned"""

def is_far_away_from(pixel_loc: Tuple[int, int], object: str, alpha: float) -> bool:
    """Returns True if the pixel location is far enough, according to the parameter, from the nearest instance of the object class"""

def is_in_the_way(pixel_loc: Tuple[int, int]) -> bool:
    """Returns True if all the four points 1 m away from the pixel location along the x, y axes in the global map frame are traversable, else False"""

def is_in_front_of(pixel_loc: Tuple[int, int], object: str, alpha: float) -> bool:
    """Returns True if the pixel location falls on the object after it is extended by a parameter in the map frame"""

def is_close_to(pixel_loc: Tuple[int, int], object: str, alpha: float) -> bool:
    """Returns True if the pixel location is not far away, according to the parameter, from the nearest object of the object class"""

def is_too_inclined(pixel_loc: Tuple[int, int], alpha: float) -> bool:
    """Returns True if the elevation gradient at the pixel location is more than the threshold"""
```