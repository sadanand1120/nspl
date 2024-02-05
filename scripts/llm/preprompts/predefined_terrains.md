## Predefined terrain classes
Assume only the following terrain classes exist, and they are mapped to a corresponding integer label:
- "sidewalk": 0
- "grass": 1
- "speedway": 2
- "tiles": 3
- "bricks": 4
- "road": 5
- "NAT": 6  # Not A Terrain
- "rest": 7  # Rest of the terrains
If you see some other terrain, first try to **meaningfully** map it to one of the above non-rest terrain classes. If you can't, then you can use the `rest` terrain class.
