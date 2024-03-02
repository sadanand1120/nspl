NSLABELS_TWOWAY_NSINT = {
    "sidewalk": 1,
    "road": 2,
    "ELSE": 3,
    1: "sidewalk",
    2: "road",
    3: "ELSE",
}

DATASETINTstr_TO_DATASETLABELS = {
    "0": "unlabeled",
    "1": "sidewalk",
    "2": "road",
    "3": "ELSE"
}

DATASETLABELS_TO_NSLABELS = {
    "unlabeled": "ELSE",
    "sidewalk": "sidewalk",
    "road": "road",
    "ELSE": "ELSE",
}

# ----------------------------------------------------------------------

TERRAINMARKS_NSLABELS_TWOWAY_NSINT = {
    1: "parking_lines",
    2: "X",
    3: "ELSE",
    "parking_lines": 1,
    "X": 2,
    "ELSE": 3,
}

TERRAINMARKS_DATASETINTstr_TO_DATASETLABELS = {
    "0": "unlabeled",
    "1": "parking_lines",
    "2": "X",
    "3": "ELSE"
}

TERRAINMARKS_DATASETLABELS_TO_NSLABELS = {
    "unlabeled": "ELSE",
    "parking_lines": "parking_lines",
    "X": "X",
    "ELSE": "ELSE",
}
