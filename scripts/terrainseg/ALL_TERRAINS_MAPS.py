NSLABELS_TWOWAY_NSINT = {
    "sidewalk": 0,
    "road": 1,
    "ELSE": 2,
    0: "sidewalk",
    1: "road",
    2: "ELSE",
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
    0: "parking_lines",
    1: "X",
    2: "REST",
    "parking_lines": 0,
    "X": 1,
    "REST": 2,
}

TERRAINMARKS_DATASETINTstr_TO_DATASETLABELS = {
    "0": "unlabeled",
    "1": "parking_lines",
    "2": "X",
    "3": "REST"
}

TERRAINMARKS_DATASETLABELS_TO_NSLABELS = {
    "unlabeled": "REST",
    "parking_lines": "parking_lines",
    "X": "X",
    "REST": "REST",
}
