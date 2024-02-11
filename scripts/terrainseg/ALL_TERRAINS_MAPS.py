NSLABELS_TWOWAY_NSINT = {
    "sidewalk": 0,
    "grass": 1,
    "speedway": 2,
    "tiles": 3,
    "bricks": 4,
    "road": 5,
    "NAT": 6,  # Not A Terrain
    "rest": 7,  # Rest of the terrains
    0: "sidewalk",
    1: "grass",
    2: "speedway",
    3: "tiles",
    4: "bricks",
    5: "road",
    6: "NAT",
    7: "rest",
}

DATASETINTstr_TO_DATASETLABELS = {
    "0": "unlabeled",
    "1": "NAT",
    "2": "concrete",
    "3": "grass",
    "4": "speedway bricks",
    "5": "steel",
    "6": "rough concrete",
    "7": "dark bricks",
    "8": "road",
    "9": "rough red sidewalk",
    "10": "tiles",
    "11": "red bricks",
    "12": "concrete tiles",
    "13": "REST"
}

DATASETLABELS_TO_NSLABELS = {
    "unlabeled": "NAT",
    "NAT": "NAT",
    "concrete": "sidewalk",
    "grass": "grass",
    "speedway bricks": "speedway",
    "steel": "speedway",
    "rough concrete": "sidewalk",
    "dark bricks": "bricks",
    "road": "road",
    "rough red sidewalk": "sidewalk",
    "tiles": "tiles",
    "red bricks": "bricks",
    "concrete tiles": "tiles",
    "REST": "rest",
}

# Following defs in terms of NSLABELS
# IS_A_TERRAIN => simply means not NAT
# TRAVESABLE_TERRAINS => means is a terrain + is traversable for agent (i.e., like not grass)
# NON_TRAVERSABLE_TERRAINS => means is a terrain + is not traversable for agent (i.e., like grass)

NSLABELS_TRAVERSABLE_TERRAINS = [
    "sidewalk",
    "speedway",
    "road",
    "bricks",
    "tiles",
]

NSLABELS_NON_TRAVERSABLE_TERRAINS = [
    "grass",
    "rest",
]
