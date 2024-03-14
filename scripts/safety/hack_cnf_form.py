import os
nspl_root_dir = os.environ.get("NSPL_REPO")
from ldips_datagen import LDIPSdatagenSingleEx
from synthesis.synthesize import LDIPS_synthesize, read_data


DATA_DISTILLED = {
    1: [  # It is safe since it is on a sidewalk, and is far from any person and the pole. It is not on grass.
        "terrain(pixel_loc) == 0",
        "distance_to_person(pixel_loc) > ??",
        "distance_to_pole(pixel_loc) > ??",
    ],
    2: [  # This looks good since it is on a sidewalk, and is far from the approaching person and the pole.
        "terrain(pixel_loc) == 0",
        "distance_to_person(pixel_loc) > ??",
        "distance_to_pole(pixel_loc) > ??",
    ],
    3: [  # This is not safe since it is not far from bushes, even though its on a sidewalk.
        "terrain(pixel_loc) == 0",
        "distance_to_bush(pixel_loc) > ??",
    ],
    4: [  # It seems reasonable since it is on a sidewalk, far from the pole, and is not in the way.
        "terrain(pixel_loc) == 0",
        "distance_to_pole(pixel_loc) > ??",
        "in_the_way(pixel_loc) == 0",
    ],
    5: [  # This is safe as it is on a sidewalk, is not in front of the entrance, and is just far enough from the bushes.
        "terrain(pixel_loc) == 0",
        "frontal_distance_entrance(pixel_loc) > ??",
        "distance_to_bush(pixel_loc) > ??",
    ],
    6: [  # It seems reasonable since it is on a sidewalk, far from the pole, and is not in the way.
        "terrain(pixel_loc) == 0",
        "distance_to_pole(pixel_loc) > ??",
        "in_the_way(pixel_loc) == 0",
    ],
    7: [  # It seems good since it is on a sidewalk, far from the pole and the person, and is not in the way.
        "terrain(pixel_loc) == 0",
        "distance_to_pole(pixel_loc) > ??",
        "distance_to_person(pixel_loc) > ??",
        "in_the_way(pixel_loc) == 0",
    ],
    8: [  # This is also safe as it is on a sidewalk, and far from bushes, pole, and the barricade.
        "terrain(pixel_loc) == 0",
        "distance_to_bush(pixel_loc) > ??",
        "distance_to_pole(pixel_loc) > ??",
        "distance_to_barricade(pixel_loc) > ??",
    ],
    9: [  # Looks safe to me as on a sidewalk, and far from tree and the pole.
        "terrain(pixel_loc) == 0",
        "distance_to_tree(pixel_loc) > ??",
        "distance_to_pole(pixel_loc) > ??",
    ],
    10: [  # Surely this is safe as it is pretty far from the board, and is on a sidewalk.
        "terrain(pixel_loc) == 0",
        "distance_to_board(pixel_loc) > ??",
    ],
    11: [  # Not safe as it is too close to car, even though on a sidewalk.
        "terrain(pixel_loc) == 0",
        "distance_to_car(pixel_loc) > ??",
    ],
    12: [  # Even though it is on a sidewalk and is not in the way, it is unsafe since it is too close to the bushes.
        "terrain(pixel_loc) == 0",
        "distance_to_bush(pixel_loc) > ??",
        "in_the_way(pixel_loc) == 0",
    ],
    13: [  # Not safe because it is on speedway and also because it is too close to the board.
    ],
    14: [  # Unsafe as not far from the car, even if it is on a sidewalk.
        "terrain(pixel_loc) == 0",
        "distance_to_car(pixel_loc) > ??",
    ],
    15: [  # Unsafe as it is in the way, even though it is on a sidewalk.
        "terrain(pixel_loc) == 0",
        "in_the_way(pixel_loc) == 0",
    ],
    16: [  # It is safe because it is on a sidewalk, and is far from the bushes and not in front of the staircase.
        "terrain(pixel_loc) == 0",
        "distance_to_bush(pixel_loc) > ??",
        "frontal_distance_staircase(pixel_loc) > ??",
    ],
    17: [  # It is safe because it is on a sidewalk, and is far from the pole, the person, and not in the way.
        "terrain(pixel_loc) == 0",
        "distance_to_pole(pixel_loc) > ??",
        "distance_to_person(pixel_loc) > ??",
        "in_the_way(pixel_loc) == 0",
    ],
    18: [  # Not safe because too close to the person, even though on the sidewalk.
        "terrain(pixel_loc) == 0",
        "distance_to_person(pixel_loc) > ??",
    ],
    19: [  # It is on sidewalk, but not safe as in front of the staircase.
        "terrain(pixel_loc) == 0",
        "frontal_distance_staircase(pixel_loc) > ??",
    ],
    20: [  # It is not safe because it is in front of entrance, even though on a sidewalk.
        "terrain(pixel_loc) == 0",
        "frontal_distance_entrance(pixel_loc) > ??",
    ],
    21: [  # Unsafe as even though not in front of entrance and on tiles, it is not far from the wall.
        "terrain(pixel_loc) == 3",
        "distance_to_wall(pixel_loc) > ??",
        "frontal_distance_entrance(pixel_loc) > ??",
    ],
    22: [  # Safe as on sidewalk, and far from the bushes, and the car.
        "terrain(pixel_loc) == 0",
        "distance_to_bush(pixel_loc) > ??",
        "distance_to_car(pixel_loc) > ??",
    ],
    23: [  # Unsafe as even though on sidewalk, it is in front of the staircase.
        "terrain(pixel_loc) == 0",
        "frontal_distance_staircase(pixel_loc) > ??",
    ],
    24: [  # On sidewalk, but is unsafe as it is in the way. It is not on speedway nor on the bricks area.
        "terrain(pixel_loc) == 0",
        "in_the_way(pixel_loc) == 0",
    ],
    25: [  # On sidewalk, but in the way and thus unsafe.
        "terrain(pixel_loc) == 0",
        "in_the_way(pixel_loc) == 0",
    ],
    26: [  # It is not safe because it is in front of entrance, even though on a sidewalk.
        "terrain(pixel_loc) == 0",
        "frontal_distance_entrance(pixel_loc) > ??",
    ],
    27: [  # This is safe as it is on a sidewalk, is not in front of the entrance, and is just far enough from the bushes. It is also not in the way of the road and speedway merger.
        "terrain(pixel_loc) == 0",
        "frontal_distance_entrance(pixel_loc) > ??",
        "distance_to_bush(pixel_loc) > ??",
    ],
    28: [  # Its safe as it is not too inclined, and is on a sidewalk.
        "terrain(pixel_loc) == 0",
        "slope(pixel_loc) < ??",
    ],
    29: [  # It is in the way and thus unsafe, even though it is on a sidewalk.
        "terrain(pixel_loc) == 0",
        "in_the_way(pixel_loc) == 0",
    ],
}


def hack_seqn_filled_sketches(example_nums_feedlist, data_distilled):
    """
    Given the example num seqn feed order, returns the dict of seqn filled sketches
    """
    seqn_filled_lfps_sketches = {}
    cur_pseduolfps = {"terrain(pixel_loc) == 0": [], "terrain(pixel_loc) == 3": []}
    for i, example_num in enumerate(example_nums_feedlist):
        print(f"Processing {i+1}th of {len(example_nums_feedlist)} examples")
        cur_pseduolfps = hack_pseudoprogsketch_update_and_prune(cur_pseduolfps, example_num, data_distilled)
        lfps_sketch = convert_pseudolfps_to_lfps(cur_pseduolfps)
        nums_list = example_nums_feedlist[:i + 1]
        examples_data = read_data(nums_list)
        lsps_sketch = LDIPSdatagenSingleEx.convert_LFPS_to_LSPS(lfps_sketch)
        params = LDIPS_synthesize(examples_data, lsps_sketch)
        filled_lfps_sketch = LDIPSdatagenSingleEx.fillparams_in_LFPS(lfps_sketch, params)
        seqn_filled_lfps_sketches[str(i + 1)] = filled_lfps_sketch
    return seqn_filled_lfps_sketches


def hack_pseudoprogsketch_update_and_prune(cur_pseduolfps, new_data_num, data_distilled):
    """
    Given current pseudo_lfps sketch, update it with new data_num information, and prune it.
    cur_pseduolfps: dict of preds {"terrain(pixel_loc) == 0": [...], "terrain(pixel_loc) == 3": [...]}
    new_data_num: int
    """
    new_info_list = data_distilled[new_data_num]
    if "terrain(pixel_loc) == 0" in new_info_list:
        cur_set = set(cur_pseduolfps["terrain(pixel_loc) == 0"])
        new_set = set(new_info_list)
        new_set.remove("terrain(pixel_loc) == 0")
        cur_set.update(new_set)
        cur_pseduolfps["terrain(pixel_loc) == 0"] = list(cur_set)
        cur_pseduolfps["terrain(pixel_loc) == 0"].sort()
    elif "terrain(pixel_loc) == 3" in new_info_list:
        cur_set = set(cur_pseduolfps["terrain(pixel_loc) == 3"])
        new_set = set(new_info_list)
        new_set.remove("terrain(pixel_loc) == 3")
        cur_set.update(new_set)
        cur_pseduolfps["terrain(pixel_loc) == 3"] = list(cur_set)
        cur_pseduolfps["terrain(pixel_loc) == 3"].sort(reverse=True)
    return cur_pseduolfps


def convert_pseudolfps_to_lfps(cur_pseduolfps: dict):
    """
    Converts it into sketch string format.
    """
    base = "def is_safe(pixel_loc):\n"
    part0 = ""
    part3 = ""
    if len(cur_pseduolfps["terrain(pixel_loc) == 0"]) != 0:
        base += "    if terrain(pixel_loc) == 0:\n"
        part0 = "if "
        for pred in cur_pseduolfps["terrain(pixel_loc) == 0"]:
            part0 += f"{pred} and "
        part0 = part0[:-5] + ":"
        base += f"        {part0}\n"
        base += "            return True\n"
    if len(cur_pseduolfps["terrain(pixel_loc) == 3"]) != 0:
        base += "    elif terrain(pixel_loc) == 3:\n"
        part3 = "if "
        for pred in cur_pseduolfps["terrain(pixel_loc) == 3"]:
            part3 += f"{pred} and "
        part3 = part3[:-5] + ":"
        base += f"        {part3}\n"
        base += "            return True\n"
    base += "    return False\n"
    return base.strip()
