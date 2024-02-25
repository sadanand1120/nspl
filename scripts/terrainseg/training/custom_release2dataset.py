from __future__ import annotations

import json
import os
from string import Template
from typing import TYPE_CHECKING, Any, Dict, cast
from datasets import concatenate_datasets
import datasets
import segments
import requests
from PIL import Image
from segments.utils import load_image_from_url, load_label_bitmap_from_url
from segments.huggingface import get_taxonomy_table
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from segments.typing import Release


def my_release2dataset(release: Release, num_parts: int = -1, download_images: bool = True) -> datasets.Dataset:
    if num_parts <= 0:
        num_parts = release.samples_count // 16
        if num_parts < 1:
            num_parts = 1

    content = requests.get(cast(str, release.attributes.url))
    release_dict = json.loads(content.content)
    task_type = release_dict["dataset"]["task_type"]

    if task_type in ["vector", "bboxes", "keypoint"]:
        features = datasets.Features(
            {
                "name": datasets.Value("string"),
                "uuid": datasets.Value("string"),
                "image": {"url": datasets.Value("string")},
                "status": datasets.Value("string"),
                "label": {
                    "annotations": [
                        {
                            "id": datasets.Value("int32"),
                            "category_id": datasets.Value("int32"),
                            "type": datasets.Value("string"),
                            "points": [[datasets.Value("float32")]],
                        }
                    ],
                },
            }
        )
    elif task_type in ["segmentation-bitmap", "segmentation-bitmap-highres"]:
        features = datasets.Features(
            {
                "name": datasets.Value("string"),
                "uuid": datasets.Value("string"),
                "image": {"url": datasets.Value("string")},
                "status": datasets.Value("string"),
                "label": {
                    "annotations": [
                        {
                            "id": datasets.Value("int32"),
                            "category_id": datasets.Value("int32"),
                        }
                    ],
                    "segmentation_bitmap": {"url": datasets.Value("string")},
                },
            }
        )
    elif task_type in ["text-named-entities", "text-span-categorization"]:
        features = datasets.Features(
            {
                "name": datasets.Value("string"),
                "uuid": datasets.Value("string"),
                "text": datasets.Value("string"),
                "status": datasets.Value("string"),
                "label": {
                    "annotations": [
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                            "category_id": datasets.Value("int32"),
                        }
                    ],
                },
            }
        )
    else:
        raise ValueError("This type of dataset is not yet supported.")

    samples = release_dict["dataset"]["samples"]

    data_rows = []
    for sample in samples:
        try:
            del sample["labels"]["ground-truth"]["attributes"]["format_version"]
        except (KeyError, TypeError):
            pass
        data_row: Dict[str, Any] = {}
        data_row["name"] = sample["name"]
        data_row["uuid"] = sample["uuid"]
        try:
            data_row["status"] = sample["labels"]["ground-truth"]["label_status"]
        except (KeyError, TypeError):
            data_row["status"] = "UNLABELED"

        # Image or text
        if task_type in [
            "vector",
            "bboxes",
            "keypoint",
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            try:
                data_row["image"] = sample["attributes"]["image"]
            except (KeyError, TypeError):
                data_row["image"] = {"url": None}
        elif task_type in ["text-named-entities", "text-span-categorization"]:
            try:
                data_row["text"] = sample["attributes"]["text"]
            except (KeyError, TypeError):
                data_row["text"] = None

        # Label
        try:
            label = sample["labels"]["ground-truth"]["attributes"]

            # Remove the image-level attributes
            if "attributes" in label:
                del label["attributes"]

            # Remove the object-level attributes
            for annotation in label["annotations"]:
                if "attributes" in annotation:
                    del annotation["attributes"]

            data_row["label"] = label

        except (KeyError, TypeError):
            error_label: Dict[str, Any] = {"annotations": []}
            if task_type in ["segmentation-bitmap", "segmentation-bitmap-highres"]:
                error_label["segmentation_bitmap"] = {"url": None}
            data_row["label"] = error_label

        data_rows.append(data_row)

    all_ds = []

    def split_list(data, n_parts):
        split_len = len(data) // n_parts
        splits = [data[i * split_len:(i + 1) * split_len] for i in range(n_parts)]
        splits[-1] += data[n_parts * split_len:]  # add the remaining elements to the last part
        return splits

    list_of_data_rows = split_list(data_rows, num_parts)

    for drows in list_of_data_rows:
        # Now transform to column format
        dataset_dict: Dict[str, Any] = {key: [] for key in features.keys()}
        for data_row in drows:
            for key in dataset_dict.keys():
                dataset_dict[key].append(data_row[key])

        # Create the HF Dataset and flatten it
        dataset = datasets.Dataset.from_dict(dataset_dict, features, split="train")
        dataset = dataset.flatten()
        # Optionally download the images
        if (
            task_type
            in [
                "vector",
                "bboxes",
                "keypoint",
                "segmentation-bitmap",
                "segmentation-bitmap-highres",
            ]
            and download_images
        ):

            def download_image(data_row: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    data_row["image"] = load_image_from_url(data_row["image.url"])
                except Exception:
                    data_row["image"] = None
                return data_row

            def download_segmentation_bitmap(data_row: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    segmentation_bitmap = load_label_bitmap_from_url(
                        data_row["label.segmentation_bitmap.url"]
                    )
                    data_row["label.segmentation_bitmap"] = Image.fromarray(
                        segmentation_bitmap
                    )
                except Exception:
                    data_row["label.segmentation_bitmap"] = Image.new(
                        "RGB", (1, 1)
                    )  # TODO: replace with None
                return data_row

            dataset = dataset.map(download_image, remove_columns=["image.url"])
            if task_type in ["segmentation-bitmap", "segmentation-bitmap-highres"]:
                dataset = dataset.map(
                    download_segmentation_bitmap,
                    remove_columns=["label.segmentation_bitmap.url"],
                )
                # Reorder the features
                dataset.info.features = datasets.Features(
                    {
                        "name": dataset.features["name"],
                        "uuid": dataset.features["uuid"],
                        "status": dataset.features["status"],
                        "image": datasets.Image(),
                        "label.annotations": dataset.features["label.annotations"],
                        "label.segmentation_bitmap": datasets.Image(),
                    }
                )
            else:
                # Reorder the features
                dataset.info.features = datasets.Features(
                    {
                        "name": dataset.features["name"],
                        "uuid": dataset.features["uuid"],
                        "status": dataset.features["status"],
                        "image": datasets.Image(),
                        "label.annotations": dataset.features["label.annotations"],
                    }
                )
        all_ds.append(dataset)

    full_dataset = concatenate_datasets(all_ds)

    # Create id2label
    id2label = {}
    for category in release_dict["dataset"]["task_attributes"]["categories"]:
        id2label[category["id"]] = category["name"]
    id2label[0] = "unlabeled"
    full_dataset.id2label = id2label

    # Create readme.md and update DatasetInfo
    # https://stackoverflow.com/questions/6385686/is-there-a-native-templating-system-for-plain-text-files-in-python

    task_type = release_dict["dataset"]["task_type"]
    if task_type in ["segmentation-bitmap", "segmentation-bitmap-highres"]:
        task_category = "image-segmentation"
    elif task_type in ["vector", "bboxes"]:
        task_category = "object-detection"
    elif task_type in ["text-named-entities", "text-span-categorization"]:
        task_category = "named-entity-recognition"
    else:
        task_category = "other"

    info = {
        "name": release_dict["dataset"]["name"],
        "segments_url": f'https://segments.ai/{release_dict["dataset"]["owner"]}/{release_dict["dataset"]["name"]}',
        "short_description": release_dict["dataset"]["description"],
        "release": release_dict["name"],
        "taxonomy_table": get_taxonomy_table(
            release_dict["dataset"]["task_attributes"]
        ),
        "task_category": task_category,
    }

    # Create readme.md
    segments_dir = os.path.dirname(segments.__file__)
    with open(
        os.path.join(segments_dir, "data", "dataset_card_template.md"), "r"
    ) as f:
        template = Template(f.read())
        readme = template.substitute(info)
        full_dataset.readme = readme

    # Update DatasetInfo
    full_dataset.info.description = info["short_description"]
    full_dataset.info.homepage = info["segments_url"]

    return full_dataset
