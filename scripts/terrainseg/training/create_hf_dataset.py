from segments import SegmentsClient
from terrainseg.training.custom_release2dataset import my_release2dataset
from segments.utils import get_semantic_bitmap
import argparse
import os
from dropoff.dutils import remove_username

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--segments_dataset_name',
                    type=str,
                    default="safety-utcustom-train",
                    help='Name of the segments-ai dataset')

parser.add_argument('-r', '--segments_release_name',
                    type=str,
                    default="v1.0",
                    help='Name of the segments-ai release')

parser.add_argument('-n', '--hf_dataset_name',
                    type=str,
                    default=None,
                    help='Name of the hf dataset being generated')

# parse the arguments
args = parser.parse_args()

segments_dataset_name = remove_username(args.segments_dataset_name)
segments_release_name = args.segments_release_name
segments_username = "smodak"
hf_username = "sam1120"

segments_key = os.environ.get("SEGMENTSAI_API_KEY")
segments_dataset_identifier = f"{segments_username}/{segments_dataset_name}"
segments_client = SegmentsClient(segments_key)

hf_key = os.environ.get("HF_API_KEY")
if args.hf_dataset_name is not None:
    hf_dataset_name = remove_username(args.hf_dataset_name)
else:
    hf_dataset_name = f"{segments_dataset_name}-{segments_release_name}"
hf_dataset_identifier = f"{hf_username}/{hf_dataset_name}"

release = segments_client.get_release(segments_dataset_identifier, segments_release_name)
hf_dataset = my_release2dataset(release)


def convert_segmentation_bitmap_batch(batch):
    output = []
    for idx in range(len(batch["label.segmentation_bitmap"])):
        output.append(get_semantic_bitmap(
            batch["label.segmentation_bitmap"][idx],
            batch["label.annotations"][idx]
        ))
    return {
        "label.segmentation_bitmap": output
    }


semantic_dataset = hf_dataset.map(convert_segmentation_bitmap_batch, batched=True, batch_size=32)
semantic_dataset = semantic_dataset.rename_column('image', 'pixel_values')
semantic_dataset = semantic_dataset.rename_column('label.segmentation_bitmap', 'labels')
semantic_dataset = semantic_dataset.remove_columns(['uuid', 'status', 'label.annotations'])  # has 'pixel_values', 'labels', 'name' columns

print("LOGINFO: Pushing dataset to HuggingFace...")
semantic_dataset.push_to_hub(hf_dataset_identifier, token=hf_key)

# TODO: ADD manually id2label.json to the dataset repo on huggingface
