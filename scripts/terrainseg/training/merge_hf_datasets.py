from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict
import os
from dropoff.dutils import remove_username

HF_USERNAME = "sam1120"
HF_DATASETS_NAMES = [
    "sam1120/safety-utcustom-train-v1.0",
    "sam1120/safety-utcustom-test-v1.0",
]
MERGED_DATASET_NAME = "merged-safety-utcustom-train-test-v1.0"

MERGED_DATASET_NAME = remove_username(MERGED_DATASET_NAME)
HF_DATASETS_NAMES = [remove_username(hf_dataset_name) for hf_dataset_name in HF_DATASETS_NAMES]

print(f"Merging datasets: {HF_DATASETS_NAMES}")
print(f"MERGED_DATASET_NAME = {MERGED_DATASET_NAME}")

hf_key = os.environ.get("HF_API_KEY")
dss = []
for hf_dataset_name in HF_DATASETS_NAMES:
    hf_dataset_identifier = f"{HF_USERNAME}/{hf_dataset_name}"
    dss.append(load_dataset(hf_dataset_identifier))

merged_dataset = DatasetDict({split: concatenate_datasets([ds[split] for ds in dss]) for split in dss[0].keys()})
merged_dataset_identifier = f"{HF_USERNAME}/{MERGED_DATASET_NAME}"
merged_dataset.push_to_hub(merged_dataset_identifier, token=hf_key)

# TODO: ADD manually id2label.json to the dataset repo on huggingface
