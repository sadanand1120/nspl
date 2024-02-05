import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
from datasets import load_dataset, DatasetDict
import numpy as np
from PIL import Image


def calculate_counts_percentage(arr):
    values, counts = np.unique(arr, return_counts=True)
    total = arr.size
    counts_dict = dict(zip(values, counts))
    percentages = {val: round((count / total) * 100, 4) for val, count in counts_dict.items()}
    return counts_dict, percentages


def find_hf_dataset_labels_px_percentages(hfdi, num_labels=3):
    ds_dict = load_dataset(hfdi)

    # intialize
    tot_percentages = {i: 0 for i in range(num_labels)}
    tot_data = 0

    # calculate percentages
    for split in ds_dict.keys():
        print(f"Processing split {split}...")
        ds = ds_dict[split]
        for i in range(len(ds)):
            print(f"Processing idx {i+1}/{len(ds)} of split {split}...")
            gt_seg_np = np.array(ds[i]['labels'])  # (H, W)
            _, percentages = calculate_counts_percentage(gt_seg_np)
            for label, percentage in percentages.items():
                tot_percentages[label] += percentage
            tot_data += 1

    # average percentages
    for label, percentage in tot_percentages.items():
        tot_percentages[label] = round(percentage / tot_data, 4)

    return tot_percentages


def remove_username(dataset_name):
    return dataset_name.split("/")[-1]


def correct_ds(batch, unlabeled_idx, correct_idx):
    output = []
    for idx in range(len(batch["labels"])):
        gt_seg_np = np.array(batch["labels"][idx])  # (H, W)
        unlabeled_mask = (gt_seg_np == unlabeled_idx)
        if np.any(unlabeled_mask):
            gt_seg_np[unlabeled_mask] = correct_idx
        output.append(Image.fromarray(gt_seg_np.astype('int32'), mode='I'))
    return {
        "labels": output
    }


def check_hf_ds_for_unlabeled(hfdi, unlabeled_idx=0, correct_idx=2, reupload=False):
    ds_dict = load_dataset(hfdi)
    splits = ds_dict.keys()
    new_ds_dict = DatasetDict()
    FOUND = False
    for split in splits:
        print(f"Checking split {split}...")
        counter = 0
        ds = ds_dict[split]
        print(f"len(ds) = {len(ds)} for split {split}")
        for i in range(len(ds)):
            print(f"Processing idx {i} of split {split}...")
            # pil_img_np = np.array(ds[i]['pixel_values'])  # (H, W, C)
            gt_seg_np = np.array(ds[i]['labels'])  # (H, W)
            unlabeled_mask = (gt_seg_np == unlabeled_idx)
            if np.any(unlabeled_mask):
                print(f">>>>>>>>>> Found unlabeled in idx {i} - image name {ds[i]['name']} - split {split} <<<<<<<<<<")
                FOUND = True
                counts_dict, percentages = calculate_counts_percentage(gt_seg_np)
                print(f"Counts: {counts_dict}")
                print(f"Percentages: {percentages}")
                counter += 1
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"********************** Found {counter} images with unlabeled in split {split} **********************")
        print("------------------------------------------------------------------------------------------------------------------")
        if reupload:
            new_ds = ds.map(correct_ds, batched=True, batch_size=16, fn_kwargs={"unlabeled_idx": unlabeled_idx, "correct_idx": correct_idx})
            new_ds_dict[split] = new_ds
    if reupload and FOUND:
        print("LOGINFO: Reuploading corrected dataset to HuggingFace...")
        new_ds_dict.push_to_hub(hfdi, token=os.environ.get("HF_API_KEY"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--hfdi", type=str, required=True)
    args.add_argument("--unlabeled_idx", type=int, default=0)
    args.add_argument("--correct_idx", type=int, default=2)
    args.add_argument("--reupload", action="store_true")
    args = args.parse_args()
    check_hf_ds_for_unlabeled(args.hfdi, args.unlabeled_idx, args.correct_idx, args.reupload)

    # hfdi = "sam1120/safety-utcustom-EVAL"
    # pprint(find_hf_dataset_labels_px_percentages(hfdi, num_labels=3))
