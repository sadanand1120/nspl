import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
import evaluate
import numpy as np
from terrainseg.inference import TerrainSegFormer
import cv2
from pprint import pprint

EXCLUDE_IDX = {
    "train/utcustom": [11, 17, 34, 38, 46, 14, 13, 1, 26, 21, 9, 16, 20, 37, 47, 28, 48],
    "test": [38, 63, 69, 11, 58, 97, 46, 77, 37, 31, 9, 53, 94, 60, 84, 43, 56, 87, 26, 91, 36, 61, 2, 27, 7, 4, 28, 64, 86, 13, 62],
    "eval": [5, 8, 9, 10, 11, 19, 22, 32, 35, 36, 48, 49],
}


def iou_viz_individual(root_dir, method_num, H=540, W=960):
    exclude_idx_dict = {}
    id2label = {
        0: "unlabeled",
        1: "dropoff",
        2: "undropoff",
    }
    gt_root_dir = os.path.join(root_dir, "gt_preds")
    pred_root_dir = os.path.join(root_dir, f"methods_preds/{method_num}")
    all_gt_bins_paths = [os.path.join(gt_root_dir, f) for f in sorted(os.listdir(gt_root_dir)) if f.endswith(".bin")]
    all_pred_bins_paths = [os.path.join(pred_root_dir, f) for f in sorted(os.listdir(pred_root_dir)) if f.endswith(".bin")]
    all_gt_bins = []
    all_pred_bins = []
    for gt_bin_path in all_gt_bins_paths:
        pc_np = np.fromfile(gt_bin_path, dtype=np.uint8).reshape((H, W))
        all_gt_bins.append(pc_np)

    for pred_bin_path in all_pred_bins_paths:
        pc_np = np.fromfile(pred_bin_path, dtype=np.uint8).reshape((H, W))
        all_pred_bins.append(pc_np)

    _metric = evaluate.load("mean_iou")
    for i in range(len(all_gt_bins)):
        print(f"Processing idx {i}/{len(all_gt_bins)}")
        gt_bin = all_gt_bins[i].reshape((1, H, W))
        pred_bin = all_pred_bins[i].reshape((1, H, W))
        metrics = _metric._compute(
            predictions=pred_bin,
            references=gt_bin,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=False,
        )
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        iou_dict = {}
        iou_dict["mIOU"] = metrics["mean_iou"]
        iou_dict["IOU_dropoff"] = metrics["iou_dropoff"]
        iou_dict["IOU_undropoff"] = metrics["iou_undropoff"]
        pprint(iou_dict)
        print("-------------------------------------------------------------------")
        if iou_dict["mIOU"] < 0.8:
            exclude_idx_dict[i] = iou_dict
            noext_name = os.path.splitext(os.path.basename(all_gt_bins_paths[i]))[0]
            cv2_img = cv2.imread(os.path.join(root_dir, "images", f"{noext_name}.png"))
            gt_overlay = TerrainSegFormer.get_seg_overlay(cv2_img, gt_bin[0], alpha=0.24)
            pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_img, pred_bin[0], alpha=0.24)
            side_by_side = np.concatenate([gt_overlay, pred_overlay], axis=1)
            side_by_side = cv2.resize(side_by_side, None, fx=0.75, fy=0.75)
            cv2.imshow(f"{noext_name}.png", side_by_side)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # mydir = "/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/evals_data_safety/utcustom/ns_viz_dir/train"
            # cv2.imwrite(os.path.join(mydir, f"{noext_name}.png"), side_by_side)
    pprint(exclude_idx_dict)
    sorted_exclude_idx_list = sorted(exclude_idx_dict, key=lambda x: exclude_idx_dict[x]["IOU_dropoff"])
    print(sorted_exclude_idx_list)


def compute_iou(root_dir, root_dirnames, method_num, H=540, W=960, do_exclude=True):
    """
    Computes the IoU between the ground truth and the predictions using the bins
    """
    if not isinstance(root_dirnames, list):
        root_dirnames = [root_dirnames]
    id2label = {
        0: "unlabeled",
        1: "dropoff",
        2: "undropoff",
    }
    all_gt_bins = []
    all_pred_bins = []
    for root_dirname in root_dirnames:
        root_dir2 = os.path.join(root_dir, root_dirname)
        if do_exclude:
            EXCLUDE_IDX_LIST = EXCLUDE_IDX[root_dirname]
        else:
            EXCLUDE_IDX_LIST = []
        gt_root_dir = os.path.join(root_dir2, "gt_preds")
        pred_root_dir = os.path.join(root_dir2, f"methods_preds/{method_num}")
        all_gt_bins_paths = [os.path.join(gt_root_dir, f) for f in sorted(os.listdir(gt_root_dir)) if f.endswith(".bin")]
        all_pred_bins_paths = [os.path.join(pred_root_dir, f) for f in sorted(os.listdir(pred_root_dir)) if f.endswith(".bin")]
        for i, gt_bin_path in enumerate(all_gt_bins_paths):
            if i in EXCLUDE_IDX_LIST:
                continue
            pc_np = np.fromfile(gt_bin_path, dtype=np.uint8).reshape((H, W))
            all_gt_bins.append(pc_np)
        for i, pred_bin_path in enumerate(all_pred_bins_paths):
            if i in EXCLUDE_IDX_LIST:
                continue
            pc_np = np.fromfile(pred_bin_path, dtype=np.uint8).reshape((H, W))
            all_pred_bins.append(pc_np)
    full_gt_bin = np.concatenate(all_gt_bins, axis=0).reshape((-1, H, W))
    full_pred_bin = np.concatenate(all_pred_bins, axis=0).reshape((-1, H, W))
    print("Evaluating...")
    _metric = evaluate.load("mean_iou")
    metrics = _metric._compute(
        predictions=full_pred_bin,
        references=full_gt_bin,
        num_labels=len(id2label),
        ignore_index=0,
        reduce_labels=False,
    )
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()
    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    iou_dict = {}
    iou_dict["mIOU"] = round(metrics["mean_iou"] * 100, 2)
    iou_dict["IOU_dropoff"] = round(metrics["iou_dropoff"] * 100, 2)
    iou_dict["IOU_undropoff"] = round(metrics["iou_undropoff"] * 100, 2)
    return iou_dict


if __name__ == "__main__":
    _H = 540
    _W = 960
    root_dir = os.path.join(nspl_root_dir, "evals_data_dropoff/utcustom")

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirname", type=str, default="train")  # either eval/train/test, could be many separated by commas
    parser.add_argument("--method_num", type=int, default=1)
    parser.add_argument("--do_exclude", action="store_true")
    args = parser.parse_args()
    root_dirnames = args.root_dirname.split(",")
    iou_dict = compute_iou(root_dir, root_dirnames, args.method_num, H=_H, W=_W, do_exclude=args.do_exclude)
    print(f"Evaluation results for method {args.method_num}:")
    pprint(iou_dict)
