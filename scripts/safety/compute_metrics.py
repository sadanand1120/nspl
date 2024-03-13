import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
import evaluate
import numpy as np
from terrainseg.inference import TerrainSegFormer
import cv2
from pprint import pprint
from simple_colors import red, green

EXCLUDE_IDX = {
    "train/utcustom": [11, 17, 34, 38, 46, 14, 13, 1, 26, 21, 9, 16, 20, 37, 47, 28, 48],
    "test": [38, 63, 69, 11, 58, 97, 46, 77, 37, 31, 9, 53, 94, 60, 84, 43, 56, 87, 26, 91, 36, 61, 2, 27, 7, 4, 28, 64, 86, 13, 62],
    "eval": [5, 8, 9, 10, 11, 19, 22, 32, 35, 36, 48, 49],
}


def downsample(arr, grid_size=20, threshold=0.80):
    # Dimensions of the new grid
    new_height, new_width = grid_size, grid_size
    # Size of blocks
    block_height, block_width = arr.shape[0] // new_height, arr.shape[1] // new_width
    # Reduced array
    reduced = np.zeros((new_height, new_width), dtype=int)
    for i in range(new_height):
        for j in range(new_width):
            # Extracting each block
            block = arr[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            # Counting 1s and 0s
            ones = np.sum(block)
            # Determining the value for reduced array cell, 80% threshold
            reduced[i, j] = 1 if ones > (block_height * block_width) * threshold else 0
    return reduced


def iou_viz_individual(root_dir, method_num, H=540, W=960):
    exclude_idx_dict = {}
    id2label = {
        0: "unlabeled",
        1: "safe",
        2: "unsafe",
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
    # full_gt_bin = np.concatenate(all_gt_bins, axis=0).reshape((-1, H, W))

    for pred_bin_path in all_pred_bins_paths:
        pc_np = np.fromfile(pred_bin_path, dtype=np.uint8).reshape((H, W))
        all_pred_bins.append(pc_np)
    # full_pred_bin = np.concatenate(all_pred_bins, axis=0).reshape((-1, H, W))

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
        iou_dict["IOU_safe"] = metrics["iou_safe"]
        iou_dict["IOU_unsafe"] = metrics["iou_unsafe"]
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
    sorted_exclude_idx_list = sorted(exclude_idx_dict, key=lambda x: exclude_idx_dict[x]["IOU_safe"])
    print(sorted_exclude_idx_list)


def compute_iou_downsampled(root_dir, root_dirnames, method_num, H=540, W=960, grid_size=20, do_exclude=True, downsample_threshold=0.8):
    """
    Computes the IoU between the ground truth and the predictions using the bins (downsampled)
    """
    if not isinstance(root_dirnames, list):
        root_dirnames = [root_dirnames]
    id2label = {
        0: "unlabeled",
        1: "safe",
        2: "unsafe",
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
            pc_np[pc_np == 2] = 0
            pc_np = downsample(pc_np, threshold=downsample_threshold, grid_size=grid_size)
            pc_np[pc_np == 0] = 2
            all_gt_bins.append(pc_np)
        for i, pred_bin_path in enumerate(all_pred_bins_paths):
            if i in EXCLUDE_IDX_LIST:
                continue
            pc_np = np.fromfile(pred_bin_path, dtype=np.uint8).reshape((H, W))
            pc_np[pc_np == 2] = 0
            pc_np = downsample(pc_np, threshold=downsample_threshold, grid_size=grid_size)
            pc_np[pc_np == 0] = 2
            all_pred_bins.append(pc_np)
    full_gt_bin = np.concatenate(all_gt_bins, axis=0).reshape((-1, grid_size, grid_size))
    full_pred_bin = np.concatenate(all_pred_bins, axis=0).reshape((-1, grid_size, grid_size))
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
    # iou_dict["mIOU"] = round(metrics["mean_iou"] * 100, 2)
    iou_dict["IOU_safe"] = round(metrics["iou_safe"] * 100, 2)
    iou_dict["IOU_unsafe"] = round(metrics["iou_unsafe"] * 100, 2)
    return iou_dict


def compute_iou(root_dir, root_dirnames, method_num, H=540, W=960, do_exclude=True):
    """
    Computes the IoU between the ground truth and the predictions using the bins
    """
    if not isinstance(root_dirnames, list):
        root_dirnames = [root_dirnames]
    id2label = {
        0: "unlabeled",
        1: "safe",
        2: "unsafe",
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
    # iou_dict["mIOU"] = round(metrics["mean_iou"] * 100, 2)
    iou_dict["IOU_safe"] = round(metrics["iou_safe"] * 100, 2)
    iou_dict["IOU_unsafe"] = round(metrics["iou_unsafe"] * 100, 2)
    return iou_dict


if __name__ == "__main__":
    _H = 540
    _W = 960

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirname", type=str, default="train/utcustom")  # either eval or train+test, could be many separated by commas
    parser.add_argument("--concept", type=str, default="safety")
    parser.add_argument("--method_num", type=int, default=1)
    parser.add_argument('-d', "--downsample_threshold", type=float, default=None)
    parser.add_argument("--do_exclude", action="store_true")
    args = parser.parse_args()
    root_dir = os.path.join(nspl_root_dir, f"evals_data_{args.concept}/utcustom")
    root_dirnames = args.root_dirname.split(",")
    if args.downsample_threshold is None:
        iou_dict = compute_iou(root_dir, root_dirnames, args.method_num, H=_H, W=_W, do_exclude=args.do_exclude)
    else:
        iou_dict = compute_iou_downsampled(root_dir, root_dirnames, args.method_num, H=_H, W=_W, grid_size=20, do_exclude=args.do_exclude, downsample_threshold=args.downsample_threshold)
    print(green(f"Evaluation results for method {args.method_num}:", "bold"))
    pprint(iou_dict)

    # eval_root_dir = os.path.join(root_dir, "test")
    # iou_viz_individual(eval_root_dir, 1, H=_H, W=_W)

    # gt_bin = np.fromfile("/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/evals_data_safety/utcustom/test/gt_preds/000002_morning_mode1_11062023_000464.bin", dtype=np.uint8).reshape((540, 960))
    # pred_bin = np.fromfile("/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/evals_data_safety/utcustom/test/methods_preds/1/000002_morning_mode1_11062023_000464.bin", dtype=np.uint8).reshape((540, 960))
    # cv2_img = cv2.imread("/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/evals_data_safety/utcustom/test/images/000002_morning_mode1_11062023_000464.png")
    # gt_overlay = TerrainSegFormer.get_seg_overlay(cv2_img, gt_bin, alpha=0.24)
    # pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_img, pred_bin, alpha=0.24)
    # side_by_side = np.concatenate([gt_overlay, pred_overlay], axis=1)
    # cv2.imshow("side_by_side", side_by_side)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
