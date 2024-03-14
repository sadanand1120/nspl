import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
from safety.hack_cnf_form import DATA_DISTILLED, hack_seqn_filled_sketches
from safety.gen_preds import gt_save
from utilities.std_utils import json_reader
from safety.faster_ns_inference import FasterImageInference
from safety.compute_metrics import compute_iou
from datasets import load_dataset
from simple_colors import red, green, yellow
from safety.gen_preds import bind_method
import argparse
import cv2
from pprint import pprint


def eval_seqn_main(example_nums_feedlist, data_distilled, eval_subdirname, eval_di):
    """
    Returns seq iou evals for the given eval_subdirname.
    """
    def custom_ns_save_pred(hfdi, root_dir, ldips_infer_ns_obj: FasterImageInference, is_safe, method_num=1, start=0, step_size=1):
        pred_root_dir = os.path.join(root_dir, f"methods_preds/{method_num}")
        fi_data_dir = os.path.join(root_dir, "fi_data")
        os.makedirs(pred_root_dir, exist_ok=True)
        os.makedirs(fi_data_dir, exist_ok=True)
        ds = load_dataset(hfdi)['train']
        for i in range(start, len(ds), step_size):
            img_name = ds[i]['name']
            noext_name = os.path.splitext(img_name)[0]
            pc_name = f"{noext_name}.bin"
            img_path = os.path.join(root_dir, "images", img_name)
            pc_path = os.path.join(root_dir, "pcs", pc_name)
            cv2_img = cv2.imread(img_path)
            pc_xyz = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))[:, :3]
            ldips_infer_ns_obj.set_state(fi_data_dir=fi_data_dir,
                                         noext_name=noext_name,
                                         img_bgr=cv2_img,
                                         pc_xyz=pc_xyz)
            is_safe_mask = np.zeros((cv2_img.shape[0], cv2_img.shape[1]), dtype=np.uint8)
            for k in range(cv2_img.shape[0]):
                for j in range(cv2_img.shape[1]):
                    is_safe_mask[k, j] = is_safe((j, k))

            is_safe_mask[is_safe_mask == 0] = 2
            flat_is_safe_mask = is_safe_mask.reshape(-1).astype(np.uint8)
            flat_is_safe_mask.tofile(os.path.join(pred_root_dir, f"{noext_name}.bin"))

    print(green("Generating filled lfps sketches", 'bold'))
    seqn_filled_lfps_sketches = hack_seqn_filled_sketches(example_nums_feedlist, data_distilled)
    root_dir = os.path.join(nspl_root_dir, "evals_data_safety/utcustom")
    eval_root_dir = os.path.join(root_dir, eval_subdirname)
    if not os.path.exists(os.path.join(eval_root_dir, "gt_preds")):
        print(f"Saving ground truth labels for {eval_subdirname}...")
        gt_save(hfdi=eval_di,
                root_dir=eval_root_dir)

    # ns setup
    hitl_llm_state = json_reader(os.path.join(nspl_root_dir, "scripts/llm/state.json"))
    DOMAIN = hitl_llm_state["domain"]
    fi = FasterImageInference(DOMAIN)
    globals()['terrain'] = fi._terrain
    globals()['in_the_way'] = fi._in_the_way
    globals()['slope'] = fi._slope
    for method_name in ['distance_to_' + obj for obj in DOMAIN["objects"]]:
        globals()[method_name] = bind_method(fi, f"_{method_name}")
    for method_name in ['frontal_distance_' + obj for obj in DOMAIN["objects"]]:
        globals()[method_name] = bind_method(fi, f"_{method_name}")

    iou_res = np.zeros((len(example_nums_feedlist), 3))  # cols: iou_safe, iou_unsafe, miou
    print(green("Evaluating filled lfps sketches", 'bold'))
    for stri1, filled_lfps_sketch in seqn_filled_lfps_sketches.items():
        print(yellow(f"Processing {stri1}th of {len(example_nums_feedlist)} examples", 'bold'))
        exec(filled_lfps_sketch, globals())
        custom_ns_save_pred(hfdi=eval_di,
                            root_dir=eval_root_dir,
                            ldips_infer_ns_obj=fi,
                            is_safe=is_safe)
        iou_dict = compute_iou(root_dir=root_dir,
                               root_dirnames=eval_subdirname,
                               method_num=1,
                               H=540,
                               W=960,
                               do_exclude=False)
        iou_res[int(stri1) - 1, 0] = iou_dict["IOU_safe"]
        iou_res[int(stri1) - 1, 1] = iou_dict["IOU_unsafe"]
        iou_res[int(stri1) - 1, 2] = iou_dict["mIOU"]
        if stri1 == "29":
            print(yellow(f"For this sequence ==> IOU safe: {iou_dict['IOU_safe']}, IOU unsafe: {iou_dict['IOU_unsafe']}, mIOU: {iou_dict['mIOU']}", 'bold'))
    return iou_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npermuts", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--root_dirname", type=str, default="eval")
    parser.add_argument("--eval_di", type=str, default="sam1120/safety-utcustom-EVAL")
    args = parser.parse_args()

    N_PERMUTS = args.npermuts
    dest_dir = os.path.join(nspl_root_dir, "scripts/safety/lifelong_seqn_permuts", args.root_dirname)
    os.makedirs(dest_dir, exist_ok=True)
    nums_list = np.arange(1, 30)
    for i in range(args.start, args.start + N_PERMUTS):
        print(green(f"******************************************************************** Processing {i+1-args.start}/{N_PERMUTS} ***************************************************************", 'bold'))
        print(green(f"Processing permuted example nums {[nums_list]}", 'bold'))
        try:
            iou_res = eval_seqn_main(example_nums_feedlist=nums_list,
                                     data_distilled=DATA_DISTILLED,
                                     eval_subdirname=args.root_dirname,
                                     eval_di=args.eval_di)
            flat_iou_res = iou_res.reshape(-1).astype(np.float32)
            flat_iou_res.tofile(os.path.join(dest_dir, f"{i:004}.bin"))
        except:
            print(red(f"Failed to process {i+1}/{N_PERMUTS}", 'bold'))
        np.random.shuffle(nums_list)
