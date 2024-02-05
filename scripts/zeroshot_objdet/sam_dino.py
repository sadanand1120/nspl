import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_hq_model_registry, SamPredictor


class GroundedSAM:
    def __init__(self, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8, ann_thickness=2, ann_text_scale=0.3, ann_text_thickness=1, ann_text_padding=5):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GROUNDING_DINO_CONFIG_PATH = os.path.join(nspl_root_dir, "third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(nspl_root_dir, "third_party/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth")
        self.SAM_ENCODER_VERSION = "vit_h"
        self.SAM_CHECKPOINT_PATH = os.path.join(nspl_root_dir, "third_party/Grounded-Segment-Anything/weights/sam_hq_vit_h.pth")
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
        self.sam = sam_hq_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam)
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.NMS_THRESHOLD = nms_threshold
        self.box_annotator = sv.BoxAnnotator(
            thickness=ann_thickness,
            text_scale=ann_text_scale,
            text_thickness=ann_text_thickness,
            text_padding=ann_text_padding
        )
        self.mask_annotator = sv.MaskAnnotator()

    def predict_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO on image.
        img: A x B x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels
            detections:
                - xyxy: (N, 4) boxes (float pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
        """
        if box_threshold is None:
            box_threshold = self.BOX_THRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_THRESHOLD
        if nms_threshold is None:
            nms_threshold = self.NMS_THRESHOLD
        detections = self.grounding_dino_model.predict_with_classes(
            image=img,
            classes=text_prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # print(f"Detected {len(detections.xyxy)} boxes")
        if do_nms:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            # print(f"After NMS: {len(detections.xyxy)} boxes")
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_frame = self.box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
        return annotated_frame, detections

    @staticmethod
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    @staticmethod
    def get_per_class_mask(img, masks, class_ids, num_classes):
        """
        Create a per-class segmentation mask.
        Parameters:
            masks: N x H x W array, where N is the number of masks
            class_ids: (N,) array of corresponding class ids
            num_classes: Total number of classes, C
        Returns:
            per_class_mask: C x H x W array
        """
        H, W = img.shape[0], img.shape[1]
        per_class_mask = np.zeros((num_classes, H, W), dtype=bool)
        if len(masks) == 0:
            return per_class_mask
        for i in range(num_classes):
            class_idx = np.where(class_ids == i)[0]
            if class_idx.size > 0:
                per_class_mask[i] = np.any(masks[class_idx], axis=0)
        return per_class_mask

    @staticmethod
    def HACK_filter_keep0(detections):
        mask0 = detections.class_id == 0
        detections.xyxy = detections.xyxy[mask0].reshape((-1, 4))
        detections.confidence = detections.confidence[mask0].reshape((-1, ))
        detections.class_id = detections.class_id[mask0].reshape((-1, ))
        if detections.mask is not None:
            H = detections.mask.shape[1]
            W = detections.mask.shape[2]
            detections.mask = detections.mask[mask0].reshape((-1, H, W))
        return detections

    def predict_and_segment_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO and segmentation using HQ-SAM on image.
        img: H x W x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels and segment masks
            detections: If there are N detections,
                - xyxy: (N, 4) boxes (int pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
                - mask: (N, H, W) boolean segmentation masks, i.e., True at locations belonging to corresponding class
        """
        # TODO: Hack
        if "entrance" in text_prompts:
            text_prompts = text_prompts + ["vertical-wall", "silver-trashcan", "blue-board"]
            idx = text_prompts.index("entrance")
            text_prompts[0], text_prompts[idx] = text_prompts[idx], text_prompts[0]  # bring it to 0th index
        elif "staircase" in text_prompts:
            text_prompts = text_prompts + ["vertical-wall", "pole", "board"]
            idx = text_prompts.index("staircase")
            text_prompts[0], text_prompts[idx] = text_prompts[idx], text_prompts[0]
        _, detections = self.predict_on_image(img, text_prompts, do_nms=do_nms, box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
        if "entrance" in text_prompts or "staircase" in text_prompts:
            detections = GroundedSAM.HACK_filter_keep0(detections)
        detections.mask = GroundedSAM.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_image = self.mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        detections.xyxy = detections.xyxy.astype(np.int32).reshape((-1, 4))
        return annotated_image, detections, GroundedSAM.get_per_class_mask(img, detections.mask, detections.class_id, len(text_prompts))


if __name__ == "__main__":
    # dirpath = os.path.join(nspl_root_dir, "examples_total/syncdata/21/images")
    # prompts = ["wall"]
    # box_threshold, text_threshold, nms_threshold = (0.5, 0.5, 0.4)

    # all_images_paths = [os.path.join(dirpath, file) for file in sorted(os.listdir(dirpath))]
    # ods = GroundedSAM(box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
    # for image_path in all_images_paths:
    #     image = cv2.imread(image_path)
    #     ann_img, detections, p = ods.predict_and_segment_on_image(image, prompts)
    #     print(detections.confidence)
    #     # print(p.shape)
    #     # ann_img = cv2.resize(ann_img, None, fx=0.6, fy=0.6)
    #     cv2.imshow("annotated", ann_img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image_path = "/home/dynamo/Music/jackal_bags/backup_unified_dataset/images/morning_mode1_11062023_000249.png"
    # prompts = ["white parking lines", "road", "car"]
    # box_threshold, text_threshold, nms_threshold = (0.1, 0.1, 0.2)

    OBJECT_THRESHOLDS = {  # 3-tuple of (box_threshold, text_threshold, nms_threshold)
        "barricade": (0.5, 0.5, 0.3),
        "board": (0.3, 0.3, 0.5),
        "bush": (0.4, 0.4, 0.4),
        "car": (0.3, 0.3, 0.3),
        "entrance": (0.3, 0.3, 0.2),
        "person": (0.25, 0.25, 0.6),
        "pole": (0.4, 0.4, 0.5),
        "staircase": (0.25, 0.25, 0.4),
        "tree": (0.4, 0.4, 0.45),
        "wall": (0.5, 0.5, 0.4)
    }

    image_path = "/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/evals_data_safety/utcustom/eval/images/000032_morning_mode1_11062023_000724.png"
    prompts = [list(OBJECT_THRESHOLDS.keys())[0]]
    box_threshold, text_threshold, nms_threshold = OBJECT_THRESHOLDS.get(prompts[0])
    ods = GroundedSAM(box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
    image = cv2.imread(image_path)
    ann_img, detections, p = ods.predict_and_segment_on_image(image, prompts)
    cv2.imshow("annotated", ann_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
