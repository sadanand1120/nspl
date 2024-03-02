import os
import sys
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
import random
from torchvision.transforms import Compose, ColorJitter, RandomAdjustSharpness, RandomSolarize, GaussianBlur
from torchvision.transforms.functional import hflip
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, TrainingArguments, Trainer, EarlyStoppingCallback
from torch import nn
import sys
import time
import signal
import evaluate
from datasets import DatasetDict
import json
from git import Repo
from torch.nn import CrossEntropyLoss
import numpy as np
from easydict import EasyDict
from parking.dutils import remove_username
from PIL import Image
from copy import deepcopy
from simple_colors import red


class CustomTrainer(Trainer):
    def __init__(self, myweights: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.myweights = myweights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        if self.myweights is None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
        else:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.myweights, device=self.args.device), ignore_index=0)
        loss = loss_fct(upsampled_logits, labels)
        return (loss, outputs) if return_outputs else loss


class RandomHorizontalFlipBoth:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return hflip(img), hflip(mask)
        return img, mask


def prepare_dataset(hfdi: str, train_percentd: float, val_percentd: float):
    if train_percentd + val_percentd > 1.0:
        raise ValueError("Sum of train and validation percentages must be b/w 0.0 and 1.0.")
    elif train_percentd + val_percentd == 1.0:
        print("Sum of train and validation percentages is equal to 1.0. Test set cannot be empty. Allocating 0.01 from train to test.")
        train_percentd -= 0.01
    ds = load_dataset(hfdi)
    train_ds, val_ds, test_ds = split_dataset(ds, train_percentd, val_percentd)
    return train_ds, val_ds, test_ds, ds['train']


def apply_transforms(train_ds, val_ds, train_trfn_name, val_trfn_name):
    train_ds.set_transform(train_trfn_name)
    val_ds.set_transform(val_trfn_name)
    return train_ds, val_ds


def prepare_model(pretrained_model_name, id2label, label2id):
    global global_params
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        num_channels=6 if global_params.use_depth else 3,
    )
    return feature_extractor, model


def train_transforms(example_batch):
    global global_params, id2label, label2id
    image_transforms = Compose([
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        RandomSolarize(threshold=128, p=0.5),
        GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ])
    flip_transform = RandomHorizontalFlipBoth(p=0.5)
    images = []
    labels = []
    for img, mask in zip(example_batch['pixel_values'], example_batch['labels']):
        img, mask = flip_transform(img, mask)
        img = image_transforms(img)  # Apply other image-only transforms
        images.append(img)
        labels.append(mask)
    inputs = global_params.feature_extractor(images, labels)
    return inputs


def val_transforms(example_batch):
    global global_params, id2label, label2id
    images = example_batch['pixel_values']
    labels = example_batch['labels']
    inputs1 = global_params.feature_extractor(images, labels)
    if not global_params.use_depth:
        return inputs1
    names = example_batch['name']
    depth_images = []
    for name in names:
        depth_images.append(global_params.all_depth_pil_images[name])
    inputs2 = global_params.feature_extractor(depth_images, labels)
    inputs = deepcopy(inputs1)
    for i in range(len(inputs['pixel_values'])):
        rgb = inputs1['pixel_values'][i]
        d = inputs2['pixel_values'][i]
        inputs['pixel_values'][i] = np.concatenate((rgb, d), axis=0)
    return inputs


def git_pull(repo_path):
    repo = Repo(repo_path)
    if not repo.bare:
        print('Pulling latest changes for repository: {0}'.format(repo_path))
        repo.remotes.origin.pull()
    else:
        print('Cannot pull changes, the repository is bare.')


def signal_handler(sig, frame):
    print("Ctrl+C detected! Stopping training and saving model and pushing to hub...")
    time.sleep(2)
    try:
        end_training_cleanly()
    except:
        print("LOGINFO: Error while saving model and pushing to hub!")
        pass
    time.sleep(6)
    sys.exit(0)


def end_training_cleanly():
    global global_params, id2label, label2id
    global_params.trainer.save_model()  # so that the best model will get now loaded when you do from_pretrained()
    global_params.feature_extractor.push_to_hub(global_params.hub_model_id, use_auth_token=global_params.hf_key)
    git_pull(global_params.output_dir)
    kwargs = {
        "tags": ["vision", "image-segmentation"],
        "finetuned_from": global_params.pretrained_model_name,
        "dataset": global_params.hf_dataset_identifier,
    }
    global_params.trainer.push_to_hub(**kwargs)
    print("Training terminated cleanly! Model saved and pushed to hub!")


def train():
    global global_params, id2label, label2id
    num_params = sum(p.numel() for p in global_params.model.parameters() if p.requires_grad)
    print(red(f"LOGINFO: Number of trainable parameters: {num_params}", 'bold'))
    if global_params.continue_training:
        print("Continuing training from the last checkpoint...")
        global_params.trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch...")
        global_params.trainer.train()
    end_training_cleanly()


def split_dataset(ds: DatasetDict, train_percent: float, val_percent: float):
    assert 0.0 <= train_percent + val_percent <= 1.0, "Sum of train and validation percentages must be b/w 0.0 and 1.0."
    ds = ds.shuffle(seed=42)
    train_temp_ds = ds["train"].train_test_split(test_size=1.0 - train_percent, seed=42)
    val_prop = val_percent / (1.0 - train_percent)
    val_test_ds = train_temp_ds["test"].train_test_split(test_size=1.0 - val_prop, seed=42)
    _train_ds = train_temp_ds["train"]
    _val_ds = val_test_ds["train"]
    _test_ds = val_test_ds["test"]
    return _train_ds, _val_ds, _test_ds


def compute_metrics(eval_pred):
    global global_params, id2label, label2id
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = global_params.metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=global_params.feature_extractor.do_reduce_labels,
        )
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        return metrics


def prepare_trainer():
    global global_params, id2label, label2id
    training_args = TrainingArguments(
        output_dir=global_params.output_dir,
        learning_rate=global_params.lr,
        lr_scheduler_type=global_params.lr_decay,
        weight_decay=global_params.lambd,
        dataloader_num_workers=global_params.dataloader_num_workers,
        overwrite_output_dir=False,
        evaluation_strategy=global_params.global_strategy,
        eval_steps=global_params.eval_steps,
        report_to="tensorboard",  # new version reports to all (including wandb)
        eval_accumulation_steps=5,
        per_device_train_batch_size=global_params.device_batch_size,
        per_device_eval_batch_size=global_params.device_batch_size,
        num_train_epochs=global_params.epochs,
        warmup_ratio=0.05,
        log_level="warning",
        log_level_replica="warning",
        logging_strategy=global_params.global_strategy,
        logging_first_step=True,
        logging_steps=1,
        save_strategy=global_params.global_strategy,
        save_steps=4 * global_params.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # metric_for_best_model's behaviour
        ignore_data_skip=False,  # for resuming training from the exact same condition
        optim="adamw_hf",
        push_to_hub=True,
        hub_model_id=global_params.hub_model_id,
        hub_strategy="all_checkpoints",
        hub_token=global_params.hf_key,
        hub_private_repo=False,
        auto_find_batch_size=False,
        remove_unused_columns=False,
    )

    if global_params.do_early_stopping:
        early_stopping_patience = global_params.early_stopping_patience  # Number of evaluations with no improvement after which training will be stopped.
        early_stopping_threshold = 0.0  # Minimum change needed to qualify as an improvement.
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )

    global_params.trainer = CustomTrainer(
        model=global_params.model,
        args=training_args,
        train_dataset=global_params.train_ds,
        eval_dataset=global_params.val_ds,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback] if global_params.do_early_stopping else None,
        myweights=global_params.myweights,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',
                        type=str,
                        required=True,
                        help='CUDA gpus allowed to be used (comma-separated)')

    parser.add_argument('-c', '--continue_training',
                        action='store_true',
                        help='Flag to indicate if resume training from the last checkpoint')

    parser.add_argument('-d', '--hf_dataset_name',
                        type=str,
                        default="terrain-jackal-morning_v0.1",
                        help='Name of the huggingface dataset')

    # Provide either of the two
    parser.add_argument('-m', '--hf_model_name',
                        type=str,
                        default=None,
                        help='Name of the HuggingFace model that will be trained')
    parser.add_argument('-v', '--hf_model_ver',
                        type=int,
                        default=0,
                        help='Version of the HuggingFace model that will be trained')

    parser.add_argument('-i', '--segformer_model_id',
                        type=int,
                        default=5,
                        help='ID of the Segformer model to use')

    parser.add_argument('-e', '--epochs',  # total epochs, not "more epochs"
                        type=int,
                        default=400,
                        help='Number of epochs to train for')

    parser.add_argument('-l', '--lr',
                        type=float,
                        default=5e-5,
                        help='Learning rate')

    parser.add_argument('--lambd',
                        type=float,
                        default=1e-5,
                        help='Lambda regularization parameter')

    parser.add_argument('--lr_decay',
                        type=str,
                        default="linear",
                        help='Learning rate decay type')

    parser.add_argument('--dataloader_num_workers',
                        type=int,
                        default=6,
                        help='Number of workers to use for data loading')

    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')

    parser.add_argument('-s', '--eval_steps',
                        type=int,
                        default=20,
                        help='Number of steps after which to evaluate')

    parser.add_argument('--train_percentd',
                        type=float,
                        default=0.7,
                        help='Percentage of dataset to use for training')

    parser.add_argument('--val_percentd',
                        type=float,
                        default=0.3,
                        help='Percentage of dataset to use for validation')

    parser.add_argument('--do_early_stopping',
                        action='store_true',
                        help='Flag to indicate if early stopping should be done')

    parser.add_argument('--do_transforms',
                        action='store_true',
                        help='Flag to indicate if train transforms should be applied')

    parser.add_argument('--use_depth',
                        action='store_true',
                        help='Flag to indicate if train transforms should be applied')

    parser.add_argument('--depth_dir',
                        type=str,
                        default=None,
                        help='Path to the depth images directory')

    parser.add_argument('--loss_mode',  # does not give expected betterment!
                        type=int,
                        default=None,
                        help='0-terrain, 1-safety')

    parser.add_argument('--early_stopping_patience',
                        type=int,
                        default=20,
                        help='Number of evaluations with no improvement after which training will be stopped')

    args = parser.parse_args()

    if args.use_depth and args.depth_dir is None:
        raise ValueError("Please provide the path to the depth images directory")

    if args.use_depth and args.do_transforms:
        print("LOGINFO: Using depth images and applying transforms not possible. Ignoring transforms")
        args.do_transforms = False

    global_params = EasyDict()
    global_params.update(vars(args))

    # in weights ignore index 0 (unlabeled)
    if global_params.loss_mode == 0:
        ds_perc = {0: 0.0,
                   1: 60.0731,
                   2: 8.3973,
                   3: 3.2594,
                   4: 18.0778,
                   5: 1.0806,
                   6: 0.474,
                   7: 0.4219,
                   8: 7.0253,
                   9: 0.5043,
                   10: 0.1111,
                   11: 0.1001,
                   12: 0.4653,
                   13: 0.0097}
        pref_weights = {0: 0.0,  # unlabeled
                        1: 4.0,  # NAT
                        2: 10.0,  # concrete
                        3: 10.0,  # grass
                        4: 10.0,  # speedway bricks
                        5: 2.0,  # steel
                        6: 3.0,  # rough concrete
                        7: 3.0,  # dark bricks
                        8: 5.0,  # road
                        9: 8.0,  # rough red sidewalk
                        10: 6.0,  # tiles
                        11: 2.0,  # red bricks
                        12: 10.0,  # concrete tiles
                        13: 1.0}  # REST
        global_params.myweights = [(max(ds_perc.values()) / max(ds_perc[i], 6.0)) * 2 * pref_weights[i] if i != 0 else 1.0 for i in range(len(ds_perc))]
    elif global_params.loss_mode == 1:
        ds_perc = {0: 0.0, 1: 3.2165, 2: 96.7835}
        pref_weights = {0: 0.0, 1: 1.0, 2: 1.0}
        global_params.myweights = [(max(ds_perc.values()) / max(ds_perc[i], 6.0)) * 2 * pref_weights[i] if i != 0 else 1.0 for i in range(len(ds_perc))]
    else:
        global_params.myweights = None

    print("LOGINFO: Using weights: ", global_params.myweights)

    if global_params.use_depth:
        all_depth_pil_images = {}
        for img_name in sorted(os.listdir(global_params.depth_dir)):
            img_path = os.path.join(global_params.depth_dir, img_name)
            img = Image.open(img_path)
            all_depth_pil_images[img_name] = img
        global_params.all_depth_pil_images = all_depth_pil_images

    global_params.hf_dataset_name = remove_username(global_params.hf_dataset_name)
    global_params.hf_username = "sam1120"

    global_params.hf_key = os.environ.get("HF_API_KEY")
    os.environ["CUDA_VISIBLE_DEVICES"] = global_params.cuda
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    os.makedirs(model_path, exist_ok=True)
    if torch.cuda.is_available():
        device_batch_size = global_params.batch_size // torch.cuda.device_count()
    else:
        device_batch_size = global_params.batch_size
    global_params.model_path = model_path
    global_params.device_batch_size = device_batch_size
    id2label_filename = "id2label.json"
    global_params.global_strategy = "steps"
    global_params.hf_dataset_identifier = f"{global_params.hf_username}/{global_params.hf_dataset_name}"
    global_params.pretrained_model_name = f"nvidia/mit-b{global_params.segformer_model_id}"
    global_params.hub_model_id = global_params.hf_model_name
    if global_params.hub_model_id is None:
        global_params.hub_model_id = f"segformer-b{global_params.segformer_model_id}-finetuned-{global_params.hf_dataset_name}-v{global_params.hf_model_ver}"
    else:
        global_params.hub_model_id = remove_username(global_params.hub_model_id)
    id2label = json.load(open(hf_hub_download(repo_id=global_params.hf_dataset_identifier, filename=id2label_filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    global_params.num_labels = len(id2label)
    global_params.metric = evaluate.load("mean_iou")
    global_params.output_dir = f"{model_path}/{global_params.hub_model_id}"

    signal.signal(signal.SIGINT, signal_handler)

    global_params.trainer = None
    global_params.feature_extractor = None
    global_params.train_ds = None
    global_params.val_ds = None
    global_params.test_ds = None
    global_params.model = None

    print("LOGINFO: Preparing model...")
    global_params.feature_extractor, global_params.model = prepare_model(pretrained_model_name=global_params.pretrained_model_name,
                                                                         id2label=id2label,
                                                                         label2id=label2id)

    print("LOGINFO: Preparing dataset...")
    global_params.train_ds, global_params.val_ds, global_params.test_ds, _ = prepare_dataset(hfdi=global_params.hf_dataset_identifier,
                                                                                             train_percentd=global_params.train_percentd,
                                                                                             val_percentd=global_params.val_percentd)
    global_params.train_ds, global_params.val_ds = apply_transforms(train_ds=global_params.train_ds,
                                                                    val_ds=global_params.val_ds,
                                                                    train_trfn_name=train_transforms if global_params.do_transforms else val_transforms,
                                                                    val_trfn_name=val_transforms)

    print("LOGINFO: Preparing trainer...")
    prepare_trainer()

    print("LOGINFO: Starting training...")
    train()
