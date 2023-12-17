
import datetime
import gc
import itertools
import json
import logging
import os
import shutil
import math
from pathlib import Path

import albumentations as A
import cv2
import japanize_matplotlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import faiss

class Config:
    competition = "sake"
    name = "meigara_classification_baseline_archface"

    debug = False

    training = True
    evaluation = True
    embedding = True

    seed = 8823
    n_fold = 5
    trn_fold = [0,1,2,3,4]

    target_columns = ["meigara_label"]
    size = 512

    model_name = "tf_efficientnet_b0_ns"
    max_epochs = 5
    train_batch_size = 32
    valid_batch_size = 20 # 128
    num_workers = 4
    gradient_accumulation_steps = 1
    clip_grad_norm = 1000

    optimizer = dict(
        optimizer_name="AdamW",
        lr=1e-4,
        weight_decay=1e-2,
        eps=1e-6,
        beta=(0.9, 0.999),
        encoder_lr=1e-4,
        decoder_lr=1e-4,
    )

    scheduler = dict(
        scheduler_name="cosine",
        num_warmup_steps_rate=0,
        num_cycles=0.5,
    )
    batch_scheduler = True

def setup(Config):
    # mount
    from google.colab import drive

    CONTENT_DRIVE = Path("/content/drive")
    if not CONTENT_DRIVE.is_dir():
        drive.mount(CONTENT_DRIVE.as_posix())

    for d in [
        HOME,
        INPUTS,
        SUBMISSIONS,
        EXP_MODELS,
        EXP_REPORTS,
        EXP_PREDS,
        INTERMIDIATES,
        SCRIPTS,
    ]:
        d.mkdir(parents=True, exist_ok=True)

def check_file_exists(folder_path, file_name):
    folder = Path(folder_path)
    for file_path in folder.glob("**/*"):
        if file_path.is_file() and file_path.stem == file_name:
            return True
    return False


def create_new_datasets_in_kaggle(dataset_name, upload_dir):
    from kaggle.api.kaggle_api_extended import KaggleApi

    dataset_metadata = {}
    dataset_metadata["id"] = f'{os.environ["KAGGLE_USERNAME"]}/{dataset_name}'
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = dataset_name
    with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)
    api = KaggleApi()
    api.authenticate()
    api.dataset_create_new(folder=upload_dir, convert_to_csv=False, dir_mode="tar")


class Logger:
    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, "Experiment.log"))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def copy_file(source_file, destination_file):
    try:
        shutil.copy(source_file, destination_file)
        print("copy complete")
    except IOError as e:
        print(f"copy error: {e}")

def make_train_label(train_df) -> pd.DataFrame:
    le = LabelEncoder()
    train_df["brand_id_label"] = le.fit_transform(train_df["brand_id"])
    train_df["meigara_label"] = le.fit_transform(train_df["meigara"])

    return train_df


def make_filepath(input_df):
    def join_path(dirpath, filename):
        return (dirpath / filename).as_posix()

    output_df = input_df.assign(
        filepath=input_df["filename"].apply(
            lambda x: join_path(QUERY_IMAGES, x) if str(x)[0] == "2" else join_path(CITE_IMAGES, x)
        )
    )
    return output_df

def make_label_dict(train_df):
    MLABEL2MEIGARA = train_df[["meigara", "meigara_label"]].set_index("meigara_label").to_dict()["meigara"]
    BLABEL2BLAND = train_df[["brand_id", "brand_id_label"]].set_index("brand_id_label").to_dict()["brand_id"]
    return MLABEL2MEIGARA, BLABEL2BLAND

def add_fold_idx(config, train_df):
    fold = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
    train_df["fold"] = -1
    for i_fold, (train_index, val_index) in enumerate(
        fold.split(train_df, train_df[config.target_columns])
    ):
        train_df.iloc[val_index, train_df.columns.get_loc("fold")] = int(i_fold)
    train_df["fold"] = train_df["fold"].astype(int)
    return train_df

class TrainDataset(Dataset):
    def __init__(self,df, target_columns=Config.target_columns, transform_fn=None):
        self.df = df
        self.file_names = df["filepath"].to_numpy()
        self.targets = df[target_columns].to_numpy()
        if len(target_columns) == 1:
            self.targets = np.ravel(self.targets)

        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.file_names[idx]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform_fn:
            image = self.transform_fn(image=image)["image"]
        target = torch.tensor(self.targets[idx])
        return {"images":image, "targets":target}


class TestDataset(Dataset):
    def __init__(self, df, transform_fn=None):
        self.df = df
        self.file_names = df["filepath"].to_numpy()
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.file_names[idx]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform_fn:
            image = self.transform_fn(image=image)["image"]
        return {"images":image}

def get_transforms(*, size, data="train"):

    if data == 'train':
        return A.Compose([
            # 正方形切り出し
            A.RandomResizedCrop(size, size, scale=(0.85, 1.0)),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

# class ArcFaceLoss(nn.Module):
#     def __init__(self, num_classes, emb_size, device, s=30.0, m=0.50, easy_margin=False):
#         super(ArcFaceLoss, self).__init__()
#         self.s = s
#         self.m = m
#         self.easy_margin = easy_margin
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
#         self.weight = self.weight.to(device)
#         nn.init.xavier_uniform_(self.weight)
        
#         # m parameter
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m


#     def forward(self, inputs, targets):
#         cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
#         sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         one_hot = torch.zeros(cosine.size(), device=targets.device)
#         one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.s
#         loss = F.cross_entropy(output, targets)
#         return loss

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        th = math.cos(math.pi - self.m)
        mm = math.sin(math.pi - self.m) * self.m

        # cos(theta + m)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)

        # output
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class CustomModel(nn.Module):
    def __init__(self, config, out_dim, pretrained=False):
        super().__init__()
        self.config = config
        self.model = timm.create_model(self.config.model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()  # custom head にするため
        self.arcface = ArcFace(in_features=self.n_features, out_features=out_dim)

    def feature(self, images):
        feature = self.model(images)
        return feature

    def forward(self, images, labels=None):
        x = self.feature(images)
        x = self.arcface(x, labels)
        return x



def train_fn(
    config,
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    wandb_logger,
    _custom_step,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = []

    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in tbar:
        for k, v in batch.items():
            batch[k] = v.to(device)
        targets = batch["targets"]
        batch_size = targets.size(0)

        with torch.cuda.amp.autocast():
            batch_outputs = model(batch["images"], labels=batch["targets"])
            loss = criterion(batch_outputs, targets)

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()
        if config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            _custom_step += 1

            if config.batch_scheduler:
                scheduler.step()

        wandb_logger.log(
            {"train_loss": loss, "lr": scheduler.get_lr()[0], "train_step": _custom_step}
        )

        losses.append(float(loss))
        tbar.set_description(f"loss: {np.mean(losses):.4f} lr: {scheduler.get_lr()[0]:.6f}")

    loss = np.mean(losses)
    return loss, _custom_step


def valid_fn(
    config,
    model,
    dataloader,
    criterion,
    device,
    wandb_logger,
    _custom_step,
):
    model.eval()
    outputs, targets = [], []
    losses = []

    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in tbar:
        targets.append(batch["targets"])

        for k, v in batch.items():
            batch[k] = v.to(device)

        batch_size = batch["targets"].size(0)
        with torch.no_grad():
            batch_outputs = model(batch["images"], labels=batch["targets"])
            loss = criterion(batch_outputs, batch["targets"])

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        batch_outputs = torch.softmax(batch_outputs, dim=1)  # to proba
        batch_outputs = batch_outputs.to("cpu").numpy()
        outputs.append(batch_outputs)

        wandb_logger.log({"valid_loss": loss, "valid_step": _custom_step})
        _custom_step += 1
        losses.append(float(loss))

        tbar.set_description(f"loss: {np.mean(losses):.4f}")

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    loss = np.mean(losses)
    return (loss, outputs, targets, _custom_step)

def get_optimizer(optimizer_config, model):
    if optimizer_config["optimizer_name"] == "AdamW":
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=optimizer_config["beta"],
            eps=optimizer_config["eps"],
        )
        return optimizer
    else:
        raise NotImplementedError


def get_scheduler(scheduler_config, optimizer, num_train_steps):
    if scheduler_config["scheduler_name"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(scheduler_config["num_warmup_steps_rate"] * num_train_steps),
            num_training_steps=num_train_steps,
        )
        return scheduler

    elif scheduler_config["scheduler_name"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(scheduler_config["num_warmup_steps_rate"] * num_train_steps),
            num_training_steps=num_train_steps,
            num_cycles=scheduler_config["num_cycles"],
        )
        return scheduler

    elif scheduler_config["scheduler_name"] == "cosine_restarts":
        """
        example:
            first_cycle_steps_ratio = 0.25,
            cycle_mult = 1.0,
            max_lr = 2e-5,
            min_lr = 1e-7,
            warmup_steps=100,
            gamma=0.8)
        """
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(num_train_steps * scheduler_config["first_cycle_steps_ratio"]),
            cycle_mult=scheduler_config["cycle_mult"],
            max_lr=scheduler_config["max_lr"],
            min_lr=scheduler_config["min_lr"],
            warmup_steps=scheduler_config["warmup_steps"],
            gamma=scheduler_config["gamma"],
        )
        return scheduler

    else:
        raise NotImplementedError


def mrr_at_k(y_true, y_pred, k: int=20):
    """
    MRR（Mean Reciprocal Rank）@k を計算する関数。
    y_true : 各クエリに対する正解ラベルのリスト（正解アイテムのインデックス）。
    y_pred : モデルの予測スコア。各クエリに対する全アイテムの予測スコアのリスト。
    """
    mrr = 0.0
    for i in tqdm(range(len(y_true)), desc="[mrr]"):
        # Sort predictions and get the top k
        top_k_indices = np.argsort(y_pred[i])[::-1][:k]
        for rank, index in enumerate(top_k_indices):
            if index == y_true[i]:
                mrr += 1.0 / (rank + 1)
                break
    mrr /= len(y_true)
    return mrr

def train_loop(config, name, train_df, valid_df, out_dim, device):
    LOGGER.info(f"========== {name} training ==========")

    # set wandb logger
    wandb.init(
        project=config.competition,
        name=name,
        group=f"{config.name}",
        job_type="train",
        anonymous=None,
        reinit=True,
    )

    # dataset, dataloader
    train_dataset = TrainDataset(
        df=train_df,
        target_columns=config.target_columns,
        transform_fn=get_transforms(data="train", size=config.size)
    )
    valid_dataset = TrainDataset(
        df=valid_df,
        target_columns=config.target_columns,
        transform_fn=get_transforms(data="valid", size=config.size)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # set model & optimizer
    model = CustomModel(config, out_dim=out_dim, pretrained=True)
    model.to(device)
    optimizer = get_optimizer(optimizer_config=config.optimizer, model=model)

    # set scheduler
    num_train_steps = int(
        len(train_dataloader) * config.max_epochs // config.gradient_accumulation_steps
    )
    scheduler = get_scheduler(
        scheduler_config=config.scheduler, optimizer=optimizer, num_train_steps=num_train_steps
    )

    # loop
    criterion = nn.CrossEntropyLoss()
    score_fn = mrr_at_k  # mrr を使わなくもいいかも

    best_score = -np.inf
    tr_step, val_step = 0, 0
    for epoch in range(Config.max_epochs):
        # training
        loss, tr_step = train_fn(
            config=config,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            wandb_logger=wandb,
            _custom_step=tr_step,
        )

        # validation
        val_loss, val_outputs, val_targets, val_step = valid_fn(
            config=config,
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            device=device,
            wandb_logger=wandb,
            _custom_step=val_step,
        )

        # calc score (the higher, the better)
        eval_score = score_fn(y_pred=val_outputs, y_true=val_targets)
        logs = {
            "Epoch": epoch,
            "eval_score": eval_score,
            "train_loss_epoch": loss.item(),
            "valid_loss_epoch": val_loss.item(),
        }
        wandb.log(logs)
        LOGGER.info(logs)

        if best_score < eval_score:
            best_score = eval_score
            LOGGER.info(f"epoch {epoch} - best score: {best_score:.4f} model")

            torch.save(model.state_dict(), f"{name}.pth")  # save model weight
            joblib.dump(val_outputs, f"{name}.pkl")  # save outputs

        if not config.batch_scheduler:
            scheduler.step()

    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish(quiet=True)

    # to escape drive storage error
    copy_file(f"{name}.pth", EXP_MODELS / f"{name}.pth")
    copy_file(f"{name}.pkl", EXP_PREDS / f"{name}.pkl")

    # save best predictions with id
    best_val_outputs = joblib.load(f"{name}.pkl")
    outputs = {
        "gid": valid_df["gid"].tolist(),
        "predictions": np.array(best_val_outputs, dtype=np.float16),
        "targets":val_targets,  # type: ignore
    }
    joblib.dump(outputs, EXP_PREDS / f"{name}_best.pkl")


def inference_fn(test_dataloader, model, device, features=False):
    preds, targets_masks = [], []
    model.eval()
    model.to(device)

    tbar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in tbar:
        for k, v in batch.items():
            batch[k] = v.to(device)

        if not features:
            with torch.no_grad():
                outputs = model(batch["images"])
                outputs = torch.softmax(outputs, dim=1) # to proba

            outputs = outputs.cpu().detach().numpy()
            preds.append(outputs)
        else:
            with torch.no_grad():
                outputs = model.feature(batch["images"])

            outputs = outputs.cpu().detach().numpy()
            preds.append(outputs)

    return np.concatenate(preds)


def get_predictions(config, test_df, model_path, out_dim, device):
    test_dataset = TestDataset(df=test_df, transform_fn=get_transforms(data="valid", size=config.size))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # get model
    model = CustomModel(config, out_dim=out_dim, pretrained=False)
    state = torch.load(model_path)
    model.load_state_dict(state)
    predictions = inference_fn(test_dataloader, model, device)

    del model, state, test_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    outputs = {
        "gid": test_df["gid"].tolist(),
        "predictions": np.array(predictions, dtype=np.float16),
    }
    return outputs


def get_features(config, test_df, model_path, out_dim, device):
    test_dataset = TestDataset(df=test_df, transform_fn=get_transforms(data="valid", size=config.size))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # get model
    model = CustomModel(config, out_dim=out_dim, pretrained=False)
    state = torch.load(model_path)
    model.load_state_dict(state)
    features = inference_fn(test_dataloader, model, device, features=True)

    del model, state, test_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    outputs = {
        "gid": test_df["gid"].tolist(),
        "features": np.array(features, dtype=np.float16),
    }
    return outputs

class SimilaritySearcher:
    def __init__(self, embeddings):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)  # type:ignore

    def search(self, queries, k=10):
        assert queries.shape[1] == self.dimension, "Query dimensions should match embeddings dimension."
        faiss.normalize_L2(queries)
        D, I = self.index.search(queries, k)  # type:ignore
        return D, I

def make_submission(indices):
    vfunc = np.vectorize(lambda x: IDX2CITE_GID[x])
    gid_array = vfunc(I)
    submission_df = test_df[["gid"]].assign(cite_gid=[" ".join(list(x)) for x in gid_array.astype(str)])
    return submission_df

import cv2
import matplotlib.pyplot as plt

def display_similar_images(query_path, cite_paths):
    num_similar = len(cite_paths)

    # Query画像の読み込み
    query_image = cv2.imread(query_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    # 類似画像の読み込みと表示
    num_rows = (num_similar + 1) // 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 10), squeeze=False)

    axs[0, 0].imshow(query_image)
    axs[0, 0].set_title(f'Query Image: {query_path.split("/")[-1]}')
    axs[0, 0].axis('off')

    for i, path in enumerate(cite_paths):
        similar_image = cv2.imread(path)
        similar_image = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)

        axs[(i+1)//2, (i+1)%2].imshow(similar_image)
        axs[(i+1)//2, (i+1)%2].set_title(f'Similar Image {i+1}: {path.split("/")[-1]}')
        axs[(i+1)//2, (i+1)%2].axis('off')

    # 空白のsubplotを削除
    if num_similar % 2 != 0:
        axs[num_rows-1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def get_qyery_and_cite_path(submission_df, query_gid=None, k=4):
    if query_gid is None:
        query_gid = submission_df["gid"].sample(1).to_numpy()[0]

    cites = submission_df.loc[submission_df["gid"] ==query_gid, "cite_gid"].to_numpy()[0].split(" ")[:k]

    query_path = QUERY_IMAGES / f"{query_gid}.jpg"
    cite_paths = [(CITE_IMAGES / f"{cite_gid}.jpg").as_posix() for cite_gid in cites]
    return {"query_path":query_path.as_posix(), "cite_paths":cite_paths}


def plot_sake(submission_df, query_gid=None, k=11):
    path_dict = get_qyery_and_cite_path(submission_df, query_gid=query_gid, k=k)
    display_similar_images(**path_dict)

if __name__ == '__main__':
    if Config.debug:
        Config.max_epochs = 2
        Config.n_fold = 2
        Config.trn_fold = [0, 1]
        Config.name = Config.name + "_debug"
        Config.size = 128
        Config.train_batch_size = 8

    # constants
#     HOME = Path("/home/ec2-user/sake/")
    HOME = Path("/home/ec2-user/gitwork/nishika_sake")
    NOTEBOOK_NAME = "notebooke"
    EXP_NAME = Config.name if Config.name is not None else NOTEBOOK_NAME
    INPUTS = HOME / "data/inputs"  # input data
    OUTPUTS = HOME / "data/outputs"
    INTERMIDIATES = HOME / "data/intermidiates"  # intermidiate outputs
    SUBMISSIONS = HOME / "data/submissions"
    OUTPUTS_EXP = OUTPUTS / EXP_NAME
    EXP_MODELS = OUTPUTS_EXP / "models"
    EXP_REPORTS = OUTPUTS_EXP / "reports"
    EXP_PREDS = OUTPUTS_EXP / "predictions"
    SCRIPTS = HOME / "scripts"

#     CITE_IMAGES = Path("/home/ec2-user/sake/data/inputs/cite_images")
#     QUERY_IMAGES = Path("/home/ec2-user/sake/data/inputs/query_images")
    CITE_IMAGES = Path("/home/ec2-user/gitwork/nishika_sake/data/inputs/cite_images")
    QUERY_IMAGES = Path("/home/ec2-user/gitwork/nishika_sake/data/inputs/query_images")
    

    LOGGER = Logger(OUTPUTS_EXP.as_posix())
    wandb.login()  # need wandb account
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load raw data
    train_df = pd.read_csv(INPUTS / "train.csv")
    test_df = pd.read_csv(INPUTS / "test.csv")
    cite_df = pd.read_csv(INPUTS / "cite.csv").rename(columns={"cite_gid":"gid", "cite_filename":"filename"})
    sample_submission_df = pd.read_csv(INPUTS / "sample_submission.csv")

    if Config.debug:
        train_df = train_df.sample(1000, random_state=Config.seed).reset_index(drop=True)
        cite_df = cite_df.sample(100, random_state=Config.seed).reset_index(drop=True)

    # make label
    train_df = make_train_label(train_df)
    MLABEL2MEIGARA, BLABEL2BLAND = make_label_dict(train_df)
    TARGET_DIM = len(MLABEL2MEIGARA)  # if use meigara as target

    # make filepath
    train_df = make_filepath(train_df)
    test_df = make_filepath(test_df)
    cite_df = make_filepath(cite_df)

    # for submission
    IDX2CITE_GID = cite_df.to_dict()["gid"]

    print(train_df.shape)

    train_df = add_fold_idx(config=Config, train_df=train_df)
    train_df.head()

    if Config.training:
        for i_fold in range(Config.n_fold):
            if i_fold not in Config.trn_fold:
                continue

            train_df_fold = train_df[train_df["fold"] != i_fold]
            valid_df_fold = train_df[train_df["fold"] == i_fold]

            train_loop(
                config=Config,
                name=f"fold_{i_fold}",
                train_df=train_df_fold,
                valid_df=valid_df_fold,
                out_dim=TARGET_DIM,
                device=DEVICE,
            )

    if Config.evaluation:
        outputs = [joblib.load(EXP_PREDS / f"fold_{i_fold}_best.pkl") for i_fold in range(Config.n_fold) if i_fold in Config.trn_fold]

        meta_df = pd.DataFrame(
            {"gid":list(itertools.chain.from_iterable([output["gid"] for output in outputs]))}
        ).merge(train_df, on="gid", how="left")
        preds_df = pd.concat([pd.DataFrame(output["predictions"]) for output in outputs ]).reset_index(drop=True).add_prefix("label_")

        meta_preds_df = pd.concat([meta_df, preds_df], axis=1)
        joblib.dump(meta_preds_df, EXP_PREDS / "oof_pred_df.pkl")

        score = mrr_at_k(y_true=meta_df[Config.target_columns].to_numpy(), y_pred=preds_df.to_numpy(), k=20)
        LOGGER.info(f"score={score:.4f}")

    if Config.embedding:
        # oof
        oof_features_filepath = EXP_PREDS / "oof_embeddings.pkl"
        if not oof_features_filepath.is_file():
            oof_features, oof_gids = [], []
            for i_fold in range(Config.n_fold):
                if i_fold not in Config.trn_fold:
                    continue

                gids = joblib.load(EXP_PREDS / f"fold_{i_fold}_best.pkl")["gid"]
                df = train_df[train_df["gid"].isin(gids)].reset_index(drop=True)

                outputs = get_features(
                    config=Config,
                    test_df=df,
                    model_path=EXP_MODELS / f"fold_{i_fold}.pth",
                    out_dim=TARGET_DIM,
                    device=DEVICE
                    )
                oof_features.append(outputs["features"])
                oof_gids.extend(outputs["gid"])
            oof_features = np.concatenate(oof_features, axis=0)
            joblib.dump({"gid":oof_gids, "embeddings":oof_features}, oof_features_filepath)

        # query images
        query_features_filepath = EXP_PREDS / "test_embeddings.pkl"
        if not query_features_filepath.is_file():
            query_features = []
            for i_fold in range(Config.n_fold):
                if i_fold not in Config.trn_fold:
                    continue

                outputs = get_features(
                    config=Config,
                    test_df=test_df,
                    model_path=EXP_MODELS / f"fold_{i_fold}.pth",
                    out_dim=TARGET_DIM,
                    device=DEVICE
                    )
                query_features.append(outputs["features"])
            joblib.dump({"gid":outputs["gid"], "embeddings_list":query_features}, query_features_filepath)

        # cite images
        cite_features_filepath = EXP_PREDS / "cite_embeddings.pkl"
        if not cite_features_filepath.is_file():
            cite_features = []
            for i_fold in range(Config.n_fold):
                if i_fold not in Config.trn_fold:
                    continue

                outputs = get_features(
                    config=Config,
                    test_df=cite_df,
                    model_path=EXP_MODELS / f"fold_{i_fold}.pth",
                    out_dim=TARGET_DIM,
                    device=DEVICE
                    )
                cite_features.append(outputs["features"])
            joblib.dump({"gid":outputs["gid"], "embeddings_list":cite_features}, cite_features_filepath)

    cite_features = joblib.load(EXP_PREDS / "cite_embeddings.pkl")["embeddings_list"]
    query_features = joblib.load(EXP_PREDS / "test_embeddings.pkl")["embeddings_list"]

    ave_cite_feature = np.mean(cite_features, axis=0)
    ave_query_feature = np.mean(query_features, axis=0)

    searcher = SimilaritySearcher(ave_cite_feature.astype(np.float32))
    D, I = searcher.search(ave_query_feature.astype(np.float32), k=20)

    submission_df = make_submission(indices=I)
    submission_df.to_csv(SUBMISSIONS / f"{Config.name}.csv", index=False)