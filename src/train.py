#!/usr/bin/env python3
import argparse, os, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from transformers import AutoImageProcessor, Siglip2VisionModel


# --------------------------------------------------------------------------- #
# Basic utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------------------------- #
# Dataset + split
class ImageFolderFlat(Dataset):
    """Flat folder-of-folders loader that keeps PIL images until the model processes them."""
    def __init__(self, root: str, transform: Optional[T.Compose] = None,
                 exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        for idx, cls_dir in enumerate(sorted([p for p in self.root.iterdir() if p.is_dir()])):
            self.class_to_idx[cls_dir.name] = idx
            for img_path in cls_dir.rglob("*"):
                if img_path.suffix.lower() in exts:
                    self.samples.append((str(img_path), idx))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            tensor = self.transform(img)
            img = T.ToPILImage()(tensor) if isinstance(tensor, torch.Tensor) else tensor
        return img, label


def build_train_val_indices_by_class(dataset, num_val_classes: int, seed: int):
    rng = random.Random(seed)

    # Collect indices per class
    per_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        per_class.setdefault(label, []).append(idx)

    all_classes = list(per_class.keys())
    rng.shuffle(all_classes)

    # Choose some classes entirely for val
    val_classes = set(all_classes[:num_val_classes])

    train_idx, val_idx = [], []
    for label, indices in per_class.items():
        if label in val_classes:
            val_idx.extend(indices)      # ALL images for this class -> val
        else:
            train_idx.extend(indices)    # ALL images for this class -> train

    return train_idx, val_idx


# def build_train_val_indices(dataset: ImageFolderFlat, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
#     """Put at least one example from every multi-image class into val; keep singletons in train."""
#     rng = random.Random(seed)
#     per_class: Dict[int, List[int]] = {}
#     for idx, (_, label) in enumerate(dataset.samples):
#         per_class.setdefault(label, []).append(idx)

#     train_idx, val_idx = [], []
#     for label, indices in per_class.items():
#         rng.shuffle(indices)
#         if len(indices) < 2:
#             train_idx.extend(indices)
#             continue
#         n_val = max(1, int(round(len(indices) * val_split)))
#         val_idx.extend(indices[:n_val])
#         train_idx.extend(indices[n_val:])
#     return train_idx, val_idx


# --------------------------------------------------------------------------- #
# Embedding model (no classifier)
class Siglip2EmbeddingModel(nn.Module):
    """SigLIP2 backbone followed by a projection head that outputs normalized embeddings."""
    def __init__(self, hf_id: str, embed_dim: int, bf16: bool):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        torch_dtype = torch.bfloat16 if bf16 else torch.float32
        self.backbone = Siglip2VisionModel.from_pretrained(
            hf_id,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        )
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, embed_dim)
        self.use_bf16 = bf16

    def encode(self, images: List[Image.Image], max_num_patches: Optional[int]) -> torch.Tensor:
        device = next(self.parameters()).device
        proc_kwargs = dict(return_tensors="pt", do_resize=True, do_center_crop=False)
        if max_num_patches is not None:
            proc_kwargs["max_num_patches"] = max_num_patches

        inputs = self.processor(images=images, **proc_kwargs)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=self.use_bf16 or device.type == "cuda"):
            features = self.backbone(**inputs).pooler_output

        embeddings = self.proj(features.float() if features.dtype != torch.float32 else features)
        return F.normalize(embeddings, dim=1)


# --------------------------------------------------------------------------- #
# Fine-tune helpers
def _backbone_layers(model: Siglip2EmbeddingModel) -> List[nn.Module]:
    layers: List[nn.Module] = []
    vision_model = getattr(model.backbone, "vision_model", None)
    if vision_model is not None:
        encoder = getattr(vision_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layers"):
            layers = list(encoder.layers)
    if not layers:
        encoder = getattr(model.backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layers"):
            layers = list(encoder.layers)
    return layers


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def freeze_backbone(model: Siglip2EmbeddingModel) -> None:
    _set_requires_grad(model.backbone, False)


def unfreeze_backbone(model: Siglip2EmbeddingModel) -> None:
    _set_requires_grad(model.backbone, True)


def unfreeze_backbone_layers(model: Siglip2EmbeddingModel, num_layers: int) -> None:
    if num_layers is None or num_layers <= 0:
        unfreeze_backbone(model)
        return

    freeze_backbone(model)
    layers = _backbone_layers(model)
    if not layers:
        # fallback: expose entire backbone if we cannot find layer list
        unfreeze_backbone(model)
        return

    for block in layers[-num_layers:]:
        _set_requires_grad(block, True)

    # also unfreeze trailing normalisation heads if present
    vision_model = getattr(model.backbone, "vision_model", None)
    if vision_model is not None:
        for attr in ("post_layernorm", "ln_post", "final_layernorm"):
            module = getattr(vision_model, attr, None)
            if module is not None:
                _set_requires_grad(module, True)


def build_optimizer(model: Siglip2EmbeddingModel,
                    head_lr: float,
                    backbone_lr: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})

    if not param_groups:
        raise ValueError("No trainable parameters available to optimise.")

    return torch.optim.AdamW(param_groups)


# --------------------------------------------------------------------------- #
# Data loader + metrics
def pil_collate(batch: List[Tuple[Image.Image, int]]) -> Tuple[List[Image.Image], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def recall_at_k(sim: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = sim.topk(k, dim=1).indices
    correct = (labels[topk] == labels.view(-1, 1)).any(dim=1)
    return correct.float().mean().item()


def mean_average_precision_at_k(sim: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    order = torch.argsort(sim, dim=1, descending=True)[:, :k]
    aps = []
    for idx in range(sim.size(0)):
        rel = (labels[order[idx]] == labels[idx]).float()
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        precision = torch.cumsum(rel, dim=0) / torch.arange(1, k + 1, device=rel.device).float()
        aps.append((precision * rel).sum().item() / rel.sum().item())
    return float(np.mean(aps))


def info_nce_loss(embeddings: torch.Tensor,
                  labels: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """
    Multi-positive InfoNCE over a batch of normalized embeddings.
    Each sample pulls all same-class examples (excluding itself) as positives.
    """
    device = embeddings.device
    sim = embeddings @ embeddings.t()
    sim = sim / temperature

    # mask out self similarities
    logits_mask = torch.ones_like(sim, dtype=torch.bool, device=device)
    logits_mask.fill_(True)
    logits_mask.fill_diagonal_(False)

    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & logits_mask
    num_pos = positive_mask.sum(dim=1)

    valid_mask = num_pos > 0
    if not valid_mask.any():
        raise ValueError("No positive pairs in the batch; increase batch size or adjust sampling.")

    sim_masked = sim.masked_fill(~logits_mask, float("-inf"))
    log_denom = torch.logsumexp(sim_masked, dim=1)

    sim_pos = sim.masked_fill(~positive_mask, float("-inf"))
    log_pos = torch.logsumexp(sim_pos, dim=1)

    loss = -(log_pos - log_denom)
    loss = loss[valid_mask].mean()
    return loss


@torch.no_grad()
def evaluate(model: Siglip2EmbeddingModel, loader: DataLoader, max_num_patches: Optional[int],
             device: torch.device, eval_k: int) -> Tuple[float, float, int, int]:
    model.eval()
    embeddings, labels_all = [], []

    for images, labels in loader:
        labels = labels.to(device)
        emb = model.encode(images, max_num_patches)
        embeddings.append(emb.cpu())
        labels_all.append(labels.cpu())

    if not embeddings:
        return 0.0, 0.0, 0, 0

    embeddings = torch.cat(embeddings)
    labels_all = torch.cat(labels_all)

    sim = embeddings @ embeddings.t()
    sim.fill_diagonal_(-1e9)
    r1 = recall_at_k(sim, labels_all, k=1)
    mapk = mean_average_precision_at_k(sim, labels_all, k=eval_k)

    return r1, mapk, embeddings.size(0), labels_all.unique().numel()


# --------------------------------------------------------------------------- #
# Training
def train(
    data_root: str,
    hf_id: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    embed_dim: int,
    max_num_patches: Optional[int],
    aug_strength: float,
    frozen_epochs: int,
    partial_epochs: int,
    partial_backbone_layers: int,
    partial_backbone_lr: float,
    full_backbone_lr: float,
    val_split: float,
    eval_k: int,
    bf16: bool,
    seed: int,
    num_workers: int,
    grad_clip: float,
    temperature: float,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dataset = ImageFolderFlat(data_root, transform=None)
    num_classes = len(base_dataset.class_to_idx)
    # train_idx, val_idx = build_train_val_indices(base_dataset, val_split, seed)
    train_idx, val_idx = build_train_val_indices_by_class(base_dataset, num_val_classes=3000, seed=seed)
    print(f"Dataset: {len(base_dataset.samples)} images | {num_classes} classes")
    print(f"Train indices: {len(train_idx)} | Val (multi-class only): {len(val_idx)}")

    train_loader = DataLoader(
        Subset(base_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pil_collate,
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pil_collate,
    )

    model = Siglip2EmbeddingModel(hf_id, embed_dim=embed_dim, bf16=bf16).to(device)
    current_stage = None
    opt: Optional[torch.optim.Optimizer] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    trainable_params: List[nn.Parameter] = []

    best_map = 0.0
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frozen_epochs = max(0, frozen_epochs)
    partial_epochs = max(0, partial_epochs)
    partial_backbone_layers = max(0, partial_backbone_layers)
    stage1_end = frozen_epochs
    stage2_end = stage1_end + partial_epochs

    for epoch in range(1, epochs + 1):
        if epoch <= stage1_end:
            stage = "frozen"
        elif epoch <= stage2_end:
            stage = "partial"
        else:
            stage = "full"

        if stage != current_stage:
            if stage == "frozen":
                freeze_backbone(model)
                opt = build_optimizer(model, head_lr=lr, backbone_lr=partial_backbone_lr, weight_decay=weight_decay)
            elif stage == "partial":
                unfreeze_backbone_layers(model, partial_backbone_layers)
                opt = build_optimizer(model, head_lr=lr, backbone_lr=partial_backbone_lr, weight_decay=weight_decay)
            else:
                unfreeze_backbone(model)
                opt = build_optimizer(model, head_lr=lr, backbone_lr=full_backbone_lr, weight_decay=weight_decay)

            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not bf16))
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            current_stage = stage
            total_elems = sum(p.numel() for p in trainable_params)
            print(f"[stage] epoch {epoch} -> {stage} fine-tuning "
                  f"(trainable tensors: {len(trainable_params)} | scalars: {total_elems:,})")

        assert opt is not None and scaler is not None, "Optimizer and scaler must be initialised."

        model.train()
        running = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            labels = labels.to(device)

            autocast_dtype = torch.bfloat16 if bf16 else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(bf16 or scaler.is_enabled())):
                embeddings = model.encode(images, max_num_patches)
                try:
                    loss = info_nce_loss(embeddings, labels, temperature=temperature)
                except ValueError:
                    continue  # skip batches without positive pairs

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                opt.step()
            opt.zero_grad(set_to_none=True)

            running += loss.item()
            if step % 25 == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} | loss {running / step:.4f}")

        r1, mapk, n_imgs, n_cls = evaluate(model, val_loader, max_num_patches, device, eval_k)
        print(f"[eval] epoch {epoch} | val images {n_imgs} | classes {n_cls} | "
              f"R@1 {r1:.4f} | mAP@{eval_k} {mapk:.4f}")

        if mapk > best_map:
            best_map = mapk
            ckpt_path = Path(out_dir) / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_to_idx": base_dataset.class_to_idx,
                    "config": {
                        "hf_id": hf_id,
                        "embed_dim": embed_dim,
                        "max_num_patches": max_num_patches,
                        "aug_strength": aug_strength,
                        "temperature": temperature,
                    },
                },
                ckpt_path,
            )
            print(f"  saved best checkpoint -> {ckpt_path}")

    print("Training finished.")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.environ.get("DATA_ROOT", "./mem_22k_aug"))
    parser.add_argument("--hf_id", type=str, default="google/siglip2-so400m-patch16-naflex")
    parser.add_argument("--out_dir", type=str, default="./siglip2_arcface_books")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--max_num_patches", type=int, default=256)
    parser.add_argument("--aug_strength", type=float, default=1.0)
    parser.add_argument("--frozen_epochs", type=int, default=2,
                        help="Number of initial epochs with the backbone fully frozen.")
    parser.add_argument("--partial_epochs", type=int, default=5,
                        help="Epochs spent training with only the last N backbone layers unfrozen.")
    parser.add_argument("--partial_backbone_layers", type=int, default=4,
                        help="How many of the final backbone blocks to unfreeze during the partial stage (0 = full).")
    parser.add_argument("--partial_backbone_lr", type=float, default=2e-5,
                        help="Learning rate for backbone params during the partial stage.")
    parser.add_argument("--full_backbone_lr", type=float, default=1e-5,
                        help="Learning rate for backbone params once fully unfrozen.")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--eval_k", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="Clip gradient norm to this value (<=0 disables clipping).")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for InfoNCE contrastive loss.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_root=args.data_root,
        hf_id=args.hf_id,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        max_num_patches=args.max_num_patches,
        aug_strength=args.aug_strength,
        frozen_epochs=args.frozen_epochs,
        partial_epochs=args.partial_epochs,
        partial_backbone_layers=args.partial_backbone_layers,
        partial_backbone_lr=args.partial_backbone_lr,
        full_backbone_lr=args.full_backbone_lr,
        val_split=args.val_split,
        eval_k=args.eval_k,
        bf16=args.bf16,
        seed=args.seed,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
