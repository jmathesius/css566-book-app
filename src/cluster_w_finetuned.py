#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, Siglip2VisionModel

# --------------------------------------------------------------------------- #
# Configuration

# Set to None to use ALL images in the folder (recommended for full clustering run)
NUM_IMAGES: Optional[int] = None

THRESHOLD: float = 0.9
DATASET: str = "./processed_images"
OUTPUT_FOLDER: str = "clusters_finetuned"
UNCLUSTERED_FOLDER: str = "unclustered_finetuned"
CHECKPOINT_PATH: str = "./siglip2_arcface_books/best.pt"

# Valid image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# --------------------------------------------------------------------------- #
# Embedding model for finetuned (matches train_aug3.py)

class Siglip2EmbeddingModel(nn.Module):
    """
    SigLIP2 backbone followed by a projection head that outputs normalized embeddings.
    """
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
        proc_kwargs = dict(
            return_tensors="pt",
            do_resize=True,
            do_center_crop=False,
        )

        if max_num_patches is not None:
            proc_kwargs["max_num_patches"] = max_num_patches

        inputs = self.processor(images=images, **proc_kwargs)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float16

        # Use autocast only when appropriate
        use_autocast = self.use_bf16 or device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            features = self.backbone(**inputs).pooler_output

        # Projection + L2-normalization
        if features.dtype != torch.float32:
            features = features.float()

        embeddings = self.proj(features)
        return F.normalize(embeddings, dim=1)


# --------------------------------------------------------------------------- #
# Load finetuned model

def load_finetuned_model(checkpoint_path: str) -> Tuple[Siglip2EmbeddingModel, Optional[int]]:
    print("Loading finetuned model checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    hf_id = config["hf_id"]
    embed_dim = config["embed_dim"]
    max_num_patches = config.get("max_num_patches", None)

    model = Siglip2EmbeddingModel(hf_id, embed_dim=embed_dim, bf16=False)
    # strict=False for safety if keys differ slightly (e.g., from DDP)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    print("Finetuned model loaded.")
    return model, max_num_patches


# --------------------------------------------------------------------------- #
# Embedding function

def generate_embedding_finetuned(
    model: Siglip2EmbeddingModel,
    max_num_patches: Optional[int],
    image_path: str,
) -> Optional[np.ndarray]:
    """
    Generate embedding for finetuned model for a single image.
    Returns a (1, D) numpy array or None on failure.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open {image_path}: {e}")
        return None

    try:
        with torch.no_grad():
            embedding = model.encode([image], max_num_patches)
        return embedding.cpu().numpy().reshape(1, -1)
    except Exception as e:
        print(f"Could not process {image_path}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Clustering

def cluster_images(
    images: List[Tuple[int, str]],
    model: Siglip2EmbeddingModel,
    max_num_patches: Optional[int],
    model_name: str = "finetuned",
) -> List[dict]:
    """
    Cluster images using the given (finetuned) model.

    images: list of (index, filepath)
    returns: list of cluster dicts:
        {
            "avg_embedding": np.ndarray,
            "images": [(idx, filepath), ...]
        }

    Note: No max-distance restriction; any image can join any cluster.
    """
    clusters: List[dict] = []

    for idx, path in images:
        print(f"[{model_name}] Processing image {idx + 1}/{len(images)}: {path}")
        embedding = generate_embedding_finetuned(model, max_num_patches, path)
        if embedding is None:
            print("  Skipped (no embedding).")
            continue

        if not clusters:
            clusters.append({
                "avg_embedding": embedding,
                "images": [(idx, path)],
            })
            continue

        # Compute similarity to all existing clusters
        avg_embeddings = np.vstack([c["avg_embedding"] for c in clusters])
        sims = cosine_similarity(embedding, avg_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= THRESHOLD:
            # Add to best cluster
            cluster = clusters[best_idx]
            n = len(cluster["images"])
            # Update running average
            cluster["avg_embedding"] = (cluster["avg_embedding"] * n + embedding) / (n + 1)
            cluster["images"].append((idx, path))
            print(f"  Added to cluster {best_idx} (similarity: {best_score:.4f}, size now {len(cluster['images'])})")
        else:
            # Create a new cluster
            clusters.append({
                "avg_embedding": embedding,
                "images": [(idx, path)],
            })
            print(f"  Created new cluster {len(clusters) - 1} (similarity: {best_score:.4f})")

    return clusters


# --------------------------------------------------------------------------- #
# Saving & Stats

def save_clusters(
    clusters: List[dict],
    output_folder: str,
    unclustered_folder: str,
):
    """
    Save clustered and unclustered images to separate folders.
    Clusters with >1 image go into cluster_X folders.
    Singletons go into the unclustered folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(unclustered_folder, exist_ok=True)

    for i, cluster in enumerate(clusters):
        images = cluster["images"]

        if len(images) > 1:
            cluster_folder = os.path.join(output_folder, f"cluster_{i}")
            os.makedirs(cluster_folder, exist_ok=True)

            for _, img_path in images:
                filename = os.path.basename(img_path)
                dst = os.path.join(cluster_folder, filename)
                if not os.path.exists(dst):
                    shutil.copy(img_path, dst)
        else:
            # Single image -> unclustered
            _, img_path = images[0]
            filename = os.path.basename(img_path)
            dst = os.path.join(unclustered_folder, filename)
            if not os.path.exists(dst):
                shutil.copy(img_path, dst)


def print_clustering_stats(clusters: List[dict], model_name: str):
    multi_image_clusters = [c for c in clusters if len(c["images"]) > 1]
    single_image_clusters = [c for c in clusters if len(c["images"]) == 1]
    total_clustered = sum(len(c["images"]) for c in multi_image_clusters)
    total_images = total_clustered + len(single_image_clusters)

    print(f"\n{'=' * 60}")
    print(f"Statistics for {model_name}:")
    print(f"{'=' * 60}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Clusters with multiple images: {len(multi_image_clusters)}")
    print(f"Unclustered (single image clusters): {len(single_image_clusters)}")
    print(f"Total images seen: {total_images}")
    print(f"Total images in multi-image clusters: {total_clustered}")

    if multi_image_clusters:
        sizes = [len(c["images"]) for c in multi_image_clusters]
        print(f"Average multi-image cluster size: {np.mean(sizes):.2f}")
        print(f"Max cluster size: {max(sizes)}")
        print(f"Min cluster size: {min(sizes)}")
    print(f"{'=' * 60}\n")


# --------------------------------------------------------------------------- #
# Main

def main():
    print("\nStarting clustering with finetuned model...")
    print(f"Dataset folder: {DATASET}")
    print(f"Threshold: {THRESHOLD}")
    print("No max-distance restriction between images.\n")

    # Load model
    model, max_num_patches = load_finetuned_model(CHECKPOINT_PATH)

    # Collect image files
    folder = Path(DATASET)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET}")

    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]

    # Deterministic ordering (by name); adjust if you prefer mtime or something else
    files = sorted(files, key=lambda f: f.name)

    if NUM_IMAGES is not None:
        files = files[:NUM_IMAGES]

    if not files:
        raise RuntimeError(f"No valid image files found in {DATASET}")

    print(f"Found {len(files)} images to process.\n")

    # Prepare (index, full_path) list
    indexed_paths: List[Tuple[int, str]] = [
        (i, str(path)) for i, path in enumerate(files)
    ]

    # Cluster with finetuned model only
    print("=" * 60)
    print("CLUSTERING WITH FINETUNED MODEL")
    print("=" * 60)
    clusters = cluster_images(indexed_paths, model, max_num_patches, model_name="finetuned")

    # Save results
    save_clusters(clusters, OUTPUT_FOLDER, UNCLUSTERED_FOLDER)
    print_clustering_stats(clusters, "Finetuned Model")

    print(f"Cluster folders written to: {OUTPUT_FOLDER}/")
    print(f"Unclustered images written to: {UNCLUSTERED_FOLDER}/")
    print("\n" + "=" * 60)
    print("CLUSTERING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
