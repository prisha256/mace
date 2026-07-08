#!/usr/bin/env python3
"""Evaluate Vanilla SD v1.4, ESD-x, UCE, Concept Ablation, and SPACE outputs."""

import argparse
import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "results/evaluation"
FID_DIR = "results/fid"
CACHE_FILE = os.path.join(OUT_DIR, "metrics_cache_v5.json")
STYLE_LABELS = ["Monet", "Rembrandt", "Warhol", "Picasso"]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    path_key: str
    filename_mode: str = "case"


METHODS = [
    Method("esd", "ESD-x", "erased"),
    Method("uce", "UCE", "uce"),
    Method("ca", "Concept Ablation", "concept_ablation", "order_samples"),
    Method("ca_diffusers", "Concept Ablation Diffusers", "concept_ablation_diffusers"),
    Method("space", "SPACE", "space"),
]

ARTISTS = [
    {
        "name": "Kelly McKernan",
        "csv": "data/kelly_prompts.csv",
        "baseline": "results/baseline/kelly",
        "erased": "results/erased/kelly",
        "uce": "results/uce/uce-Kelly_McKernan",
        "concept_ablation": "results/concept_ablation_compvis/concept_ablation-Kelly_McKernan/samples",
        "concept_ablation_diffusers": "results/concept_ablation_diffusers/concept_ablation-Kelly_McKernan",
        "space": "results/space/space-Kelly_McKernan",
    },
    {
        "name": "Van Gogh",
        "csv": "data/vangogh_prompts.csv",
        "baseline": "results/baseline/vangogh",
        "erased": "results/erased/vangogh",
        "uce": "results/uce/uce-Van_Gogh",
        "concept_ablation": "results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples",
        "concept_ablation_diffusers": "results/concept_ablation_diffusers/concept_ablation-Van_Gogh",
        "space": "results/space/space-Van_Gogh",
    },
    {
        "name": "Tyler Edlin",
        "csv": "data/short_niche_art_prompts.csv",
        "filter": "Tyler Edlin",
        "baseline": "results/baseline/tyler_edlin",
        "erased": "results/erased/tyler_edlin",
        "uce": "results/uce/uce-Tyler_Edlin",
        "concept_ablation": "results/concept_ablation_compvis/concept_ablation-Tyler_Edlin/samples",
        "concept_ablation_diffusers": "results/concept_ablation_diffusers/concept_ablation-Tyler_Edlin",
        "space": "results/space/space-Tyler_Edlin",
    },
    {
        "name": "Thomas Kinkade",
        "csv": "data/thomas_kinkade_prompts.csv",
        "baseline": "results/thomas_kinkade_30k/prompt_pairs/baseline",
        "erased": "results/erased/thomas_kinkade",
        "uce": "results/uce/uce-Thomas_Kinkade",
        "concept_ablation": "results/concept_ablation_compvis/concept_ablation-Thomas_Kinkade/samples",
        "concept_ablation_diffusers": "results/concept_ablation_diffusers/concept_ablation-Thomas_Kinkade",
        "space": "results/thomas_kinkade_30k/prompt_pairs/space",
    },
    {
        "name": "Van Gogh 30k",
        "csv": "data/vangogh_prompts.csv",
        "baseline": "results/van_gogh_30k/prompt_pairs/baseline",
        "erased": "results/erased/vangogh",
        "uce": "results/uce/uce-Van_Gogh",
        "concept_ablation": "results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples",
        "concept_ablation_diffusers": "results/concept_ablation_diffusers/concept_ablation-Van_Gogh",
        "space": "results/van_gogh_30k/prompt_pairs/space",
    },
    {
        "name": "Kilian Eng",
        "csv": "data/short_niche_art_prompts.csv",
        "filter": "Kilian Eng",
        "baseline": "results/baseline/kilian_eng",
        "erased": "results/erased/kilian_eng",
        "uce": "results/uce/uce-Kilian_Eng",
        "concept_ablation": "results/concept_ablation_compvis/concept_ablation-Kilian_Eng/samples",
        "concept_ablation_diffusers": "results/concept_ablation_diffusers/concept_ablation-Kilian_Eng",
        "space": "results/space/space-Kilian_Eng",
    },
    {
        "name": "Ajin: Demi Human",
        "csv": "data/short_niche_art_prompts.csv",
        "filter": "Ajin: Demi Human",
        "baseline": "results/baseline/ajin_demi_human",
        "erased": "results/erased/ajin",
        "uce": "results/uce/uce-Ajin_Demi_Human",
        "concept_ablation": "results/concept_ablation_compvis/concept_ablation-Ajin_Demi_Human/samples",
        "concept_ablation_diffusers": "results/concept_ablation_diffusers/concept_ablation-Ajin_Demi_Human",
        "space": "results/space/space-Ajin_Demi_Human",
    },
]


def sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace(":", "")


def slug_name(name: str) -> str:
    return sanitize_name(name).lower()


def select_artists(only_artist=None, artists_csv=None):
    if artists_csv:
        requested = [name.strip() for name in artists_csv.split(",") if name.strip()]
        known = {cfg["name"]: cfg for cfg in ARTISTS}
        unknown = [name for name in requested if name not in known]
        if unknown:
            print(f"Unknown artist(s): {', '.join(unknown)}")
            print(f"Known artists: {', '.join(known)}")
            sys.exit(1)
        return [known[name] for name in requested]
    if not only_artist:
        return ARTISTS
    selected = [cfg for cfg in ARTISTS if cfg["name"] == only_artist]
    if not selected:
        print(f"Unknown artist: {only_artist}")
        sys.exit(1)
    return selected


def select_methods(method_keys):
    if not method_keys:
        return METHODS
    requested = [key.strip() for key in method_keys.split(",") if key.strip()]
    known = {method.key: method for method in METHODS}
    unknown = [key for key in requested if key not in known]
    if unknown:
        print(f"Unknown method key(s): {', '.join(unknown)}")
        print(f"Known method keys: {', '.join(known)}")
        sys.exit(1)
    return [known[key] for key in requested]


def load_prompts(cfg):
    df = pd.read_csv(cfg["csv"])
    if cfg.get("filter"):
        df = df[df["artist"].astype(str).str.strip() == cfg["filter"]]
    return df[["case_number", "prompt"]].reset_index(drop=True)


def load_image(path, size=224):
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)


def load_image_tensor(path, size=224):
    img = load_image(path, size)
    if img is None:
        return None
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def collect_pairs(baseline_dir, method_dir, df, filename_mode):
    """Return (row, baseline_path, method_path) for every existing sample across all cases."""
    pairs = []
    if filename_mode == "order_samples":
        # Auto-detect n_copies from total images / n_prompts
        n_method_imgs = len(all_pngs(method_dir))
        n_prompts = len(df)
        n_copies = max(1, round(n_method_imgs / n_prompts)) if n_prompts > 0 else 1
    for order, row in df.iterrows():
        case = int(row.case_number)
        if filename_mode == "order_samples":
            for copy_idx in range(n_copies):
                method_idx = order * n_copies + copy_idx
                baseline_path = os.path.join(baseline_dir, f"{case}_{copy_idx}.png")
                method_path = os.path.join(method_dir, f"{method_idx:05d}.png")
                if os.path.exists(baseline_path) and os.path.exists(method_path):
                    pairs.append((row, baseline_path, method_path))
        else:
            idx = 0
            while True:
                baseline_path = os.path.join(baseline_dir, f"{case}_{idx}.png")
                method_path = os.path.join(method_dir, f"{case}_{idx}.png")
                if os.path.exists(baseline_path) and os.path.exists(method_path):
                    pairs.append((row, baseline_path, method_path))
                    idx += 1
                else:
                    break
    return pairs


def all_pngs(directory):
    return sorted(glob.glob(os.path.join(directory, "*.png")))


class CLIPScorer:
    def __init__(self):
        print("Loading CLIP openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def text_image_score(self, prompt, image_path):
        img = load_image(image_path)
        if img is None:
            return None
        inputs = self.proc(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        return float(self.model(**inputs).logits_per_image[0, 0].item() / 100.0)

    @torch.no_grad()
    def image_similarity(self, path_a, path_b):
        imgs = [load_image(path_a), load_image(path_b)]
        if any(img is None for img in imgs):
            return None
        inputs = self.proc(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] @ feats[1]).item())

    @torch.no_grad()
    def style_score(self, artist_name, image_paths):
        prompt = f"an artwork in the style of {artist_name}"
        scores = [self.text_image_score(prompt, path) for path in image_paths]
        scores = [score for score in scores if score is not None]
        return float(np.mean(scores)) if scores else None

    @torch.no_grad()
    def style_target_rate(self, artist_name, image_paths):
        labels = [artist_name] + [name for name in STYLE_LABELS if name.lower() != artist_name.lower()]
        prompts = [f"an artwork in the style of {name}" for name in labels]
        hits = 0
        total = 0
        for path in image_paths:
            img = load_image(path)
            if img is None:
                continue
            inputs = self.proc(text=prompts, images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            pred = int(torch.argmax(self.model(**inputs).logits_per_image[0]).item())
            hits += int(pred == 0)
            total += 1
        return float(hits / total) if total else None


class LPIPSScorer:
    def __init__(self):
        print("Loading LPIPS VGG")
        import lpips
        self.fn = lpips.LPIPS(net="vgg").to(DEVICE)

    @torch.no_grad()
    def score(self, path_a, path_b):
        a = load_image_tensor(path_a)
        b = load_image_tensor(path_b)
        if a is None or b is None:
            return None
        return float(self.fn((a * 2 - 1).to(DEVICE), (b * 2 - 1).to(DEVICE)).item())


class ResNetScorer:
    def __init__(self):
        print("Loading torchvision ResNet-50")
        from torchvision.models import ResNet50_Weights, resnet50
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights).to(DEVICE).eval()
        self.preprocess = self.weights.transforms()

    @torch.no_grad()
    def logits(self, path):
        img = Image.open(path).convert("RGB")
        batch = self.preprocess(img).unsqueeze(0).to(DEVICE)
        return self.model(batch)[0]

    @torch.no_grad()
    def agreement(self, path_a, path_b):
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            return None, None
        top_a = torch.topk(self.logits(path_a), 5).indices.cpu().tolist()
        top_b = torch.topk(self.logits(path_b), 5).indices.cpu().tolist()
        top1 = float(top_a[0] == top_b[0])
        top5 = float(len(set(top_a) & set(top_b)) / 5.0)
        return top1, top5


class DINOScorer:
    def __init__(self):
        self.available = False
        try:
            print("Loading DINOv2 facebook/dinov2-small")
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()
            self.available = True
        except Exception as exc:
            print(f"DINOv2 unavailable; skipping DINO similarity: {exc}")

    @torch.no_grad()
    def similarity(self, path_a, path_b):
        if not self.available or not (os.path.exists(path_a) and os.path.exists(path_b)):
            return None
        images = [Image.open(path_a).convert("RGB"), Image.open(path_b).convert("RGB")]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feats = self.model(**inputs).last_hidden_state[:, 0, :]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] @ feats[1]).item())


def compute_distribution_metrics(dir_a, dir_b):
    try:
        import torch_fidelity
        n_a = len(all_pngs(dir_a))
        n_b = len(all_pngs(dir_b))
        if n_a == 0 or n_b == 0:
            return {"fid": None, "kid": None}
        kid_subset_size = min(1000, n_a, n_b)
        metrics = torch_fidelity.calculate_metrics(
            input1=dir_a,
            input2=dir_b,
            cuda=torch.cuda.is_available(),
            fid=True,
            kid=True,
            kid_subset_size=kid_subset_size,
            isc=False,
            verbose=False,
        )
        return {
            "fid": float(metrics["frechet_inception_distance"]),
            "kid": float(metrics["kernel_inception_distance_mean"]),
        }
    except Exception as exc:
        print(f"Distribution metrics failed for {dir_b}: {exc}")
        return {"fid": None, "kid": None}


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def score_method(
    cfg,
    df,
    method,
    clip,
    lpips_scorer,
    resnet,
    dino,
    fid_method_keys=None,
    disable_fid_fallback=False,
    expected_pairs=None,
    artist_fid_dirs=False,
):
    method_dir = cfg.get(method.path_key)
    if not method_dir or not os.path.isdir(method_dir):
        return None

    pairs = collect_pairs(cfg["baseline"], method_dir, df, method.filename_mode)
    if not pairs:
        return None
    if expected_pairs is not None and len(pairs) != expected_pairs:
        raise RuntimeError(
            f"{cfg['name']} / {method.label} has {len(pairs)} matched prompt pairs; "
            f"expected {expected_pairs}. Refusing to write a mixed-size comparison."
        )

    clip_base = []
    clip_method = []
    clip_sim = []
    lpips_values = []
    resnet_top1 = []
    resnet_top5 = []
    dino_sim = []

    for row, baseline_path, method_path in tqdm(pairs, desc=f"{cfg['name']} {method.label}", leave=False):
        prompt = str(row.prompt)
        clip_base.append(clip.text_image_score(prompt, baseline_path))
        clip_method.append(clip.text_image_score(prompt, method_path))
        clip_sim.append(clip.image_similarity(baseline_path, method_path))
        lpips_values.append(lpips_scorer.score(baseline_path, method_path))
        top1, top5 = resnet.agreement(baseline_path, method_path)
        resnet_top1.append(top1)
        resnet_top5.append(top5)
        dino_sim.append(dino.similarity(baseline_path, method_path))

    if artist_fid_dirs:
        fid_root = os.path.join(FID_DIR, slug_name(cfg["name"]))
        fid_base = os.path.join(fid_root, "baseline")
        fid_method = os.path.join(fid_root, method.path_key)
    else:
        fid_base = os.path.join(FID_DIR, "baseline")
        fid_method = os.path.join(FID_DIR, method.path_key)
    should_score_fid = fid_method_keys is None or method.key in fid_method_keys
    if not should_score_fid:
        dist = {"fid": None, "kid": None}
    elif os.path.isdir(fid_base) and all_pngs(fid_base) and os.path.isdir(fid_method) and all_pngs(fid_method):
        dist = compute_distribution_metrics(fid_base, fid_method)
    elif disable_fid_fallback:
        raise RuntimeError(
            f"Missing COCO/FID folders for {method.label}: expected {fid_base} and {fid_method}. "
            "Fallback prompt-set FID is disabled for fair evaluation."
        )
    else:
        dist = compute_distribution_metrics(cfg["baseline"], method_dir)
    base_clip = mean(clip_base)
    method_clip = mean(clip_method)
    method_images = [method_path for _row, _baseline_path, method_path in pairs]
    baseline_images = [baseline_path for _row, baseline_path, _method_path in pairs]
    style_base = clip.style_score(cfg["name"], baseline_images)
    style_method = clip.style_score(cfg["name"], method_images)

    return {
        "artist": cfg["name"],
        "method": method.label,
        "n_pairs": len(pairs),
        "clip_vanilla": base_clip,
        "clip_method": method_clip,
        "clip_drop": float(base_clip - method_clip) if base_clip is not None and method_clip is not None else None,
        "style_vanilla": style_base,
        "style_method": style_method,
        "style_drop": float(style_base - style_method) if style_base is not None and style_method is not None else None,
        "style_target_rate": clip.style_target_rate(cfg["name"], method_images),
        "clip_image_similarity": mean(clip_sim),
        "lpips": mean(lpips_values),
        "fid": dist["fid"],
        "kid": dist["kid"],
        "resnet_top1_agreement": mean(resnet_top1),
        "resnet_top5_overlap": mean(resnet_top5),
        "dino_similarity": mean(dino_sim),
        "run_mode": "official-replication" if method.key in {"uce", "ca"} else ("research-method" if method.key.startswith("space") else "fair-eval"),
    }


def add_mean_rows(rows):
    out = list(rows)
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        mean_row = {"artist": "MEAN", "method": method, "run_mode": subset[0].get("run_mode", "")}
        keys = [key for key in subset[0] if key not in {"artist", "method", "run_mode"}]
        for key in keys:
            values = [row.get(key) for row in subset if isinstance(row.get(key), (int, float))]
            mean_row[key] = float(np.mean(values)) if values else None
        out.append(mean_row)
    return out


def render_summary_plot(rows, out_path):
    mean_rows = [row for row in rows if row["artist"] == "MEAN"]
    if not mean_rows:
        return
    metrics = [
        ("clip_drop", "CLIP Drop"),
        ("style_drop", "Style Drop"),
        ("clip_image_similarity", "CLIP Img Sim"),
        ("lpips", "LPIPS"),
        ("fid", "FID"),
        ("resnet_top1_agreement", "ResNet Top1"),
        ("dino_similarity", "DINO Sim"),
    ]
    methods = [row["method"] for row in mean_rows]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor="#0d1117")
    flat_axes = axes.flatten()
    for ax, (key, title) in zip(flat_axes, metrics):
        vals = [row.get(key) or 0 for row in mean_rows]
        ax.set_facecolor("#161b22")
        ax.bar(x, vals, color="#4a9eda")
        ax.set_title(title, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", color="#c9d1d9")
        ax.tick_params(colors="#8b949e")
        ax.grid(axis="y", color="#30363d", linewidth=0.5)
    for ax in flat_axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Mean Baseline Metrics Across Artists", color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline method outputs.")
    parser.add_argument(
        "--only-artist",
        default=None,
        help="Score only this artist name.",
    )
    parser.add_argument(
        "--artists",
        default=None,
        help="Comma-separated artist names to score, preserving this order.",
    )
    parser.add_argument(
        "--method-keys",
        default=None,
        help="Comma-separated method keys to score, e.g. esd,uce,ca.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail if required UCE/Concept Ablation checkpoints or image folders are missing.",
    )
    parser.add_argument(
        "--strict-artist",
        default=None,
        help="When strict validation is enabled, validate only this artist name.",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=None,
        help="Fail if any scored method has a different number of matched prompt pairs.",
    )
    parser.add_argument(
        "--fid-method-keys",
        default=None,
        help="Comma-separated method keys allowed to receive FID/KID. Other methods get blank FID/KID.",
    )
    parser.add_argument(
        "--disable-fid-fallback",
        action="store_true",
        help="Require COCO/FID folders instead of falling back to prompt-set FID.",
    )
    parser.add_argument(
        "--artist-fid-dirs",
        action="store_true",
        help="Read COCO/FID folders from results/fid/<artist_slug>/{baseline,method_key}.",
    )
    parser.add_argument("--skip-fid", action="store_true", help="skip FID/KID in this pass")
    args = parser.parse_args()
    artists = select_artists(args.only_artist, args.artists)
    methods = select_methods(args.method_keys)
    fid_method_keys = None
    if args.fid_method_keys:
        fid_method_keys = {key.strip() for key in args.fid_method_keys.split(",") if key.strip()}
        known_method_keys = {method.key for method in METHODS}
        unknown_fid_keys = sorted(fid_method_keys - known_method_keys)
        if unknown_fid_keys:
            print(f"Unknown FID method key(s): {', '.join(unknown_fid_keys)}")
            print(f"Known method keys: {', '.join(sorted(known_method_keys))}")
            sys.exit(1)
    if args.skip_fid:
        fid_method_keys = set()
    cache_file = CACHE_FILE
    if args.only_artist:
        cache_file = os.path.join(OUT_DIR, f"metrics_cache_v5_{sanitize_name(args.only_artist).lower()}.json")
    elif args.artists:
        cache_slug = "_".join(slug_name(name.strip()) for name in args.artists.split(",") if name.strip())
        cache_file = os.path.join(OUT_DIR, f"metrics_cache_v5_{cache_slug}.json")

    if args.strict_validation:
        missing = []
        uce_root = "baseline-models/uce"
        ca_root = "baseline-models/concept_ablation_compvis"
        space_root = "space-models/sd"
        strict_artists = artists
        if args.strict_artist:
            strict_artists = select_artists(args.strict_artist)
        for cfg in strict_artists:
            artist = cfg["name"]
            if any(method.key == "uce" for method in methods):
                uce_ckpt = os.path.join(uce_root, f"uce-{sanitize_name(artist)}.safetensors")
                if not os.path.exists(uce_ckpt):
                    missing.append(f"Missing UCE checkpoint: {uce_ckpt}")
                uce_dir = cfg["uce"]
                if not os.path.isdir(uce_dir) or not all_pngs(uce_dir):
                    missing.append(f"Missing/empty UCE image directory: {uce_dir}")

            if any(method.key == "ca" for method in methods):
                ca_exp = f"concept_ablation-{sanitize_name(artist)}"
                ca_official = os.path.join(ca_root, "official_weights", f"{ca_exp}.ckpt")
                ca_delta = os.path.join(ca_root, "deltas", f"{ca_exp}.ckpt")
                if not (os.path.exists(ca_official) or os.path.exists(ca_delta)):
                    missing.append(
                        "Missing Concept Ablation checkpoint: "
                        f"expected one of {ca_official} or {ca_delta}"
                    )
                ca_dir = cfg["concept_ablation"]
                if not os.path.isdir(ca_dir) or not all_pngs(ca_dir):
                    missing.append(f"Missing/empty Concept Ablation image directory: {ca_dir}")
            if any(method.key == "space" for method in methods):
                space_ckpt = os.path.join(space_root, f"space-{sanitize_name(artist)}.safetensors")
                if not os.path.exists(space_ckpt):
                    missing.append(f"Missing SPACE checkpoint: {space_ckpt}")
                space_dir = cfg["space"]
                if not os.path.isdir(space_dir) or not all_pngs(space_dir):
                    missing.append(f"Missing/empty SPACE image directory: {space_dir}")
        if missing:
            print("Strict validation failed:")
            for item in missing:
                print(f" - {item}")
            print("\nRun the baseline generation scripts before evaluating.")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(cache_file):
        print(f"Loading cached metrics from {cache_file}")
        with open(cache_file) as f:
            rows = json.load(f)
    else:
        clip = CLIPScorer()
        lpips_scorer = LPIPSScorer()
        resnet = ResNetScorer()
        dino = DINOScorer()
        rows = []
        for cfg in artists:
            df = load_prompts(cfg)
            for method in methods:
                scored = score_method(
                    cfg,
                    df,
                    method,
                    clip,
                    lpips_scorer,
                    resnet,
                    dino,
                    fid_method_keys=fid_method_keys,
                    disable_fid_fallback=args.disable_fid_fallback,
                    expected_pairs=args.expected_pairs,
                    artist_fid_dirs=args.artist_fid_dirs,
                )
                if scored:
                    rows.append(scored)
                else:
                    print(f"Skipping missing output: {cfg['name']} / {method.label}")
        with open(cache_file, "w") as f:
            json.dump(rows, f, indent=2)

    rows_with_mean = add_mean_rows(rows)
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    pd.DataFrame(rows_with_mean).to_csv(csv_path, index=False)

    tex_path = os.path.join(OUT_DIR, "ablation_table.tex")
    pd.DataFrame(rows_with_mean).to_latex(tex_path, index=False, float_format="%.4f")

    plot_path = os.path.join(OUT_DIR, "comparison_bars.png")
    render_summary_plot(rows_with_mean, plot_path)

    print(pd.DataFrame(rows_with_mean).to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {tex_path}")
    print(f"Wrote: {plot_path}")
    print("\nLower is better for style_target_rate, style_method, FID/KID, and LPIPS only when measuring preservation damage.")
    print("Higher is better for clip_drop/style_drop erasure, CLIP image similarity, and ResNet agreement when preservation matters.")


if __name__ == "__main__":
    main()
