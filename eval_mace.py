#!/usr/bin/env python3

import argparse
import os
import pandas as pd

from evaluate import (
    Method,
    CLIPScorer,
    LPIPSScorer,
    ResNetScorer,
    DINOScorer,
    score_method,
)

parser = argparse.ArgumentParser()

parser.add_argument("--artist", required=True)
parser.add_argument("--prompts_csv", required=True)
parser.add_argument("--baseline_dir", required=True)
parser.add_argument("--mace_dir", required=True)
parser.add_argument("--expected_pairs", type=int, default=500)

args = parser.parse_args()

cfg = {
    "name": args.artist,
    "csv": args.prompts_csv,
    "baseline": args.baseline_dir,
    "mace": args.mace_dir,
}

df = pd.read_csv(args.prompts_csv)[["case_number", "prompt"]]

clip = CLIPScorer()
lpips = LPIPSScorer()
resnet = ResNetScorer()
dino = DINOScorer()

mace = Method(
    key="mace",
    label="MACE",
    path_key="mace",
)

row = score_method(
    cfg=cfg,
    df=df,
    method=mace,
    clip=clip,
    lpips_scorer=lpips,
    resnet=resnet,
    dino=dino,
    fid_method_keys=set(),          # disable FID
    disable_fid_fallback=True,
    expected_pairs=args.expected_pairs,
    artist_fid_dirs=False,
)

print("\n========== MACE Evaluation ==========\n")

metrics = [
    ("clip_image_similarity", "CLIP image similarity"),
    ("style_target_rate", "Style target rate"),
    ("style_drop", "Style drop"),
    ("lpips", "LPIPS"),
    ("dino_similarity", "DINO similarity"),
]

for key, name in metrics:
    print(f"{name:25s}: {row[key]:.6f}")

print("\n=====================================")