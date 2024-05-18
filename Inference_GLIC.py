import argparse
import json
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import numpy as np
from pytorch_msssim import ms_ssim
import torchvision
import compressai
from compressai.zoo import load_state_dict
import torch
import os
import math
import torch.nn as nn

from utils.datasets.image import read_data_to_numpy, preprocess_image, unflatten_to_nd
from Network_GLIC import GLIC


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str, mode: str = "auto") -> Tuple[
    torch.Tensor, Union[Tuple[int, int, int], Tuple[int, int, int]]]:
    ndarr = read_data_to_numpy(Path(filepath))
    data_array_normalized, original_shape = preprocess_image(ndarr, None, mode)
    data_array_normalized = data_array_normalized.unsqueeze(0)
    return data_array_normalized, original_shape


@torch.no_grad()
def inference(model, x: torch.Tensor, f: str, output_path: str, patch: int, original_shape: Tuple[int, int, int],
              level: int = 7, mode: str = "auto") -> Dict[str, Any]:
    f = os.path.normpath(f)
    path_parts = f.split(os.path.sep)
    path_parts[-3] = output_path
    out_path = os.path.sep.join(path_parts)

    print(f"Decoding image: {f}")

    # Original Padding
    h, w = x.size(2), x.size(3)
    new_h = (h + patch - 1) // patch * patch
    new_w = (w + patch - 1) // patch * patch
    pad = nn.ConstantPad2d((0, new_w - w, 0, new_h - h), 0)
    x_padded = pad(x)

    # Compression
    start = time.time()
    out_enc = model.compress(x_padded, original_shape)
    enc_time = time.time() - start

    # Decompression
    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], level)
    dec_time = time.time() - start

    # Remove Padding
    out_dec["x_hat"] = nn.functional.pad(
        out_dec["x_hat"], (0, -new_w + w, 0, -new_h + h)
    )

    # Calculate Metrics
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = 0
    for s in out_enc["strings"]:
        for j in s:
            if isinstance(j, list):
                for i in j:
                    if isinstance(i, list):
                        for k in i:
                            bpp += len(k)
                    else:
                        bpp += len(i)
            else:
                bpp += len(j)
    bpp *= 8.0 / num_pixels
    PSNR = psnr(x, out_dec["x_hat"])
    MS_SSIM = calculate_ms_ssim(x, out_dec["x_hat"])

    print(
        f'Bpp: {round(bpp, 2)}, PSNR: {round(PSNR, 2)}, MS-SSIM: {round(MS_SSIM, 2)}, MS-SSIM(dB): {round(-10 * np.log10(1 - MS_SSIM), 2)}, Encoding time: {round(enc_time, 2)}, Decoding time: {round(dec_time, 2)}')

    # Save Image
    out_dec["x_hat"] = unflatten_to_nd(out_dec["x_hat"], original_shape, mode=mode)
    dir_path = os.path.dirname(out_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torchvision.utils.save_image(out_dec["x_hat"], out_path, nrow=1)

    return {
        "psnr": PSNR,
        "MS-SSIM": MS_SSIM,
        "MS-SSIM_dB": -10 * np.log10(1 - MS_SSIM),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


def calculate_ms_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    # Add batch and channel dimensions if they're not present
    img1 = img1.unsqueeze(0) if len(img1.shape) == 2 else img1
    img2 = img2.unsqueeze(0) if len(img2.shape) == 2 else img2

    # Calculate MS-SSIM
    ms_ssim_value = ms_ssim(img1, img2, data_range=1, size_average=True)
    return ms_ssim_value.item()


def eval_model(model, file_paths: List[str], output_path: str = 'outputImages', patch: int = 64,
               level: int = 7, mode: str = "auto") -> Dict[str, float]:
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in file_paths:
        x, original_shape = read_image(f, mode)
        x = x.to(device)
        results = inference(model, x, f, output_path, patch, original_shape, level, mode)

        for k, v in results.items():
            metrics[k] += v

    num_files = len(file_paths)
    for k in metrics.keys():
        metrics[k] /= num_files

    return metrics


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--dataset",
        type=str,
        default='./dataset_val/Kodak',
        help="Dataset path."
    )
    parser.add_argument(
        "--output_path",
        default='outputImages',
        help="Result output path."
    )
    parser.add_argument(
        "-c", "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder."
    )
    parser.add_argument(
        "-p", "--path",
        dest="paths",
        type=str,
        default='./model/GLIC-0.0075.pth.tar',
        help="Model path."
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=64,
        help="Padding patch size."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=8,
        help="Level."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        help='Preprocessing format, optional: ["auto","CHW","HCW", "C1HC2W"].'
    )

    return parser


def main(argv: List[str]) -> None:
    parser = setup_args()
    args = parser.parse_args(argv)
    filepaths = sorted([os.path.join(args.dataset, filename) for filename in os.listdir(args.dataset)])
    if not filepaths:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)
    state_dict = load_state_dict(torch.load(args.paths))

    model = GLIC().from_state_dict(state_dict).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    results = defaultdict(list)
    metrics = eval_model(model, filepaths, args.output_path, args.patch, args.level - 1, args.mode)

    for k, v in metrics.items():
        results[k].append(v)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
