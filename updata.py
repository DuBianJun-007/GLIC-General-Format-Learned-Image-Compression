import argparse

from pathlib import Path
from typing import Dict

import os

import torch
from torch import Tensor

from Network_GFPC import GFPC


def rename_key(key: str) -> str:
    """Rename state_dict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def load_pretrained(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = load_pretrained(state_dict)
    return state_dict


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-update",
        action="store_true",
        default=False,
        help="Do not update the model CDFs parameters.",
    )
    parser.add_argument("--run",
                        type=str,
                        default='1001',
                        help="Path to a run_xx.")
    return parser

def main():
    args = setup_args().parse_args()
    run_path = f'checkpoint/run_{args.run}'
    filepath = os.path.join(run_path, r'best_checkpoint\checkpoint_best_loss.pth.tar')
    filepath = Path(filepath)
    if not filepath.is_file():
        raise RuntimeError(f'"{filepath}" is not a valid file.')
    state_dict = load_checkpoint(filepath)
    
    model_cls = GFPC()
    net = model_cls.from_state_dict(state_dict)
    
    if not args.no_update:
        net.update(force=True)
    state_dict = net.state_dict()
    
    save_path = os.path.join(run_path, r'updataModel/updataModel.pth.tar')
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    main()
