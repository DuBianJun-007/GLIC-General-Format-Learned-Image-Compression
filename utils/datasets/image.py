from typing import List, Tuple, Union
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import SimpleITK as sitk
from torch.utils.data import Dataset
import pickle


class ImageFolder(Dataset):
    def __init__(self, root: str, patch_size: Union[int, None] = None, split: str = "train"):
        
        self.patch_size = patch_size
        splitdir = Path(root) / split
        self.cache_file = Path(root) / f'{split}.cache'

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = self._load_or_cache_samples(splitdir)

    def _load_or_cache_samples(self, splitdir: Path) -> List[Path]:
        if self.cache_file.is_file():
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)

        samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        with open(self.cache_file, 'wb') as f:
            pickle.dump(samples, f)

        return samples

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            ndarr = read_data_to_numpy(self.samples[index])
            data_array_normalized, original_shape = preprocess_image(ndarr, self.patch_size)
            return data_array_normalized
        except Exception as e:
            print(f"An error occurred: {e}. Retrying with a new image.")
            return self.__getitem__(np.random.randint(0, self.__len__()))

    def __len__(self) -> int:
        return len(self.samples)


def preprocess_image(ndarr: np.ndarray, patch_size: Union[int, None], mode: str = "auto") -> Tuple[
    torch.Tensor, Tuple[int, int]]:
    data_array_normalized, _, _, original_shape = ndarr_2_2darr(ndarr, mode)
    if patch_size:
        data_array_normalized = random_crop_pad(data_array_normalized, patch_size)
    data_array_normalized = torch.from_numpy(data_array_normalized).unsqueeze(0).float()
    return data_array_normalized, original_shape


def random_crop_pad(img, output_size):
    # This function will perform a random crop on a numpy array
    # and pad the image if needed
    assert len(output_size) == 2
    new_h, new_w = output_size

    # Padding if needed
    if img.shape[0] < new_h or img.shape[1] < new_w:
        padding_h = max(0, new_h - img.shape[0])
        padding_w = max(0, new_w - img.shape[1])
        img = np.pad(img, ((0, padding_h), (0, padding_w)), mode='edge')

    # Crop
    h, w = img.shape[:2]
    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    img = img[top: top + new_h, left: left + new_w]

    return img


def read_data_to_numpy(inputPath: Path) -> np.ndarray:
    extension = os.path.splitext(inputPath)[1].lower()

    if extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        with Image.open(inputPath) as img:
            image_array = np.array(img)
            if image_array.ndim == 2:
                image_array = np.expand_dims(image_array, axis=0)
            else:
                image_array = np.transpose(image_array, (2, 0, 1))
        return image_array
    elif extension == '.mhd':
        itkimage = sitk.ReadImage(str(inputPath))
        return np.array(sitk.GetArrayFromImage(itkimage))
    else:
        raise ValueError(f"The file '{inputPath}' is not a recognized image file.")


def ndarr_2_2darr(ndarr: np.ndarray, mode: str = "auto") -> Tuple[np.ndarray, float, float, Tuple[int, int]]:
    flattened, original_shape = flatten_to_2d(ndarr, mode)
    flattened = np.squeeze(flattened, axis=0)
    min_value, max_value = np.min(flattened), np.max(flattened)
    if max_value == min_value:
        data_array_normalized = flattened / 255
    else:
        data_array_normalized = (flattened - min_value) / (max_value - min_value)
    return data_array_normalized, min_value, max_value, original_shape


def find_closest_factors(n: int) -> Tuple[int, int]:
    if n < 2:  # Handle edge cases
        return None, None
    factor1 = int(n ** 0.5)
    while n % factor1 != 0:
        factor1 -= 1
    if factor1 == 1:  # n is a prime number
        return None, None
    factor2 = n // factor1
    return factor1, factor2


def flatten_to_2d(array: np.ndarray, mode: str = 'auto') -> Tuple[np.ndarray, Tuple[int, int, int]]:
    C, H, W = array.shape
    original_shape = array.shape

    if mode == 'auto':
        C1, C2 = find_closest_factors(C)
        if C1 is None:  # C is a prime number
            mode = 'CHW'
        else:
            mode = 'C1HC2W'

    if mode == 'C1HC2W':
        C1, C2 = find_closest_factors(C)
        if C1 is None:  # C is a prime number
            mode = 'CHW'

    if mode == 'CHW':
        flattened = array.reshape(1, C * H, W)
    elif mode == 'HCW':
        flattened = array.transpose(1, 0, 2).reshape(1, H, C * W)
    elif mode == 'C1HC2W':
        reshaped = array.reshape(C1, C2, H, W).transpose(0, 2, 1, 3)
        flattened = reshaped.reshape(1, C1 * H, C2 * W)
    else:
        raise ValueError("Invalid mode.")

    return flattened, original_shape


def unflatten_to_nd(array: np.ndarray, original_shape: Tuple[int, int, int], mode: str = 'auto') -> np.ndarray:
    C, H, W = original_shape

    if mode == 'auto':
        C1, C2 = find_closest_factors(C)
        if C1 is None:  # C is a prime number
            mode = 'CHW'
        else:
            mode = 'C1HC2W'

    if mode == 'CHW':
        unflattened = array.reshape(C, H, W)
    elif mode == 'HCW':
        unflattened = array.reshape(H, C, W).transpose(1, 0, 2)
    elif mode == 'C1HC2W':
        reshaped = array.reshape(C1, H, C2, W)
        unflattened = reshaped.transpose(0, 2, 1, 3).reshape(C, H, W)
    else:
        raise ValueError("Invalid mode.")

    return unflattened
