import os
import numpy as np
import cv2
import sys
import pandas as pd

sys.path.append(os.path.abspath('..'))


def csv_group_by_image(csv_path: str, drop_half_nans: bool = True) -> pd.DataFrame.groupby:
    """Groups rows from a CSV file by image using Pandas."""
    df = pd.read_csv(csv_path, encoding='utf-8')
    if drop_half_nans:
        nan_rows = df[df.iloc[:, 1].isna()]
        non_nan_rows = df[df.iloc[:, 1].notna()]
        reduced_nan_rows = nan_rows.sample(frac=0.5) if not nan_rows.empty else pd.DataFrame()
        df = pd.concat([non_nan_rows, reduced_nan_rows])
    return df.groupby(df.columns[0]).agg(list)


def encode_rle(mask: np.ndarray) -> str:
    """Encodes a binary mask to a run-length-encoded string."""
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle(rle_string: str, img_shape: tuple[int] = (768, 768)) -> np.ndarray:
    """Decodes a run-length-encoded string to a binary mask."""
    s = rle_string.split()
    starts, lengths = [int(x) for x in s[0::2]], [int(x) for x in s[1::2]]
    starts = [x-1 for x in starts]
    ends = [x+y for x, y in zip(starts, lengths)]
    mask = np.zeros(img_shape[0] * img_shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(img_shape).T


def add_mask_to_image(img: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Adds a mask to an image as an overlay."""
    mask_converted = cv2.convertScaleAbs(mask, alpha=255.0)
    mask_colored = cv2.cvtColor(mask_converted, cv2.COLOR_GRAY2BGR)
    mask_colored[mask_converted > 0] = color
    overlay = cv2.addWeighted(src1=img, alpha=1, src2=mask_colored, beta=0.4, gamma=0)
    return overlay


def color_image_mask(mask: np.ndarray) -> np.ndarray:
    """Returns colored mask."""
    mask_2d = mask[:, :, 0]
    mask_colored = np.zeros((*mask_2d.shape, 3), dtype=np.uint8)
    mask_colored[mask_2d > 0] = [0, 255, 0]
    mask_colored[mask_2d == 0] = [0, 0, 255]
    
    return mask_colored
