import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd
from utils import decode_rle, csv_group_by_image


def data_generator(csv_path: str, images_dir: str, split='train', split_ratio=0.8):
    """Generator that yields images and their masks for specified dataset split using Pandas."""
    df = csv_group_by_image(csv_path)
    total_images = len(df)
    split_index = int(total_images * split_ratio)
    
    if split == 'train':
        df_split = df.iloc[:split_index]
    elif split == 'val':
        df_split = df.iloc[split_index:]
    
    for image_name, row in df_split.iterrows():
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        mask = np.zeros((768, 768), dtype=np.uint8)

        mask_rles = row['EncodedPixels']
        for mask_rle in mask_rles:
            if pd.isnull(mask_rle):
                continue
            mask_rle = str(mask_rle)
            m = decode_rle(mask_rle)
            mask += m

        mask = np.clip(mask, 0, 1)
        yield img, mask.reshape(768, 768, 1)


def create_tf_dataset(csv_path, images_dir, batch_size=32, split_ratio=0.8):
    """Creates TensorFlow datasets for training and validation with a specified split ratio."""
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(csv_path, images_dir, split='train', split_ratio=split_ratio),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=((768, 768, 3), (768, 768, 1))
    ).batch(batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(csv_path, images_dir, split='val', split_ratio=split_ratio),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=((768, 768, 3), (768, 768, 1))
    ).batch(batch_size)

    return train_dataset, val_dataset
