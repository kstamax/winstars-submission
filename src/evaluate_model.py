import argparse
import numpy as np
import os
import pandas as pd
import cv2
from model import build_model
from utils import encode_rle
from load_data import create_tf_dataset


def calculate_dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculates the Dice score of two binary numpy arrays."""
    intersection = np.logical_and(mask1, mask2).sum()
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    eps = 1e-6  # adjustment for cases where there are no ships
    dice_score = (2 * intersection + eps) / (mask1_sum + mask2_sum + eps)
    return dice_score


def predict_and_encode(directory: str, model, csv_file_path: str):
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            predicted_mask = model.predict(np.expand_dims(image, axis=0))[0]
            
            rle_encoded = encode_rle(predicted_mask > 0.1)
            
            results.append([filename, rle_encoded])
    
    df_results = pd.DataFrame(results, columns=['ImageId', 'EncodedPixels'])
    df_results.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str,
        default=os.path.join('models', 'ships_weights.keras'),
        help='Path to the model weights'
    )
    parser.add_argument(
        '--data_path', type=str, default=os.path.join('data', 'airbus-ship-detection'), help='Path to dataset'
    )
    parser.add_argument(
        '--calculate_dice', action='store_true', default=False,
        help='Get Dice score for train dataset'
    )
    parser.add_argument(
        '--kaggle_submission', action='store_true', default=False, help='Generate a Kaggle submission file'
    )
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    calculate_dice = args.calculate_dice
    kaggle_submission = args.kaggle_submission

    model = build_model()
    model.load_weights(model_path)
    model.compile()

    if kaggle_submission:
        predict_and_encode(os.path.join(data_path, 'test_v2'), model, 'submission.csv')

    if calculate_dice:
        data, _ = create_tf_dataset(
            os.path.join(data_path, 'train_ship_segmentations_v2.csv'), os.path.join(data_path, 'train_v2'),
            batch_size=16, split_ratio=1
        )
        dice_scores = []
        for img, mask in data:
            predicted_mask = model.predict(img)
            predicted_mask_thresholded = predicted_mask > 0.1 
            dice_scores.append(calculate_dice_score(mask, predicted_mask_thresholded))
        print(f"Dice score: {np.mean(dice_scores)}")
