import argparse
import tensorflow as tf
import os
from model import build_model
from load_data import create_tf_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_model_path', type=str,
        default=os.path.join('models', 'segmentation_model_trianed.keras'),
        help='Path to save the model'
    )
    parser.add_argument(
        '--epochs', type=int, default=5, help='Number of epochs'
    )
    parser.add_argument(
        '--split_ratio', type=float, default=0.8, help='Train/validation split ratio'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Batch size'
    )
    parser.add_argument(
        '--data_path', type=str, default=os.path.join('data', 'airbus-ship-detection'), help='Path to dataset'
    )
    args = parser.parse_args()
    save_model_path = args.save_model_path
    epochs = args.epochs
    split_ratio = args.split_ratio
    batch_size = args.batch_size
    data_path = args.data_path
    
    model = build_model()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    csv_path = os.path.join(data_path, "train_ship_segmentations_v2.csv")
    images_dir = os.path.join(data_path, "train_v2")
    train_data, valid_data = create_tf_dataset(csv_path, images_dir, batch_size, split_ratio)
    train_data.prefetch(tf.data.AUTOTUNE)
    valid_data.prefetch(tf.data.AUTOTUNE)
    model.fit(train_data, epochs=5, validation_data=valid_data)
    model.save_weights(save_model_path)
    print(f"Model weights saved at {save_model_path}")
