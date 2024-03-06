<h1>U-net model for ship semantic segmentation</h1>
<h2>Requirements</h2>
Run files from project root directory.<br />
Put dataset in data/directory.<br />
Python version - 3.11.3<br />
<h2>Usage</h2>
<h3>src/train_model.py</h3>
Run src/train_model.py file to train model.<br />
Parameters:<br />
--save_model_path - path to save model weights, default: models/segmentation_model_trianed.keras<br />
--epochs - number of epochs to train, default: 5<br />
--split_ratio - train/validation split ratio, default: 0.8<br />
--batch_size - batch size, default: 4<br />
--data_path - path to dataset, default: data/airbus-ship-detection<br />
<h3>src/evaluate_model.py</h3>
Run src/evaluate_model.py file to evaluate model on train dataset or generate kaggle submission (or both).<br />
Parameters:<br />
--model_path *path*- path to saved model weights, default: models/ships_weights.keras<br />
--data_path *path*- path to dataset, default: data/airbus-ship-detection<br />
--calculate_dice - whether or not calculate dice score for test dataset, default: False<br />
--kaggle_submission - whether or not generate kaggle submission from test dataset, default: False<br />
<h2>Solution Explanation</h2>
U-net model consists of downsampling and upsampling layers. For downsampling was chosen MobileNetV2 model for it's simplicity, 
to save training and evaluation time, since dataset is large. Skip connections where established from MobileNetV2 layers 
with upsampling layers. For upsampling Conv2DTranspose layers where used.<br />
Final output layer has sigmoid activation for binary classification of a pixel.<br />
Dataset in skewed towards "no-ship" pixels, so half of the images not containing any ships where removed from training 
process (which still didn't quite help to prevent false-positive predictions).<br />

<h2>Notebooks</h2>
dataset_exploration.ipynb contains process of exploring images and their labels.<br />
<br />
model_building.ipynb contains pretty much just summary of models and layers used.<br />
<br />
ships_weights.keras model where trained in jupyter notebook in google colab, which I don't include, since it just has<br />
all the same functions as files in src directory.<br />

<h2>My kaggle competion sumbission</h2>
Potentially could be improved with more epochs to train and playing around with threshold for sigmoid output.
<img src=https://github.com/kstamax/winstarts-submission/assets/64531390/0d0ff67a-e5b9-40cd-82d3-fc8dcede3059 />
