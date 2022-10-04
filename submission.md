## Submission Template

### Project overview
The goal of this project is to use a deep learning based object detection model to detect and localize
pedestrians, cyclists and vehicles visible in a camera sensor from a self driving car ego vehicle. Such
a system is an important component of self driving car systems as it informs the control system to
navigate around such obstacles and not bump into them (causing accidents).

### Set up, Training and Improving on the base performance
0. I used the udacity project workspace to complete this project in the GPU instance. The requirements are already installed in
   that workspace but I created a requirements.txt file by running `pip freeze > requirements.txt`. These requirements may be
   installed via the command `pip install -r requirements.txt`.
1. The for training, validation and testing was already set up in the project workspace. If one were running the model in their
	own GPU instance, they could download the data from Google Cloud Bucket at 
	(https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as described in the project instructions
	and splitting them up into train, test and validation splits by running `create_splits.py`.
2. The exploratory data analysis was done by running the jupyter notebook `Explore Data Analysis.ipynb`. Please find the saved state of the
	notebook in the commited project. Similarly, various data augmentations were also experimented with using the notebook
	`Explplore augmentations.ipynb`.
3. The pipeline config file was edited using the following command to set up the pipekline for the SSD model:
	```
		python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
	```
4. Training was launched using the command: 
	```
	python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
	```
5. Evaluation process was started by running the following command:
	```
	python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
	```
6. Training and Evaluation performance can be observed through tensorboard which can be launched using the following command:
	```
	python -m tensorboard.main --logdir experiments/reference/
	```

### Dataset
#### Dataset analysis
The base training curve, the final training curve from tensorboard, the distribution of the labels
in the dataset and image augmentation analysis is attached in screenshots in the attached images.
There are a lot more cars and pedestrians compared to cyclists in the dataset.

#### Cross validation
The validation strategy adopted here is to first train the model on a train set and then evaluate it
on a held out validation set. The best performing model on the validation set was used to generate
the animation on the dataset contained in the test set. Train, validation and test sets contain
non overlapping images but they are assumed to be drawn from the same distribution.

#### Training

I used the already configured SSD based model for this project which contains a resnet 50 v1 based
backbone as the feature extractor. With the default pipeline_new.config file, the minimum loss I could get to was limited to around 7.

#### Improve on the reference
I experimented with changing the optimizer and data augmentation to improve the training
performance. Ultimately, the ADAM optimizer with an initial learning rate of 0.001 and exponentially
decaying learning rate with decay steps of 700, I was able to get down to a training loss of 
about 1.3. The augmentations I ended up using were the following:
	a. random horizontal flip
	b. randomy rgb to gray conversion
	c. randomly adjust brightness
	d. random adjust contrast
	e. random adjust hue
	f. random adjust saturation
	g. random crop image
