## Submission Template

### Project overview
The goal of this project is to use a deep learning based object detection model to detect and localize
pedestrians, cyclists and vehicles visible in a camera sensor from a self driving car ego vehicle. Such
a system is an important component of self driving car systems as it informs the control system to
navigate around such obstacles and not bump into them (causing accidents).

### Set up, Training and Improving on the base performance
1. I used the already configured SSD based model for this project which contains a resnet 50 v1 based
backbone as the feature extractor. With the default pipeline_new.config file, the minimum loss I could get to was limited to around 7.
2. I experimented with changing the optimizer and data augmentation to improve the training
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
