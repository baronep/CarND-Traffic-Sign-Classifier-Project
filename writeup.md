# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class_distribution]: ./output_images/class_distribution.png "Class Distribution"
[class_samples]: ./output_images/class_samples.png "Grayscaling"
[class_distribution_resampled]: ./output_images/class_distribution_resampled.png "Class Distribution Resampled"
[preprocessed_images]: ./output_images/preprocessed_samples.png "Preprocessed Images"
[training_accuracy]: ./output_images/training_accuracy.png "Training Accuracy"

[new1]: ./new_images/1.jpg "New Sign 1"
[new2]: ./new_images/2.jpg "New Sign 2"
[new3]: ./new_images/3.jpg "New Sign 3"
[new4]: ./new_images/4.jpg "New Sign 4"
[new5]: ./new_images/5.jpg "New Sign 5"

[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
### Writeup / README

A link to my [project code](https://github.com/baronep/CarND-Traffic-Sign-Classifier-Project/project.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The training set contains **34799 images**
* The validation set contains **4410 images**
* The size of test set is **12630 images**
* The shape of a traffic sign image is **32x32 pixels**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Below is a sample of the 43 different classes that makeup the training dataset and a diagram illustrating the distribution of classes in the training dataset

![alt_text][class_samples]
![alt text][class_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The preprocessing of the data consisted of 5 main steps.

**1. Class type normalization:**
The initial training dataset is comprised of 43 different classes, but those classes are not evenly distributed among the training set. The class with the most number of images contained 2010 images while the class with the least number of images contained only 180 images. In order to compensate for this, the aim was to modify the dataset such that each class has 2412 images in each class. In order to achieve this, each initial class of images was randomly sampled and duplicated. Each image that was duplicated and re-added to the dataset was slightly transformed in step 2

![alt text][class_distribution_resampled]

**2. Data augmentation:**
Each image that was duplicated in step 1, underwent a slight transformation. One of 4 transformations was randomly chose and applied with a random bounded magnitude to each image: rotation, shift, zoom and shear. The magnitudes of each shift was chosen to be about 5-10% of the original image in order to avoid overly distoring the image

**3. Grayscale:**
The images were converted to grayscale using the cv2.cvtColor function

**4. Histogram Equalization:**
The images were normalized to balance images contrast (some images were much darker than others). In order to achieve this, I used the cv2.equalizeHist function

**5. Normalization:**
The images were originally encoded using uint8: [0,255], but the images were normalized to [-1.0, 1.0] float64

The following are samples of each class after the images have been preprocessed
![alt text][preprocessed_images]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 1x1x512 	|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input: 512, output: 120      									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input: 120, output: 84      									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input: 84, output: 43      									|
| Softmax				|         									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the suggested Adams Optimizer using the softmax_cross_entropy_with_logits function as the loss function. 

I used the following hyperparameters to train my final configuration:

- EPOCHS: 25
- BATCH_SIZE: 128
- LEARNING_RATE: 0.0004
- keep_rate (dropout): 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.985
* validation set accuracy of 0.945
* test set accuracy of 0.918


The following image is a plot of the training accuracy per iteration while training my model

![alt text][training_accuracy]

I first started with the stock LeNet architecture used in the LeNetLab with the full RGB images, but I was unable to get higher than about 93% accuracy. In order to improve accuracy, I made the following modifications

* I changed the input images to grayscale in order to avoid excess, distracting information. While I attemped to train the layer with RGB data, I did not see a significant difference in the performance of the network using RGB vs grayscale data so I chose grayscale to decrease the training time

* I normalized the number of images in each class. By doing this, we are able to roughly ensure that the same number of each type of sign are used to train the network, preventing undesired biases towards particular signs.

* I made transformations to repeated images to improve robustness. By transforming images, I was able to add more information to the dataset and potentially add to the robustness of the system.

* I added the dropout layers to help the model train redundant representations for the same images

* I added the 3rd convolutional layer to try and help the NN recognize more complex features

The training rate was tuned by looking at the accuracy plots to determine whether or not the accuracy was jumping around at stead state (indicating a learning rate that was too high) or whether the learning rate had yet to reach steady state (indicating a need for more training epochs or a higher learning rate). 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new1 ] ![alt text][new2] ![alt text][new3] 
![alt text][new4] ![alt text][new5]

Most of the images are fairly straightforward, except for the animal crossing sign, which looks very similar to the the "Dangerous curve to the left" and "Slippery road" signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Speed Limit (30km/h)     		|  Speed Limit (30km/h)									| 
|  Turn Left Ahead    			| Turn Left Ahead										|
|  Wild Animal Crossing  					| Slippery road										|
|  Speed Limit (70km/h)     		|  Speed limit (30km/h)					 				|
|  Bumpy Road    			| Bumpy road   							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

**Image 1: Speed Limit (30km/h)**
```
Class[1]: Speed limit (30km/h) (0.954634)
Class[2]: Speed limit (50km/h) (0.035700)
Class[0]: Speed limit (20km/h) (0.008747)
Class[7]: Speed limit (100km/h) (0.000464)
Class[4]: Speed limit (70km/h) (0.000261)
```

**Image 2: Turn Left Ahead**
```
Class[34]: Turn left ahead (0.990159)
Class[35]: Ahead only (0.004857)
Class[38]: Keep right (0.003554)
Class[3]: Speed limit (60km/h) (0.000948)
Class[30]: Beware of ice/snow (0.000106)
```

**Image 3**
```
Class[23]: Slippery road (0.641859)
Class[19]: Dangerous curve to the left (0.119892)
Class[31]: Wild animals crossing (0.119260)
Class[21]: Double curve (0.100218)
Class[11]: Right-of-way at the next intersection (0.014946)
```

**Image 4: Speed Limit (70km/h)**
```
Class[1]: Speed limit (30km/h) (0.260079)
Class[8]: Speed limit (120km/h) (0.153737)
Class[4]: Speed limit (70km/h) (0.124491)
Class[2]: Speed limit (50km/h) (0.124377)
Class[7]: Speed limit (100km/h) (0.111548)
```

**Image 5: Bumpy Road**
```
Class[22]: Bumpy road (0.999979)
Class[26]: Traffic signals (0.000019)
Class[25]: Road work (0.000001)
Class[18]: General caution (0.000000)
Class[31]: Wild animals crossing (0.000000)
```
