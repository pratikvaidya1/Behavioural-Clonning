# Behaviorial Cloning Project

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[forward]: ./examples/forward.jpg "Forward"
[reverse]: ./examples/reverse.jpg "Reverse"


---
### Files Included & Standards followed for Code Quality

#### 1. This repository includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* Nvidia_20e_512_022angleOffset folder
  * Nvidia_20e_512_022angleOffset.h5 containing a trained convolution neural network 
  * Nvidia_20e_512_022angleOffset_screen.mp4 containing a screen recorded video of a full lap on track.
  * Nvidia_20e_512_022angleOffset_video.mp4 containing the output of video.py for the aboce full lap.
* writeup_behavioural_clonning.md summarizing the results

#### 2. This repository includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py Nvidia_20e_512_022angleOffset.h5
```

#### 3. This repository code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

This model uses the NVIDIA convolutional neural Network (CNN) architecture for learning which can be refered [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). It is also modified to match the requirements. 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (clone.py lines 141). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was manually set to 0.001 (clone.py line 157).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I have used the sample data provided by Udacity, in addition to it there is 1 recorded lap in the forward direction of track 1, 1 recorded lap in the reverse direction of track 1, and specific curve driving recordings in both directions.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test different models which are well known like LeNet (which is discussed during the lectures), NVIDIA, VGG and several others.

I have tried the model LeNet. It is good for the straight lines and very slight curves, When it came to sharp curves , it failed to predict the correct angle.

Then for the next step I have tried NVIDIA model architecture. I splitted my dataset into training set (80%) and validation set (20%). But it has high MSE on validation set and low MSE in training set. this is an example of Overfitting. FOr fixing this the model has been modified by adding dropout layes. First dropout layer is after the flatten layer and then the second one is after fully connected layer.

I tried running car in simulator. But still there are few spots where car fell off the track. To improve the driving behaviour One final step is taken into account. THe last convolution layer was removed. It just on trial and error basis. ANd that made the trick.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 127-151) consisted of a convolution neural network with the following layers and layer sizes ...
|Layer (type) | Output Shape | Param # | Connected to|
|-------------|--------------|---------|-------------|
|image_normalization (Lambda)|(None, 64, 64, 3)|0|lambda_input_1[0][0]|
|convolution_1 (Convolution2D)|(None, 30, 30, 24)|1824|image_normalization[0][0]|
|elu_1 (ELU)|(None, 30, 30, 24)|0|convolution_1[0][0]|
|convolution_2 (Convolution2D)|(None, 13, 13, 36)|21636|elu_1[0][0]|
|elu_2 (ELU)|(None, 13, 13, 36)|0|convolution_2[0][0]|
|convolution_3 (Convolution2D)|(None, 5, 5, 48)|43248|elu_2[0][0]|
|elu_3 (ELU)|(None, 5, 5, 48)|0|convolution_3[0][0]|
|convolution_4 (Convolution2D)|(None, 3, 3, 64)|27712|elu_3[0][0]|
|elu_4 (ELU)|(None, 3, 3, 64)|0|convolution_4[0][0]|
|flatten_1 (Flatten)|(None, 576)|0|elu_4[0][0]|
|dropout_1 (Dropout)|(None, 576)|0|flatten_1[0][0]|
|hidden1 (Dense)|(None, 100)|57700|dropout_1[0][0]|
|elu_5 (ELU)|(None, 100)|0|hidden1[0][0]|
|dropout_2 (Dropout)|(None, 100)|0|elu_5[0][0]|
|hidden2 (Dense)|(None, 50)|5050|dropout_2[0][0]|
|elu_6 (ELU)|(None, 50)|0|hidden2[0][0]|
|hidden3 (Dense)|(None, 10)|510|elu_6[0][0]|
|elu_7 (ELU)|(None, 10)|0|hidden3[0][0]|
|steering_angle (Dense)|(None, 1)|11|elu_7[0][0]|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][forward]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][reverse]

I then recorded three laps on track one in the reverse direction to further generalize the dataset.

To augment the data sat, I also flipped images and angles thinking that this would give wider range of the data to train the model. Which will also help in reducing overfitting.

left and right camera images are also used to train the model. Right camera image with an offset of -0.22 in the steering angle, and the left camera image with an offset of 0.22 in the steering angle.


After the collection process, there are 46,000 data points. I have preprocessed this data by cropping 40 pixels from top of the images which have unusable and unnecessary details, and 20 pixels from the buttom to remove the vehicle hood. Now these images are resized to 64x64.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by me.
