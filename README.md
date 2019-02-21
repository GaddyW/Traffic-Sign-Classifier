# **Traffic Sign Recognition** 

This project is part of Udacity's self-driving car nanodegree.  In this project, I use a convolutional neural network to classify street signs.  The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

#### Project output can be found here
HTML here: [project html](https://github.com/GaddyW/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.html)
Jupyter notebook here:  [project code](https://github.com/GaddyW/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

[//]: # (Image References)

[image1]: ./output_images/example_image.png "Dataset Visualization"
[image2]: ./output_images/augmented_image.png "Augmentation"
[image3]: ./output_images/sign_histogram.png "Histogram"
[image4]: ./output_images/test_sign_accuracy.png "Accuracy"
[image5]: ./output_images/right_turn.jpg "Traffic Sign"
[image6]: ./output_images/web_images.png "Images from the web"
[image7]: ./output_images/web_augmented.png "Precprocessed images from the web"



#### 0. Sources:  I read through the following papers as part of my work.  They were very helpful considering this is the first time I've used tensorflow
1)	AlexNet: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf - AlexNet
2)	Discussion on different optimizers:  http://ruder.io/optimizing-gradient-descent/ 
3)	Medium – Data Augmentation Techniques in CNN using Tensor Flow, Prasad Pai
4)	https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
5)	On dropout and batch normalization: https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16 - 
7)	Sanity checking - http://cs231n.github.io/neural-networks-3/#sanitycheck




### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy's shape function to get infromation on the data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

You can find visualization of all sign types in the Jupyter notebook and the HTML.  An example of 30kph signs is shown here:
![alt text][image1]


Not all images are equally represented in the training set as you can see in the below histogram.  I'll use augmentation to create equal amounts of each image type.
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The goal of augmentation and preprocessing was to create equal amounts of each sign type, with various levels of perturbation and augmentation to make training hard.  You will find the code for this in cell 22 of the Jupyter notebook.  

As a first step, I calculated the number of existing examples in the training set for each sign type.  I then created a loop to augment each sign type up to 5000 images.

In the first step of the augmentation, I randomly applied two of the following techniques to a randomly chosen image.  
    1) random translation of up to 8 pixels, cropping anything that moves beyond the images borders
    2) zooming in or out by up to 30%
    3) adding gaussian noise
    4) random rotation of up to +/- 20 degrees
    
I had initially included horizontal flipping as well, but this can be problematic for signs that indicate right or left turns.  The direction is a critical component of the sign's meaning.  After I removed this techniques, validation accuracy improved by 3%.  

In hindsight, I think it might have been worthwhile to use perspective transformations as well as an augmentation technique.  I would implement this in future work, because traffic signs are likely to be photographed at an angle from the vehicle's cameras.

After creating generating a total set of 43x5000 = 215,000 images, I applied three more techniques:
    1) histogram equalization - I wasn't familiar with this techqnique, but I tried after seeing Tracy X's recommendation in the student hub
    2) grayscaling
    3) normalization - I used the course provided technique of (x-128)/128.  I saw in my research other methods using the image's own max and min but decided not to implement for lack of time.

Here is an example of augmented images after all these stages:

![alt text][image2]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

To build my final model, I relied heavily on the LeNet project as a starting point.  After reading through the AlexNet paper, I added two additional convolution layers and dropout in the fully connected layers.  Each additional convolutional layer contributed nearly 3% in accuracy.  The dropout layer contributed to validation accuracy and prevented overfitting.  I attempted to use L2 normalization as well (you can see the code commented out in cell 24 of the Jupyter notebook's loss operation, but I found it to cause more trouble than it was worth.

Additionally, I had some trouble with using random_normal for weights and biases.  They had been successful in the LeNet quiz, but when I copied it here, I got terrible accuracy.  Apparently the data points beyond two standard deviations were saturating some of the neurons.  After switching to truncated normal for weights and zeros for biases, things improved dramatically.


Changing the last convulutional layer's padding from Valid to Same increased accuracy by almost 3%.  It added many more features which could be trained.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU          		| etc.        									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6    				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs  3x3x64	|
| RELU          		| etc.        									|
| Convolution 3x3	    | 1x1 stride, SAME p adding, outputs  3x3x128	|
| RELU          		| etc.        									|
| Fully connected  		| 1152 to 120  									|
| Dropout        		|           									|
| Fully connected  		| 120 to 84  									|
| Fully connected  		| 84 to 43   									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
You will find the code for this discussion in cells 23 and 24 of the Jupyter notebook

1) Optimizer:  After watching the lesson videos, I had planned to manually incorporate momentum into the optimizer.  But then I realized the LeNet quiz already used Adam instead of Gradient Descent, and momentum was inherently included in my optimization.  So, I stuck with Adam.

2) Batch size:  Stuck with 128.  It worked well.

3) Epochs:  Validation stopped improving around epoch 8, so I quit then to prevent overfitting.

4) Learning rate:  I know that Adam uses learning rate annealing with default Betas, but I also implemented it myself.  If you look in cell 24, you'll see that I dynamically anneal the learning rate with the function lr = 0.001/(1+EPOCH*0.0001)


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were (As can be seen in the results of cells 25 and 47:
* training set accuracy of 96.1%
* validation set accuracy of 97.6% 
* test set accuracy of 94.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  
I began with the LeNet architecture

* What were some problems with the initial architecture?  
It wasn't sufficiently accurate and it suffered from overfitting

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
As I mentioned earlier, I added two more convolutional layers to increase accuracy.  The LeNet model was underfit.  I also added a dropout layer to prevent overfitting.

* Which parameters were tuned? How were they adjusted and why?
Learning rate - I used dynamic annealing lr = 0.001/(1+EPOCHx0.0001)
EPOCH - I stopped at 8 to prevent overfitting when I saw that learning was not improving
Batch size - I stayed with 128

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
As we learned in the lessons, convolution is particularly useful on images because weight sharing is desired.  
Likewise, dropout is a good regularization technique to prevent overfitting.  I tried using L2 normalization as well, but found it's contribution to be unhelpful

If a well known architecture was chosen:
* What architecture was chosen?  
I started with LeNet because that's what we learned in the lesson.  However, I augmented it based on certain techniques from AlexNet including additional 3x3 convolutional layers, dropout, and some of the augmentation techniques.

* Why did you believe it would be relevant to the traffic sign application?  

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
We see that all three are well above 90%.  Likewise, accuracy is above 90% on most sign types in the training set.
 
#### 5 Accuracy
As you'll see in cell 47, the model's overall accuracy on the test set was 94.4%
Moreover, we can break this down into accuracy by signtype.  The vast majority of sign types perform well, but some score below 80% with the worst at 65%.  With more time, I would dig into this and identify additional techniques to augment the traning set for these particular sign types.  

![alt text][image4]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web.  I used google maps in Bonn, Germany and copied images from the screen:
![alt text][image6]

And here they are after preprocessing
![alt text][image7]

The stop sign might be difficult to clasiffy because of its perspective offset and the coloration changes.  The others seem relatively straightforward, and are predicted correctly by the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction (see in cell :

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30kph         		| 30kph      									| 
| Right turn only		| Right turn only   							|
| Stop					| Yield											|
| Ahead only     		| Ahead only					 				|
| Yield					| Yield											|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is worse than the test accuracy of 94.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 50 and 55 the Jupyter notebook.

##### For the 30kph sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| 30kph     									| 
| .000     				| 50kph 										|
| .000					| 80kph											|
| .000	      			| 70kph     					 				|
| .000				    | 20kph             							|


##### For the Right turn only sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right turn only								| 
| .000     				| End of speed limit (80km/h)   				|
| .000					| Turn right ahead  							|
| .000	      			| Go straight or right   		 				|
| .000				    | Roundabout mandatory  						|


##### For the stop sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .907         			| Yield         								| 
| .082     				| Stop   										|
| .007					| Turn right ahead								|
| .002	      			| Speed limit (30km/h)			 				|
| .001				    | No vehicles        							|

##### For the ahead only sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only    								| 
| .000     				| Turn right ahead								|
| .000					| Ahead only									|
| .000	      			| Turn left ahead   			 				|
| .000				    | Speed limit (60km/h) 							|

##### For the yield sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield         								| 
| .000     				| No vehicles   								|
| .000					| Turn right ahead  							|
| .000	      			| Priority road					 				|
| .000				    | Speed limit (100km/h)							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


you can see the output of the 1st and 2nd convolutional layers at the end of the Jupyter/Html files.
