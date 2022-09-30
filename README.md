# Malaria cell image Classification

# Abstract:

The main idea of this project is to perform the image classification task and in this
project we classify the given cell images into parasitized or un-infected to detect wether
the cell is infected with malaria or not, to perform this we used deep learning techniques
like CNN and deep transfer learning techniques VGG16, Resnet and Densenet.

# Introduction:

In the current situation, human health is a major concern. Various diseases have a
negative impact on human society and malaria is one of the deadliest diseases it has
taken many lives, This infectious disease which started in Africa infected every
continent except Antarctica, If the infection becomes severe it may cause kidney failure,
seizures, mental confusion, coma, and death so it is important to diagnose malaria at an
early stage to save lives. Our goal is to classify the given cell image into one of the two
classes parasitized or un-infected. The dataset used for this project contains 27,562 cell
images of two classes, and in this project, we used computer vision and deep learning
techniques like CNN, and some pre-trained techniques like Resnet-50, VGG16, and
Densenet-201. 70% of the total data i.e 19,292 images are used for training and 8266
images are used for Validation, to evaluate the performance Accuracy was used as a
metric and achieved an validation accuracy of 0.90%,0.88%,0.74,0.68,0.92 for the
above-mentioned models.

# Procedure:

**Exploratory Data Analysis:**
Exploratory data analysis is one of the important steps in solving any problem because
it gives an idea of how data is distributed across all the target classes in this project
EDA is performed to get the basic information about the data and the results shown
below.



![alt text](https://github.com/kpogula/Maleria-Cell-Classification/blob/main/images/Figure%201.png?raw=true)


![alt text](https://github.com/kpogula/Maleria-Cell-Classification/blob/main/images/Figure%202.png?raw=true)

**Data Preprocessing:**
The next important step in solving any deep learning or machine learning problem is
data processing in this step the data that is used to solve the problem is prepared for
the modeling using different techniques.
The **ImageDataGenerator()** from Keras preprocessingwas used for performing the
preprocessing task.


![alt text](https://github.com/kpogula/Maleria-Cell-Classification/blob/main/images/Figure%203.png?raw=true)


**flow_from_directory()** method was used to split datainto train and validation.
we used 70% of the data for training and 30% of the data for validation, the size of the
data set is 27558 images and out of which 19292 images are used for training and 8266
are used for validation.


![alt text](https://github.com/kpogula/Maleria-Cell-Classification/blob/main/images/Figure%204.png?raw=true)


The image size of 256x256 is used for this problem and the batch size of 64.
The loss function used is binary cross-entropy and adam is used as optimizer and
accuracy is used as a performance metric.

**Modeling:**
5 different methods CNN, VGG16, Resnet50, Resnet101, and Densenet201 were used
to solve the problem in the above methods CNN was implemented from scratch and the
remaining models are pre-trained models the pre-trained models are imported using
**tensorflow.keras.application()** method. All modelsare trained for 10 epochs and
adam as an optimizer, accuracy as a performance metric, and binary cross-entropy as
loss function.


![alt text](https://github.com/kpogula/Maleria-Cell-Classification/blob/main/images/Figure%206.png?raw=true)



**Analysis:**

```
Model Training Accuracy Validation Accuracy
```
```
CNN 0.93 0.
```
```
VGG16 0.91 0.
```
```
Resnet 50 0.67 0.
```
```
Resnet 101 0.67 0.
```
```
Densenet 201 0.92 0.
```
**Epochs vs Accuracy**

```
(1) (2)
```
```
(3) (4)
```

```
(5)
```
## (1)CNN

## (2) VGG

```
(3) Resnet 50
```
```
(4) Resnet 101
```
```
(5) Densenet 201
```
Out of 5 models CNN, VGG16, and Densenet201 achieved better accuracy than Resnet
50 and Resnet 101 The accuracies of these models was fluctuating more when
compared to other models and Resnet 50 performed worse when compared to
remaining models and Resnet 50 model was overfitting which can be concluded by
observing the training and validation accuracies of the model.

```
Fig 6: Accuracy of different models.
```
**Error Analysis:**

```
Model Training Loss Validation Loss
```
```
CNN 0.19 0.
```
```
VGG16 0.21 0.
```
```
Resnet 50 0.62 0.
```

```
Resnet 101 0.66 0.
```
```
Densenet 201 0.56 0.
```
Epochs vs Loss

```
(1) (2)
```
```
(3)
```
```
(4)
```

```
(5)
```
## (1)CNN

## (2) VGG

```
(3) Resnet 50
```
```
(4) Resnet 101
```
```
(5) Densenet 201
```
At the end of the training CNN, VGG16 and Resnet 101 loss converged well and Resnet
50, Densenet201 models losses fluated more when compared o remaining models and
VGG16 had the smallest loss value for both training(0.21) and validation(0.28).

**Discussion:**
From the project i have learned how image classification task can be performed using
the deep learning techniques and achieved highest accuracy of 0.92 using deep
transfer learning technique called Dense net 201 and for the further analysis Ensemble
techniques and attention techniques can be used for the improvement of the project.

**Acknowledgements:**
I would like to thank my professor Francis, Joseph T for inspiring me to do this project
and give credits to National Library of Medicine for making large malaria cell data set
available for the public which is used to solve this problem and i would like to thank all
the deep learning reachers who designed some of the powerful models like VGG16,
Resnet, Densenet etc which can be use to solve many deep learning problem with
better results.

**References:**

[1] Shuying Liu and Weihong Deng., “Very deep convolutional neural network based
image classification using small training sample size” 2015 3rd IAPR Asian Conference
on Pattern Recognition (ACPR).

[2]C. Mehanian, M. Jaiswal, C. Delahunt, C. Thompson, M. Horning, and L. Hu,
“Computer-automated malaria diagnosis and quantization using convolutional neural


networks,” in _Proceedings of the IEEE International Conference on Computer Vision_ ,
pp. 116–125, Venice, Italy, 2017.

[3]A. Vijayalakshmi and B. Rajesh Kanna, “Deep learning approach to detect malaria
from microscopic images,” _Multimedia Tools and Applications_ ,vol. 79, 2019.

[4]E. Var and F. B. Tek, “Malaria parasite detection with deep transfer learning,” in _2018
3rd International Conference on Computer Science and Engineering (UBMK)_ , pp.
298–302, Sarajevo, Bosnia-Herzegovina, September 2018.

[5]] Esra Var and F. Boray Tek. Malaria Parasite Detection with Deep Transfer Learning.
In UBMK 2018 - 3rd International Conference on Computer Science and Engineering,
pages 298–302. Institute of Electrical and Electronics Engineers Inc., dec 2018.

**Data:**
The data for this task is taken from kaggle and the data consists of 27,586 images in
total and 13,780 images from each class.


