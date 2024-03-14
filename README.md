# T5_Basser
### Abstract
Baseer project aims to address the challenge of detecting public moral violations through the use of an AI tool. The problem statement revolves around the difficulty in identifying and addressing such violations, prompting the development of the Baseer idea. The solution involves classifying public moral violations into categories such as clothes violation, garden fires, throwing waste, and graffiti infractions. The project utilizes a dataset divided into input and output sections for training the AI model. The methodology includes building a model using Keras and YOLO, with a focus on object detection and classification. The team achieved an accuracy of 96% with the YOLO model and 50% with the Keras model. The deployment of the Face Recognition model complements the efforts in detecting violations and reporting them effectively. The project's success in enhancing social behavior and promoting mutual respect demonstrates its potential impact. Future work includes improving model accuracy and extending its application to organizational settings for monitoring and maintaining social discipline.


### Data Description and Structure: 
This section will contain the dataset of Basser Project 
-the dataset is divided into 

- input :
The input of the Basser Project is the images of the each classifying violation , it is divided into training ,validation and test and each one has images and their Annotated  .
We Collecting the data By 
-Manually 
-Roboflow 

- Output : The Annotated data with Reported information and Violation Cost.

  ![violation](https://github.com/rmimony/T5_BASSER/assets/163456436/74f3a5ca-f347-432d-982b-302c38199e2f)


### Problem statement :
The core of our Problem is how difficult to detect the moral public violation ,since that difficulty the Basser Idea come behind this . 


### Solution :
This section will discuss the specific techniques of the core Problem.

-First Classifying the public moral violation that to detect as

1.Clothes Violation
2.Garden fire 
3.Throwing Waste
4.Graffiti infraction
5.Legal

Building the Model :
Starting with keras.models
First we use keras Model to build Model that can Classify the data by using the Conv2D to build the neural network to train the model .
-Ending with YOLO Model. 
Yolo Model is stands for the object detection problem .
-Easy to use 
-Easy to understand 
-Fast .

The Value behind changing the model is to annotated the Capturing image so the needed of the project is to use Yolo Model .Under the data we have .
-To Start Building Yolo Model we have to Annotate the dataset. We Annotate our data set using roboflow Project tools 
![classes](https://github.com/rmimony/T5_BASSER/assets/163456436/8a4f12d7-0e7e-47fc-9b2c-9ee4a1053d7a)


We should having .yaml file that can contain the data set with its arrangement .
Generate the Yolo Model with .yaml and the  data set that we collected 



Starting building the Face recognition model using the transfer learning and change the last layers wit our stick holder to apply the reporting idea  and the Violation Cost .

Saving the Yolo Model into best.pt file to use it for creating the website .
Saving the Face recognition  model to Connect it to the website .
Deploy 


### Best.pt file 

https://drive.google.com/file/d/1r5Gm6pVrfGxrp20nltmZIiKGDeLfTEqG/view?usp=drive_link


### FaceModel:

https://drive.google.com/file/d/1Ntz8rz_O8O4kpQfxylvycF2ksCrp_CdU/view?usp=drive_link








## Team


-Reem Abdulrazaq Almimony.

-Lama Sultan Askar .

-Wejdan Habib Alazmi.

-Ahlam Abdallah Alderweesh.




