# face-detection with deep learning
Face detection is a computer technology being used in a variety of applications that identifies human faces in digital images. Face detection also refers to the psychological process by which humans locate and attend to faces in a visual scene. We use feature based approach, in which a model is first trained as a classifier and then used to differentiate between facial and non-facial regions.
_**Goal:**_ determine if there are any faces in the image or video.
Problem Statement: Human faces are difficult to model as there are many variables that can change for example facial expression, orientation, lighting conditions and partial occlusions such as sunglasses, scarf, mask etc. The result of the detection gives the face location parameters and it could be required in various forms, for instance, a rectangle covering the central part of the face, eye centers or landmarks including eyes, nose and mouth corners, eyebrows, nostrils, etc.
**Face Detection Methods**: There are two main approaches for Face Detection:
    1.Feature Base Approach: Objects are usually recognized by their unique features. It locates faces by extracting structural features like eyes, nose, mouth etc. and then uses them to detect a face. Typically, some sort of statistical classifier qualified then helpful to separate between facial and non-facial regions. In addition, human faces have particular textures which can be used to differentiate between a face and other objects. Moreover, the edge of features can help to detect the objects from the face.    
    2.Image Base Approach: In general, Image-based methods rely on techniques from statistical analysis and machine learning to find the relevant characteristics of face and non-face images. The learned characteristics are in the form of distribution models or discriminant functions that is consequently used for face detection. In this method, we use different algorithms such as Neural-networks, HMM, SVM, AdaBoost learning. 

The Viola Jones algorithm has four main steps:
1.Selecting Haar-like features
2.Creating an integral image
3.Running AdaBoost training
4.Creating classifier cascades

**Face Detection**: The very first task we perform is detecting faces in the image or video stream

**Feature Extraction**: Using face embeddings to extract the features out of the face. A neural network gets an image of the person’s face as input and provides a vector to represents the most important features of a face as an output. In ML, that vector is called embedding and thus we call this vector as face embedding. 
While training the neural network, the network learns to output similar vectors for faces that look similar. For example, if I have multiple images of faces within different timespan, of course, some of the features of my face might change but not up to much extent. So in this case the vectors associated with the faces are similar or in short, they are very close in the vector space. 
**Comparing faces**: Now that we have face embeddings for every face in our data saved in a file, the next step is to recognise a new t image that is not in our data. So the first step is to compute the face embedding for the image using the same network we used above and then compare this embedding with the rest of the embeddings we have. We recognise the face if the generated embedding is closer or similar to any other embedding.

**Some Keywords to think like a data science student:**

_harcascade_: Cascade means here it goes down cascading over and over again. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, Haar features shown in the below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum of pixels under the black rectangle.

_Haar Features aka Rudimentary blocks: approximates the relationship between the pixels within the box
  
      Edge Features

      Line Features

      Four Rectangle Features_


Using these features(layering over and over again and cascading down the images are found)
**How the Face Detection Works:-**
Firstly the image is imported by providing the location of the image. Then the picture is transformed from RGB to Grayscale because it is easy to detect faces in the grayscale.
After that, the image manipulation used, in which the resizing, cropping, blurring and sharpening of the images done if needed. The next step is image segmentation, which is used for contour detection or segments the multiple objects in a single image so that the classifier can quickly detect the objects and faces in the picture.

  
  
For each feature, it finds the best threshold which will classify the faces to positive and negative. Obviously, there will be errors or misclassifications. We select the features with minimum error rate, which means they are the features that most accurately classify the face and non-face images. (The process is not as simple as this. Each image is given an equal weight in the beginning. After each classification, weights of misclassified images are increased. Then the same process is done. New error rates are calculated. Also new weights. The process is continued until the required accuracy or error rate is achieved or the required number of features are found). The final classifier is a weighted sum of these weak classifiers. It is called weak because it alone can't classify the image, but together with others forms a strong classifier.

This algorithm used for finding the location of the human faces in a frame or image. All human faces shares some universal properties of the human face like the eyes region is darker than its neighbour pixels and nose region is brighter than eye region.

The haar-like algorithm is also used for feature selection or feature extraction for an object in an image, with the help of edge detection, line detection, centre detection for detecting eyes, nose, mouth, etc. in the picture. It is used to select the essential features in an image and extract these features for face detection.

The next step is to give the coordinates of x, y, w, h which makes a rectangle box in the picture to show the location of the face. After this, it can make a rectangle box in the area of interest where it detects the face. 


How harcascade works: Harcascade Visualization: https://www.youtube.com/watch?v=hPCTwxF0qf4






**Resources**:
https://realpython.com/face-detection-in-python-using-a-webcam/

https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1

https://realpython.com/face-recognition-with-python/

https://www.mygreatlearning.com/blog/face-recognition/

https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81

https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

