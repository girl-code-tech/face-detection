# face-detection with deep learning
**Face Detection**: The very first task we perform is detecting faces in the image or video stream

**Feature Extraction**: Using face embeddings to extract the features out of the face. A neural network gets an image of the person’s face as input and provides a vector to represents the most important features of a face as an output. In ML, that vector is called embedding and thus we call this vector as face embedding. 
While training the neural network, the network learns to output similar vectors for faces that look similar. For example, if I have multiple images of faces within different timespan, of course, some of the features of my face might change but not up to much extent. So in this case the vectors associated with the faces are similar or in short, they are very close in the vector space. 
**Comparing faces**: Now that we have face embeddings for every face in our data saved in a file, the next step is to recognise a new t image that is not in our data. So the first step is to compute the face embedding for the image using the same network we used above and then compare this embedding with the rest of the embeddings we have. We recognise the face if the generated embedding is closer or similar to any other embedding as shown below:



**Resources**:
https://realpython.com/face-detection-in-python-using-a-webcam/

https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1

https://realpython.com/face-recognition-with-python/

https://www.mygreatlearning.com/blog/face-recognition/

https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81

https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

