#!/usr/bin/env python
# coding: utf-8

# ### FACE AND EYE DETECTION USING HAAR CASCADE CLASSIFIER

# In[6]:


pip install opencv


# In[1]:


import numpy as np
import cv2
from keras.preprocessing import image


# In[2]:


#We point OpenCV's CascadeClassifier funtion to where our 
#classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier("D:\IMARTICUS\Technical Seminar\Computer_Vision\haarcascade_frontalface_default.xml")


# In[3]:


#Load our image then convert it to grayscale
image = cv2.imread("D:\\IMARTICUS\\NEURAL NETWORKS\\CNN\\project data of students\\train\\7 Chiranjeevi\\CHI2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[4]:


#Our classifier returns the ROI of the deteced face as a tuple
#It stores the top left coordinate and the bottom right coordinates

faces = face_classifier.detectMultiScale(gray,1.3,5)


# In[15]:


#When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No faces found")
    
#We iterate through our faces array and draw a rectangle
#over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (127,0,225), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKeyEx(0)
    
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




