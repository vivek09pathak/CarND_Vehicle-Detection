
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg


# In[2]:


def spatial_bin(img,size=(32,32)):
    
    return cv2.resize(img,size).ravel()


# In[3]:


def color_hist(img,nbins=32,bins_range=(0,256)):
    
    channel1_hist=np.histogram(img[:,:,0],bins=nbins, range=bins_range)
    channel2_hist=np.histogram(img[:,:,1],bins=nbins, range=bins_range)
    channel3_hist=np.histogram(img[:,:,2],bins=nbins, range=bins_range)
    
    return np.concatenate((channel1_hist[0],channel2_hist[0],channel3_hist[0]))
    

