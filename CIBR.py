# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:18:17 2019

@author: HeyJude
"""
from __future__ import division

from scipy.spatial.distance import euclidean
import cv2
import mahotas as mt
import numpy as np
import glob
import time
time1 = time.time()
import cv2 as cv
from pylab import *
from numpy import *
import pickle

vect=[]
images = []
titles = []

### laod all the pictures from our folder  ###
def load_images_from_folder(folder):

	for f in glob.iglob("database\*"):
		image = cv2.imread(f)
		titles.append(f)
		images.append(image)
	return images

### Define our descriptor ###	
## Haralick,textures features ##
def haralickTexture(image):
        #image = cv2.imread(filename)
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
		
        amin, amax = min(ht_mean), max(ht_mean)
        for i, val in enumerate(ht_mean):
            ht_mean[i] = ((val-amin) / (amax-amin))*25
        return ht_mean
	


## Histogram calculation ##
def calclhist(images):
#first task, Histogram intersection
# Convert it to HSV

		image_HSV = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
		hist_img = cv2.calcHist([image_HSV], [0,2], None, [180,256], [0,180,0,256])
		cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
		print (len(hist_img.flatten()))
		histFlatten=hist_img.flatten()
		
		amin, amax = min(histFlatten), max(histFlatten)
		for i, val in enumerate(histFlatten):
                          histFlatten[i] = ((val-amin) / (amax-amin))*10
		return histFlatten
#we flatten it to make it a one dimention array    
		
## Gabor Filter  ##	
############gamma
def BuildTable(gamma):
    table=[]
    for i in range(0,256):
        x1=(i+0.5)/256
        x2=1/gamma
        x3=np.power(x1,x2)
        x4=x3*256-0.5
        table.append(x4)
    return table


#############gamma
def GammaCorrectiom(img1,gamma):
    mm=BuildTable(gamma)
    m, n = img1.shape
    for i in range(0, m):
        for j in range(0, n):
            img1[i][j] = mm[img1[i][j]]
    return img1


#############DoG
def DoG(img1,sig1,sig2):
    img2= cv.GaussianBlur(img1, (3, 3),sig1) - cv.GaussianBlur(img1, (3, 3), sig2)
    return img2



def gaborVect(images):
    hist1=[]
    
    src = cv2.cvtColor(images, cv.COLOR_BGR2GRAY)
    src=GammaCorrectiom(src,0.8)
    src=DoG(src,0.9,0.3)
    src=cv.equalizeHist(src)
    ##gabor
    src_f = np.array(src, dtype=np.float32)
    src_f /= 255.
    us=[7,12,17,21,26]             
    
    vs=[0,pi/4,2*pi/4, 3*pi/4, 4*pi/4, 5*pi/4, 6*pi/4,7*pi/4]
    kernel_size =21
    sig = 5                     
    gm = 1.0                    
    ps = 0.0                    
    i=0
    for u in us:
        for v in vs:
            lm = u
            th = v*np.pi/180
            kernel = cv.getGaborKernel((kernel_size,kernel_size),sig,th,lm,gm,ps)
            kernelimg = kernel/2.+0.5
            dest = cv.filter2D(src_f, cv.CV_32F,kernel)
            dst=np.power(dest,2)
            p1 = dst[2:5, 1:8]
            p2 = dst[5:1, 2:7]
            for each in p1:
                for each1 in each:
                    hist1.append(each1)
            for each0 in p2:
                for each2 in each0:
                    hist1.append(each2)
    amin, amax = min(hist1), max(hist1)
    for i, val in enumerate(hist1):
        hist1[i] = ((val-amin) / (amax-amin))*15
    print(len(hist1))
    return hist1

### displaying the chosen pictures based on enclidean distances ###
def displayResults(indexImagesResults):
    for j in range(len(indexImagesResults)): 
		   print (len(indexImagesResults))
		   cv2.imshow("image%d"%(j) ,dataImages[indexImagesResults[j]]   )#dataImages[indexImagesResults[j]] 
    cv2.waitKey(0)
    cv2.destroyAllWinodws()

###	extract all features that we used and stack them on one array ###
def extractFeatures(image) :   
	vect_hist = calclhist(image)
	vect_haralick = haralickTexture(image)
	vect_gabor = gaborVect(image)
	vect = np.hstack(( vect_haralick, vect_hist,vect_gabor)).ravel()
	return vect


### Compare one image with other images from a folder and normalize distances,if distance == 0 so we have same image 
def results(queryImage , dataImages) :
    results = {}
    for i in range(4):
        queryFeatures = extractFeatures(queryImage)
        img = dataImages[i]
        featuresImages = extractFeatures(img) 
        dist = euclidean(queryFeatures,featuresImages)
        print("distance is: %f" %(dist))		
        results[i] = dist		
    amin, amax = min(results), max(results)
    for i, val in enumerate(results):
        results[i] = ((val-amin) / (amax-amin))*100
    print(results)
    return results


### Compare one image with other images from an indexing dataset ###
def resultss(queryImage) :
    resultss = {}
    queryFeatures = extractFeatures(queryImage)
    print(queryFeatures)
    for i in range(1):
        featuresImages = r[i] 
        #vect = normalize(vect[:,np.newaxis], axis=0).ravel()
        dist = euclidean(queryFeatures,featuresImages)
        print("distance is: %f" %(dist))		
        resultss[i] = dist
    print(featuresImages)
    return resultss

def Reverse(lst): 
    new_lst = lst[::-1] 
    return new_lst 

### sort the distances 
def getIndexImages(resultss):
    sortedResults = sorted(resultss.items(), key=lambda x: x[1])
    arr = []
    for j in range(len(resultss)): 
       a = sortedResults[j]
       index = a[0]
       arr.append(index)		   
    print(arr)
    return arr 

######	test our functions (test comparisons on folder dataset,we calculate the vector of each image we want to compare)
dataImages = load_images_from_folder('data')###dataset folder to test
queryImage = cv2.imread('database/BurjAlKhalifa_place_07.png') 

distanceResults = results(queryImage,dataImages)
indexImagesResults = getIndexImages(distanceResults)
displayResults(indexImagesResults)
#####




### Indexing Data set
### save each vector of each image in one dictionary and save it in a pickle file
i=0
index = {}
for f in glob.iglob("data\*"):
		img = cv2.imread(f)
		titles.append(f)
		images.append(img)
		vect_hist = calclhist(img)
		vect_haralick = haralickTexture(img)
		vect_gabor = gaborVect(img)
		vect = np.hstack((  vect_haralick,vect_hist,vect_gabor))
		index[i]=vect
		print(i)
		i+=1
print(index)


with open('indexingData.pickle', 'ab') as handle:
	pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
r={}
with open('indexingData.pickle', 'rb') as handle:
    r = pickle.load(handle)
	
print(r)




### our searcher(test comparisons on indexing dataset, all dataset images vectors are calculates so we just compare)
distanceResults = resultss(queryImage)
indexImagesResults = getIndexImages(distanceResults)
displayResults(indexImagesResults)


	

