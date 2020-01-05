from __future__ import division
from scipy.spatial.distance import euclidean
from flask import Flask, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
################
# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
from gtts import gTTS 
import datetime
import pytesseract
import cv2
import mahotas as mt
import numpy as np
import os 
import glob
import time
time1 = time.time()
import cv2 as cv
from pylab import *
from numpy import *
import pickle
import pytesseract
import re
################
###install pytesseract,if windows os copy path of executable 
pytesseract.pytesseract.tesseract_cmd = r'<C:\Users/HeyJude/AppData/Local/Tesseract-OCR>'

STATIC_FOLDER = './static'

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('input.html')

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == 'POST':
        if request.files:
            #Reading the uploaded image
            r={}
            with open('indexing.pickle', 'rb') as handle:
				         r = pickle.load(handle)		
            images = []
            titles = []
            def delete():
                folder = 'C:/Users/HeyJude/flask/flaskapp/static'
                for filename in os.listdir(folder) :
                    
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
            def load_images_from_folder(folder):
		
               for f in glob.iglob("database\*"):
                    image = cv2.imread(f)
                    titles.append(f)
                    images.append(image)
               return images
		
            def haralickTexture(image):

                textures = mt.features.haralick(image)
                ht_mean = textures.mean(axis=0)
                return ht_mean
	
            def calclhist(images):
                image_HSV = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
                hist_img = cv2.calcHist([image_HSV], [0,2], None, [180,256], [0,180,0,256])
                cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
				
                return hist_img.flatten()
		    
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
                    hist1[i] = ((val-amin) / (amax-amin))*25
                return hist1

            def delete():
                folder = 'C:/Users/HeyJude/flask/flaskapp/static'
                for filename in os.listdir(folder) :
                    
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))

						
            def displayResults(indexImagesResults):
                images = []
                delete()
                for j in range(len(indexImagesResults)): 
                       cv2.imwrite(os.path.join('./static', datetime.datetime.now().strftime("%y%m%d_%H%M%S")+"plane"+str(j)+".png"), dataImages[indexImagesResults[j]])
                       images.append(datetime.datetime.now().strftime("%y%m%d_%H%M%S")+"plane"+str(j)+".png")					   #cv2.imshow("img",conImg)
                return images
			
            def extractFeatures(image) :   
                vect_hist = calclhist(image)
                vect_haralick = haralickTexture(image)
                vect_gabor = gaborVect(image)
                vect = np.hstack(( vect_haralick,vect_hist,vect_gabor)).ravel()
                print(vect)
                return vect
			
            def resultss(queryImage) :
                resultss = {}
                queryFeatures = extractFeatures(queryImage)
                for i in range(50):
                    featuresImages = r[i] 
                    dist = euclidean(queryFeatures,featuresImages)
                    resultss[i] = dist
                    print(dist)
                amin, amax = min(resultss), max(resultss)
                for i, val in enumerate(resultss):
                    resultss[i] = ((val-amin) / (amax-amin))*100
                print(resultss)
                return resultss

            def getIndexImages(resultss):
                sortedResults = sorted(resultss.items(), key=lambda x: x[1])
                arr = []
                for j in range(len(resultss)): 
                   a = sortedResults[j]
                   index = a[0]
                   arr.append(index)
                print(arr)   
                return arr 
			

			
            image = request.files["image"]
            imageName = image.filename
            image.save(os.path.join('./historique', imageName))
            dataImages = load_images_from_folder('database')
            imageBase = os.path.join('./historique', imageName)
            queryImage = cv2.imread(imageBase)
            distanceResults = resultss(queryImage)
            indexImagesResults = getIndexImages(distanceResults)
            

            return redirect(url_for('output',images=displayResults(indexImagesResults)))
            
    else :
        return redirect(url_for('home'))

@app.route('/output')
def output():
    def sorted_aphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)
    
    image = sorted_aphanumeric(os.listdir('C:/Users/HeyJude/flask/flaskapp/static'))
    return render_template('output.html', len = len(image) , image = image)

if __name__ == '__main__':
   app.run()