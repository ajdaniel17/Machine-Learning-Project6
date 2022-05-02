import glob 
import cv2 as cv
import numpy as np

path1 = "C:/Users/ajdan/Documents/PR-ML/Machine Learning/Project 4 ML/Machine-Learning-Project4/Training/C. elegans/Data/0"
path2 = "C:/Users/ajdan/Documents/PR-ML/Machine Learning/Project 4 ML/Machine-Learning-Project4/Training/C. elegans/Data/1"


filenamesNoWorms = glob.glob(path1 + '/*.png')
filenamesWorms = glob.glob(path2 + '/*.png')

images1 = [cv.imread(img) for img in filenamesNoWorms]
images2 = [cv.imread(img) for img in filenamesWorms]

# allimage = []
# allimage.append(images1)
# allimage.append(images2)
allimage = np.concatenate((images1,images2))

path = "C:/Users/ajdan/Documents/PR-ML/Machine Learning/Project 4 ML/Machine-Learning-Project4/Training/C. elegans/thing/"

for i in range(len(allimage)):
    imgname = "image" + str(i) + ".png"
    cv.imwrite(path+imgname,allimage[i])