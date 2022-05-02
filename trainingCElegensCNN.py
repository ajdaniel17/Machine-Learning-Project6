import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device('cuda')

"""
LOAD DATASET
"""
class WormsAndNoWormsDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.imgLabels = np.load(path + '/labels.npz')['labels'].tolist()
        self.path = path 
        self.transform = transform
    
    def __len__(self):
        return len(self.imgLabels)

    def __getitem__(self, index):
        image = cv.imread(self.path + '/image' + str(index) + '.png')
        image = cv.resize(image, (28, 28), interpolation=cv.INTER_AREA)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.GaussianBlur(image, (3,3), 0)

        sobelx = cv.Sobel(src=image, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
        sobely = cv.Sobel(src=image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
        sobelxy = cv.Sobel(src=image, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)

        image = cv.Canny(image=image, threshold1=60, threshold2=140)
        label = torch.tensor(int(self.imgLabels[index]), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return (image, label)

cElegansDataset = WormsAndNoWormsDataSet(path='C.ElegansData', transform=ToTensor())

trainSet, testSet = random_split(cElegansDataset, [8301, 2075])

# trainLoader = DataLoader(cElegansDataset, batch_size=1000, shuffle=True)
trainLoader = DataLoader(trainSet, batch_size=1000, shuffle=True)
testLoader = DataLoader(testSet, batch_size=1000, shuffle=True)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convLayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.convLayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.outLayer = torch.nn.Linear(32 * 7 * 7, 2)

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)

        x = x.view(x.size(0), -1)
        output = self.outLayer(x)
        return output

cnn = CNN()

lossFunction = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(cnn.parameters(), lr=0.05, momentum=0.92)

epochs = 25

def train(epochs, cnn, loaders):
    cnn.train()

    N = len(loaders)

    for currentEpoch in range(epochs):
        for i, (images, labels) in enumerate(loaders):
            currentImage = images
            currentLabel = labels
            
            output = cnn(currentImage)
            loss = lossFunction(output, currentLabel)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print(currentEpoch, i + 1, loss.item())


def test(cnn, loaders):
    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders:
            testOutput = cnn(images)
            predY = torch.max(testOutput, 1)[1].data.squeeze()
            accuracy = (predY == labels).sum().item() / float(labels.size(0))
        
    print(accuracy)

def predict20(cnn, loaders):
    images, labels = next(iter(loaders))
    testOutput = cnn(images[20:50])
    predY = torch.max(testOutput, 1)[1].data.squeeze()
    print(predY)
    print(labels[20:50].numpy())

if __name__ == '__main__':
    train(epochs, cnn, trainLoader)
    test(cnn, trainLoader)
    test(cnn, testLoader)
    predict20(cnn, testLoader)