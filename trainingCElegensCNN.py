import torch
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        image = cv.Canny(image=image, threshold1=60, threshold2=140)

        label = torch.tensor(int(self.imgLabels[index]), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return (image, label)

cElegansDataset = WormsAndNoWormsDataSet(path='C. elegans', transform=ToTensor())

trainSet, testSet = random_split(cElegansDataset, [8301, 2075])

# trainLoader = DataLoader(cElegansDataset, batch_size=1000, shuffle=True)
trainLoader = DataLoader(trainSet, batch_size=1000, shuffle=True)
testLoader = DataLoader(testSet, batch_size=1000, shuffle=True)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convLayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 5, 1, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.convLayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 5, 1, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.outLayer = torch.nn.Linear(16 * 8 * 8, 2)

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)

        x = x.view(x.size(0), -1)
        prediction = self.outLayer(x)
        return prediction

cnn = CNN()
lossFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.08, momentum=0.94)
epochs = 30

def train(epochs, cnn, loader):
    cnn.train()

    for currentEpoch in range(epochs):
        avgLoss = 0
        for i, (images, labels) in enumerate(loader):
            currentImage = images
            currentLabel = labels
            
            output = cnn(currentImage)
            loss = lossFunction(output, currentLabel)
            avgLoss += loss.item()
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        avgLoss /= (i + 1)
        print(currentEpoch, avgLoss)
        if (avgLoss < 0.16):
            break


def test(cnn, loader):
    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            testOutput = cnn(images)
            predY = torch.max(testOutput, 1)[1].data.squeeze()
            accuracy = (predY == labels).sum().item() / float(labels.size(0))
        
    print(accuracy)

if __name__ == '__main__':
    # train(epochs, cnn, trainLoader)
    cnn = torch.load('cElegansModel.pt')
    test(cnn, trainLoader)
    test(cnn, testLoader)
    torch.save(cnn, 'cElegansModel.pt')