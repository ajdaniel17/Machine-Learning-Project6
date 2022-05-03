import numpy as np
import glob
import cv2 as cv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WormsDataSet(Dataset):
    def __init__(self,images = [], transform=None,*args):
        self.transform = transform
        self.image = images
    
    def __getitem__(self, index):
        temp = self.image[index]
        temp = cv.resize(temp, (28, 28), interpolation=cv.INTER_AREA)
        temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        temp = cv.GaussianBlur(temp, (3,3), 0)
        temp = cv.Canny(image=temp, threshold1=60, threshold2=140)
       
        if self.transform:
            temp = self.transform(temp)
        return (temp)

    def __len__(self):
        return len(self.image)

def test(cnn, loader):
    cnn.eval()
    with torch.no_grad():
        for images in loader:
            testOutput = cnn(images)
            predY = torch.max(testOutput, 1)[1].data.squeeze()          
    return predY  

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

if __name__ == '__main__':
 
    path = input("Please enter the directory path containing test images:\n")
    filenames = glob.glob(path + '/*.png')
    images = [cv.imread(img) for img in filenames]
    cElegansDataset = WormsDataSet(images=images, transform=ToTensor())
    cnn = torch.load('cElegansModel.pt')
    Loader = DataLoader(cElegansDataset, batch_size=1000, shuffle=True)
    labels = test(cnn,Loader)
    print(" ___________________________________ ")
    print("|                    |              |")
    print("|    Image Name      |     Class    |")
    print("|                    |              |")
    print(" ___________________________________ ")
    totals = np.zeros((2))
    for i in range(len(filenames)):
        if labels[i] == 1:
            print("|  %s   |     %i      |" % (filenames[i].replace(path+"\\",""),1))
            totals[1] += 1
        else:
            print("|  %s   |     %i      |" % (filenames[i].replace(path+"\\",""),0))
            totals[0] += 1
    print("Total Tallies:")
    for i in range(2):
        print("Class", int(i) , ":" , int(totals[i]))