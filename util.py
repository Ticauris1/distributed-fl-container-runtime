from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)
import torch
# import torchvision
from torch import nn
import torch.nn.functional as F

import time
import numpy as np
import timm
from torchvision.models import resnet18, densenet201, alexnet, vgg19, mobilenet_v2,googlenet #, efficientnet_b0, 
size = (224, 224)
def showResults(test, pred):
    target_names = ['negative', 'positive']
    print(classification_report(test, pred, target_names=target_names))
    accuracy = accuracy_score(test, pred)
    precision=precision_score(test, pred, average='weighted')
    f1Score=f1_score(test, pred, average='weighted') 
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    cm=confusion_matrix(test, pred)
    print(cm)
    return accuracy, precision, f1Score

def eval_Skl_model(xTest,yTest, model):
    yPred = model.predict(xTest)
    startTime = time.time()
    yPredProb = model.predict_proba(xTest)
    timeRequired = (time.time() - startTime)/len(yPredProb)
    print("--- %s seconds ---" % timeRequired)
    return showResults(yTest, yPred)

def testing_time(model,xTest,isTorch=False):
    if isTorch:
        startTime = time.time()
        yPredProb = model(xTest)
        timeRequired = (time.time() - startTime)/len(yPredProb)
    else:
        startTime = time.time()
        yPredProb = model.predict_proba(xTest)
        timeRequired = (time.time() - startTime)/len(yPredProb)
    print("--- %s seconds ---" % timeRequired)
    return timeRequired
def eval_torch_model(vloader, model,device='cuda'):
    correct = 0
    total = 0
    pred = []
    test = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in vloader:
            # print(inputs.shape)
            # inputs = inputs.reshape((-1,3,128,128))
            outputs = model(inputs.to(device))
            if len(outputs.data.shape) == 1:
                data = outputs.data.reshape(1,-1)
            else:
                data = outputs.data
            try:
                _, predicted = torch.max(data, 1)
            except:
                print(outputs.data)
                exit()
            # print(predicted)
            # exit()
            # print(outputs, predicted, labels)
            # print(predicted, labels)
            # exit()
            pred += predicted.detach().cpu().numpy().tolist()
            test += labels.detach().cpu().numpy().tolist()
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))
    test,pred = np.array(test), np.array(pred)
    # print(test.shape,pred.shape)
    # print(test,pred)
    accuracy, precision, f1Score = showResults(test,pred)
    print(accuracy, precision, f1Score)
    model.train()
    return accuracy, precision, f1Score

def getXceptionModel(pretrained=True):
    model = timm.create_model('xception', pretrained=pretrained, num_classes=2)
    for param in model.parameters():
        param.requires_grad = True
    model = model.to('cuda')
    return model

def getDense201Model(pretrained=True): 

    model = densenet201(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    num_classes = 2
    classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.classifier = classifier
    model = model.to('cuda')
    return model

def getGooglenetModel(pretrained=True):
    model = googlenet(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    # model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2) # assuming that the fc7 layer has 512 neurons, otherwise change it 
    model = model.to('cuda')
    return model

def getEfficientnet_b0Model(pretrained=True):
    model = efficientnet_b0(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2) # assuming that the fc7 layer has 512 neurons, otherwise change it 
    model = model.to('cuda')
    return model

def getMobilenet_v2Model(pretrained=True):
    model = mobilenet_v2(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2) # assuming that the fc7 layer has 512 neurons, otherwise change it 
    model = model.to('cuda')
    return model

def getAlexnetModel(pretrained=True): 

    model = alexnet(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    num_classes = 2
    # classifier = torch.nn.Linear(4096, num_classes)
    # model.fc = classifier
    model = model.to('cuda')
    return model

def getResnet18Model(pretrained=True): 

    model = RNModel('resnet18',2,pretrained=pretrained)
    return model.to('cuda')

class RNModel(torch.nn.Module):
    
    def __init__(self, arch,num_classes,pretrained):
        
        super(RNModel, self).__init__()
        print('initializing image classifier')
        
        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.build_model()

    def build_model(self):
        basemodel = None
        
        if self.arch == 'resnet50':
            feat_dim = 2048
            basemodel = resnet50
        elif self.arch == 'resnet101':
            feat_dim = 2048
            basemodel = resnet101
        elif self.arch == 'resnet18':
            feat_dim = 512
            basemodel = resnet18
        print(f'Loading {self.arch} with pretrained={self.pretrained}')
        print(f'Torch seed {torch.random.initial_seed()}')
        rnmodel = basemodel(pretrained=self.pretrained)
        rnmodel = torch.nn.Sequential(*list(rnmodel.children())[:-1])
        for param in rnmodel.parameters():
            param.requires_grad = True

        rnmodel.avgpool = torch.nn.AdaptiveAvgPool2d(1) 
        self.rnmodel = rnmodel
        
        # configure final fully connected layer:
        print('linear classifier layer: randomly initialized')
        linear_classifier = torch.nn.Linear(feat_dim, self.num_classes, bias=True)
        self.linear_classifier = linear_classifier

    def forward(self, x):
        # x = batch['image']
        feats = self.rnmodel(x)
        logits = self.linear_classifier(torch.squeeze(feats))
        return logits 

class MyCNNNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv11 = nn.Conv2d(3, 5, 7)
        self.conv12 = nn.Conv2d(5, 5, 5)
        self.bn1 = nn.BatchNorm2d(5)

        self.conv21 = nn.Conv2d(5, 4, 5)
        self.conv22 = nn.Conv2d(4, 4, 3)
        self.bn2 = nn.BatchNorm2d(4)

        self.conv31 = nn.Conv2d(4, 2, 5)
        self.conv32 = nn.Conv2d(2, 2, 3)
        self.bn3 = nn.BatchNorm2d(2)

        self.conv41 = nn.Conv2d(2, 1, 5)
        self.conv42 = nn.Conv2d(1, 1, 3)
        self.bn4 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(2116, 300)
        self.fc2 = nn.Linear(300, 150)
        self.bnf1 = nn.BatchNorm1d(150)
        self.fc3 = nn.Linear(150, 60)
        self.fc4 = nn.Linear(60, 30)
        self.bnf2 = nn.BatchNorm1d(30)
        self.fc5 = nn.Linear(30, 2, bias=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv12(F.relu(self.conv11(x)))))
        x = self.pool(self.bn2(F.relu(self.conv22(F.relu(self.conv21(x))))))
        x = self.bn3(F.relu(self.conv32(F.relu(self.conv31(x)))))
        x = self.pool(self.bn4(F.relu(self.conv42(F.relu(self.conv41(x))))))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        # exit()
        x = F.relu(self.fc1(x))
        x = self.bnf1(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.bnf2(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

class MyCNNNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 4, 3)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 2, 3)
        self.bn3 = nn.BatchNorm2d(2)
        # self.conv4 = nn.Conv2d(3, 2, 5)
        # self.bn4 = nn.BatchNorm2d(2)
        self.conv5 = nn.Conv2d(2, 1, 3)
        self.bn5 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(625, 300)
        self.fc2 = nn.Linear(300, 150)
        self.bn6 = nn.BatchNorm1d(150)
        self.fc3 = nn.Linear(150, 60)
        self.bn7 = nn.BatchNorm1d(60)
        self.fc4 = nn.Linear(60, 30)
        self.fc5 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        # x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.pool(self.bn5(F.relu(self.conv5(x))))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        # exit()
        x = F.relu(self.fc1(x))
        x = self.bn6(F.relu(self.fc2(x)))
        x = self.bn7(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class MyCNNNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 4, 3)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 2, 3)
        self.bn3 = nn.BatchNorm2d(2)
        # self.conv4 = nn.Conv2d(3, 2, 5)
        # self.bn4 = nn.BatchNorm2d(2)
        self.conv5 = nn.Conv2d(2, 1, 3)
        self.bn5 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(144, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        # x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.pool(self.bn5(F.relu(self.conv5(x))))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        # exit()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyCNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 5)
        self.bn1 = nn.BatchNorm2d(2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(2, 1, 5)
        self.bn3 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(1, 1, 5)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        # exit()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, xData,yData,tx):
        self.xData = xData
        self.yData = yData
        self.tx = tx

    def __len__(self):
        return len(self.xData)

    def __getitem__(self, idx):
        # print(self.xData[idx].shape)
        x = self.xData[idx] if self.tx is None else self.tx(self.xData[idx])
        y = self.yData[idx]
        # print(x.shape,y.shape)
        return x,y
