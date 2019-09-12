import torch.nn as nn
import torch
from torchvision import models

class AlexNet(nn.Module):
    def __init__(self, model_type):
        super(AlexNet, self).__init__()
        self.model_type = model_type
        if model_type == 'pre':
            model = models.alexnet(pretrained=True)
            self.features = model.features
            fc = nn.Linear(256, 4096)
            #fc.bias = model.classifier[1].bias
            #fc.weight = model.classifier[1].weight

            self.classifier = nn.Sequential(
                    #nn.Dropout(),
                    fc)
                    #nn.ReLU(inplace=True))

        if model_type == 'new':
            self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 11, 4, 2),
                    nn.ReLU(inplace = True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(64, 192, 5, 1, 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(192, 384, 3, 1, 1),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(384, 256, 3, 1, 1),
                    nn.ReLU(inplace=True))
                    #nn.MaxPool2d(3, 2, 0))
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(1024, 4096),
                    nn.ReLU(inplace=True))
            
    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        out  = self.classifier(x)
        return out


class O3N(nn.Module):
    def __init__(self, model_type, output_size):
        super(O3N, self).__init__()
        #self.alexnet = AlexNet(model_type)
        #self.classifier = nn.Sequential(
                    #nn.Dropout(),
                    #nn.Linear(4096, 128),
                    #nn.ReLU(inplace=True),
                    #nn.Dropout(),
                    #nn.Linear(128, output_size))
                    #nn.ReLU(inplace=True))
        self.gcn = GCN(3, output_size, {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, True)
        self.fcn = nn.Conv2d(256, output_size, kernel_size=1)
        self.video_num = output_size


    def forward(self, input):
        #print(input.shape)
        shape = input.shape
        input = input.view([shape[0] * shape[1], *shape[2:4], shape[4] * shape[5]])
        #input = input.permute(0, 4, 1, 2, 3).contiguous()
        #input = input.view([-1, shape[2], shape[3], shape[4]])
        #print(input.shape)
        Xs = self.alexnet(input)
        #Xs = Xs.view([shape[0] * shape[1], shape[5], -1]).mean(dim=1).squeeze()
<<<<<<< HEAD
        Xs = Xs.view([shape[0], shape[1], -1])
        #print(Xs.shape)
        X = self.Fusion(Xs)
        #print(X.shape)
        X = self.classifier(X)
=======
        _, Xs = self.gcn.extract_feature(input)
        Xs = F.avg_pool2d(Xs, Xs.size()[2:])
        Xs = Xs.view(shape[0] * shape[1], shape[5], -1).mean(dim=1)
        Xs = Xs.view([shape[0], shape[1], -1])
        #print(Xs.shape)
        X = self.Fusion(Xs)
        X = self.fcn(X)
        #print(X.shape)
        #X = self.classifier(X)
        X = X.view(shape[0], -1)
>>>>>>> remotes/origin/master
        return X

    def Fusion(self, Xs):
        Xs = Xs.permute(1, 0, 2).contiguous()
        X = torch.zeros(Xs[0].shape).cuda()
        for j in range(self.video_num):
            for i in range(j):
                X += Xs[j] - Xs[i]
        return X
