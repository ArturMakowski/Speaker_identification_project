import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(512, 128), # 6*4*4x256
                                nn.PReLU(),
                                nn.Linear(128, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
    
class SiameseEmbeddingModel(nn.Module):
    def __init__(self):
        super(SiameseEmbeddingModel, self).__init__()

        # CONVOLUTIONAL BLOCK I
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        # CONVOLUTIONAL BLOCK II
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        # CONVOLUTIONAL BLOCK III
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        # CONVOLUTIONAL BLOCK IV
        self.conv4 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # FULLY CONNECTED V
        self.fc5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.GroupNorm(32, 2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # FULLY CONNECTED VI
        self.fc6 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GroupNorm(32, 1024),
            nn.ReLU()
        )

        # Triplet Layer
        self.triplet = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Flatten()
        )

        # Embedding Model Output
        self.embedding_model = nn.Sequential(
            nn.Linear(512, 128, bias=True),
            nn.GroupNorm(32, 128),
            nn.ReLU() 
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.triplet(x)
        x = self.embedding_model(x)
        x = F.normalize(x, p=2, dim=1)
        return x

        
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    
if __name__ == "__main__":
    model2 = SiameseEmbeddingModel()
    summary(model2, (1, 64, 79))