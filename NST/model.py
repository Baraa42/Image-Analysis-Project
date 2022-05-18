import torch.nn as nn
import torchvision.models as models


vgg = models.vgg19(pretrained=True).features
features = ["0", "5", "10", "19", "28"]


class VGG(nn.Module):
    def __init__(self, features=features):
        super(VGG, self).__init__()

        self.chosen_features = features
        self.model = vgg[:29]

    def forward(self, x):
        features = []
        out = x.clone()
        for layer_num, layer in enumerate(self.model):
            out = layer(out)

            if str(layer_num) in self.chosen_features:
                features.append(out)

        return features
