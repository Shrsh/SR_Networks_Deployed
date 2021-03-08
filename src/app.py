from flask import Flask, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms, utils
import numpy as np
import warnings
from flask import request
import io
from PIL import Image
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


warnings.filterwarnings("ignore")

# CUDA for PyTorch
print("Number of GPUs:" + str(torch.cuda.device_count()))

use_cuda = torch.cuda.is_available()
torch.no_grad()
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True

trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()


class SRSN(nn.Module):
    def __init__(self, input_dim=3, dim=64, scale_factor=4):
        super(SRSN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 9, 1, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 1, 1, 0)
        self.resnet1 = ResnetBlock(dim, 9, 1, 4, bias=True)
        self.resnet2 = ResnetBlock(dim, 7, 1, 3, bias=True)
        self.resnet3 = ResnetBlock(dim, 5, 1, 2, bias=True)
        self.resnet4 = ResnetBlock(dim, 3, 1, 1, bias=True)

        # for specifying output size in deconv filter
        #       new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
        #       new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +output_padding[1])
        self.up = torch.nn.ConvTranspose2d(64, 64, 4, stride=4)
        #         self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.conv3 = torch.nn.Conv2d(64, 16, 1, 1, 0)
        self.conv4 = torch.nn.Conv2d(16, 3, 1, 1, 0)

    def forward(self, LR):
        LR_feat = F.leaky_relu(self.conv1(LR))
        LR_feat = (F.leaky_relu(self.conv2(LR_feat)))

        ##Creating Skip connection between dense blocks
        out = self.resnet1(LR_feat)
        out = out + LR_feat

        out1 = self.resnet2(out)
        out1 = out + LR_feat + out1

        out2 = self.resnet3(out1)
        out2 = out1 + out2 + LR_feat + out

        out3 = self.resnet4(out2)
        out3 = out + out1 + out2 + out3 + LR_feat
        out3 = self.up(out3)

        #       LR_feat = self.resnet(out3)
        SR = F.leaky_relu(self.conv3(out3))
        SR = self.conv4(SR)
        #       print(SR.shape)
        return SR


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.LeakyReLU(inplace=True)
        self.act2 = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.act1(x)
        out = self.conv1(out)

        out = self.act2(out)
        out = self.conv2(out)

        return out


app = Flask(__name__)
model = SRSN()
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        super_resolved_image = get_prediction(image_bytes=img_bytes)
        super_resolved_image = super_resolved_image.detach().numpy()
        super_resolved_image = {"array": super_resolved_image}
        super_resolved_image = json.dumps(super_resolved_image, cls=NumpyArrayEncoder)
        print("Printing JSON serialized NumPy array")
        print(super_resolved_image)
        return super_resolved_image


def get_prediction(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor()])
    img = Image.open(io.BytesIO(image_bytes))
    img = my_transforms(img).unsqueeze(0)
    outputs = model.forward(img)
    return outputs


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
    print("Done")
