import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from interhand.model import Model
import torch


def test_model():
    model = Model(lr=0.001, joint_num=21)
    x = torch.zeros((1, 3, 256, 256)).cuda()
    model = model.cuda()
    out = model(x)
    joint_heatmap3d, root_depth, hand_type = out


if __name__ == "__main__":
    test_model()
