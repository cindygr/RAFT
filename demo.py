import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image, ImageFilter

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img_full = np.array(Image.open(imfile)).astype(np.uint8)
    img_full = torch.from_numpy(img_full).permute(2, 0, 1).float()
    print(f"Image size {img_full.shape}")
    return img_full[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        print(f"Images: {len(images)}")
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print(f"low {flow_low.shape}, {flow_up.shape}")
            viz(image1, flow_up)

            flow_img = flow_up[0].permute(1, 2, 0).cpu().numpy()

            # map flow to rgb image
            flo = flow_viz.flow_to_image(flow_img)
            print(f"flow_img {flow_img.shape} {flo.shape}")

            flow_img_write = Image.fromarray(flo)
            # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)

            str_fname = imfile1.split('/')
            n_index = str_fname[-1][-6:]
            str_flow_name = "./" + str_fname[-2] + "/CalculatedData/flow_" + n_index
            print(f"image name {imfile1} flow name {str_flow_name} width {flow_img_write.width} {flow_img_write.height}")
            flow_img_write.save(str_flow_name)


            im_flow1_cv2 = cv2(flo[0,:,:].squeeze())
            im_flow2_cv2 = cv2(flo[0,:,:].squeeze())
            # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            image_edge1 = cv2.Canny(im_flow1_cv2, 50, 150, apertureSize=3)
            image_edge2 = cv2.Canny(im_flow2_cv2, 50, 150, apertureSize=3)

            str_edge1_name = "./" + str_fname[-2] + "/CalculatedData/edge1_" + n_index
            str_edge2_name = "./" + str_fname[-2] + "/CalculatedData/edge2_" + n_index
            cv2.imwrite(str_edge1_name, image_edge1)
            cv2.imwrite(str_edge2_name, image_edge2)
            #edge_img_write = flow_img_write.filter(filter=ImageFilter.FIND_EDGES)
            #edge_img_write.save(str_edge_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
