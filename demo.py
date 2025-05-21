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
        imfile1 = images[0]
        image1_orig = load_image(imfile1)
        str_fname = imfile1.split('/')
        str_edge_orig_name = "./" + str_fname[-2] + "/CalculatedData/edge_orig"

        # The original image, at three different filter levels
        image_orig = cv2.imread(imfile1)
        im_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
        im_edge_accum = np.zeros(im_gray.shape)
        for min_val in [100, 150, 200, 250]:
            image_edge_orig = cv2.Canny(im_gray, min_val, min_val + 100, apertureSize=3)
            im_edge_accum = im_edge_accum + image_edge_orig
            cv2.imwrite(str_edge_orig_name + f"_{min_val}.jpg", image_edge_orig)

        for imfile2 in images[1:]:
            image2_orig = load_image(imfile2)

            padder = InputPadder(image1_orig.shape)
            image1, image2 = padder.pad(image1_orig, image2_orig)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print(f"low {flow_low.shape}, {flow_up.shape}")
            viz(image1, flow_up)

            flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()

            # map flow to rgb image
            flo_img = flow_viz.flow_to_image(flow_uv)
            print(f"flow_img {flow_uv.shape} {flo_img.shape} {flow_uv.dtype}")

            flow_img_write = Image.fromarray(flo_img)
            # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)

            str_fname = imfile2.split('/')
            n_index = str_fname[-1][-6:]
            str_flow_name = "./" + str_fname[-2] + "/CalculatedData/flow_" + n_index
            print(f"image name {imfile2} flow name {str_flow_name} width {flow_img_write.width} {flow_img_write.height}")
            flow_img_write.save(str_flow_name)

            im_flow_horiz_cv2 = np.uint8(flow_uv[:,:,0].squeeze())
            im_flow_vert_cv2 = np.uint8(flow_uv[:,:,1].squeeze())
            print(f"Before canny {im_flow_horiz_cv2.shape} {im_flow_vert_cv2.shape} {im_flow_vert_cv2.dtype}")
            image_edge_horiz = cv2.Canny(im_flow_horiz_cv2, 1, 20, apertureSize=3)
            image_edge_vert = cv2.Canny(im_flow_vert_cv2, 1, 20, apertureSize=3)

            str_edge_vertical_name = "./" + str_fname[-2] + "/CalculatedData/edge_vert_" + n_index
            str_edge_horiz_name = "./" + str_fname[-2] + "/CalculatedData/edge_horiz_" + n_index
            cv2.imwrite(str_edge_vertical_name, image_edge_vert)
            cv2.imwrite(str_edge_horiz_name, image_edge_horiz)
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
