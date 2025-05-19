import sys
sys.path.append('core')

import argparse
import os
import glob
from PIL import Image


def load_image(imfile, width=620, height=880):
    img_full = Image.open(imfile)
    print(f"Image size {img_full.width} {img_full.height}")

    start_w = (img_full.width - width) // 2
    start_h = (img_full.height - height) // 2
    print(f"Start {start_w} {start_h} size {width} {height}")
    img = img_full.crop((start_w, start_h, start_w + width, start_h + height))
    return img


def copy_images(in_folder, out_folder, start_index, end_index, skip_index, width, height):
    images = glob.glob(os.path.join(in_folder, 'rgb*.png')) + \
             glob.glob(os.path.join(in_folder, 'rgb*.jpg'))

    images = sorted(images)
    for indx, imfile in enumerate(images[start_index:end_index:skip_index]):
        image = load_image(imfile, width=width, height=height)

        fname = out_folder + f"/im{indx:03d}{imfile[-4:]}"
        print(f"In: {imfile} out {fname}")
        image.save(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--startindex', default=220, help="start index")
    parser.add_argument('--endindex', default=220+ 60, help="end index")
    parser.add_argument('--skipindex', default=5, help="skip")
    parser.add_argument('--width', default=620, help="crop image width")
    parser.add_argument('--height', default=440, help="crop image height")
    parser.add_argument('--path', default="../data/bush_8_west/", help="where to grab images from")
    parser.add_argument('--dest', default="./demo-blues/", help='destination folder')
    args = parser.parse_args()

    copy_images(in_folder=args.path, out_folder=args.dest,
                start_index=args.startindex, end_index=args.endindex, skip_index=args.skipindex,
                width=args.width, height=args.height)


