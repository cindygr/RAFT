import sys
sys.path.append('core')

import argparse
import os
import glob
from PIL import Image, ImageFilter


def load_image(imfile, start_x=-1, start_y=-1, width=620, height=880, scale=1.0):
    img_full = Image.open(imfile)
    print(f"Image size {img_full.width} {img_full.height}")

    if width == -1:
        width = img_full.width
    if height == -1:
        height = img_full.height
    if start_x == -1:
        start_w = (img_full.width - width) // 2
    else:
        start_w = start_x
    if start_y == -1:
        start_h = (img_full.height - height) // 2
    else:
        start_h = start_y
    print(f"Start {start_w} {start_h} size {width} {height}")
    img_crop = img_full.crop((start_w, start_h, start_w + width, start_h + height))
    if scale == 1.0:
        img_scl = img_crop
    else:
        scl_size_orig = (int(img_crop.width * scale), int(img_crop.height * scale))
        scl_size = (scl_size_orig[0] - scl_size_orig[0] % 2, scl_size_orig[1] - scl_size_orig[1] % 2)
        img_scl = img_crop.resize(scl_size)
        print(f"Scaled size {scl_size}")
    return img_scl


def copy_images(in_folder, out_folder, start_index, end_index, skip_index, start_x, start_y, width, height, scale, filter):
    images = glob.glob(os.path.join(in_folder, 'rgb*.png')) + \
             glob.glob(os.path.join(in_folder, 'rgb*.jpg'))

    images = sorted(images)
    for indx, imfile in enumerate(images[start_index:end_index:skip_index]):
        image = load_image(imfile, start_x=start_x, start_y=start_y, width=width, height=height, scale=scale)
        for n in range(0, filter):
            image = image.filter(filter=ImageFilter.SHARPEN)

        fname = out_folder + f"/im{indx:03d}{imfile[-4:]}"
        print(f"In: {imfile} out {fname}")
        image.save(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', default=220, type=int, help="start index")
    parser.add_argument('--end_index', default=220+ 60, type=int, help="end index")
    parser.add_argument('--skip_index', default=10, type=int, help="skip")
    parser.add_argument('--start_width', default=-1, type=int, help="crop image start x value")
    parser.add_argument('--start_height', default=500, type=int, help="crop image start y value 0 is top")
    parser.add_argument('--width', default=620, type=int, help="crop image width")
    parser.add_argument('--height', default=440, type=int, help="crop image height")
    parser.add_argument('--scale', default=1.25, type=float, help="Scale image after crop")
    parser.add_argument('--path', default="../data/bush_8_west/", type=str, help="where to grab images from")
    parser.add_argument('--dest', default="./demo-blues/", type=str, help='destination folder')
    parser.add_argument('--filter', default=1, type=int, help='number of times to filter')
    args = parser.parse_args()

    copy_images(in_folder=args.path, out_folder=args.dest,
                start_index=args.start_index, end_index=args.end_index, skip_index=args.skip_index,
                start_x=args.start_width, start_y=args.start_height,
                width=args.width, height=args.height, scale=args.scale, filter=args.filter)


