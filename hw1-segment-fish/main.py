import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious

LIGHT_ORANGE = [1, 165, 130]
DARK_ORANGE = [180, 255, 255]
LIGHT_WHITE = [60, 0, 200]
DARK_WHITE = [170, 160, 255]


def segment_fish(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    light_orange = np.array(LIGHT_ORANGE, dtype=np.uint8)
    dark_orange = np.array(DARK_ORANGE, dtype=np.uint8)

    light_white = np.array(LIGHT_WHITE, dtype=np.uint8)
    dark_white = np.array(DARK_WHITE, dtype=np.uint8)

    orange_mask = cv2.inRange(hsv_image, light_orange, dark_orange)
    white_mask = cv2.inRange(hsv_image, light_white, dark_white)

    combined_mask = cv2.bitwise_or(orange_mask, white_mask)
    kernel = np.ones((5, 5), np.uint8)
    morphology_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    morphology_mask = cv2.morphologyEx(morphology_mask, cv2.MORPH_OPEN, kernel)

    return morphology_mask


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
        # cv2.imshow('Original Image', img)
        # cv2.imshow('Segmented Image', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
