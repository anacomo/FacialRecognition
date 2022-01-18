import os
import numpy as np
import cv2 as cv
import random
# * from other files
from params_task1 import TASK1
from util_task1 import intersection_over_union
from params_task1 import TRAIN, POSITIVE, NEGATIVE

# * GENERATE POSITIVE EXAMPLES
def generate_positives(size : int):
    print('Generez pozitive...')
    names  = ["bart", "homer", "lisa", "marge"]
    image_names = []
    bboxes = []
    characters = []
    nb_examples = 0

    for name in names:
        filename_annotations = TRAIN + name + ".txt"
        f = open(filename_annotations)
        for line in f:
            a = line.split(os.sep)[-1]
            b = a.split(" ")
            
            image_name = TRAIN + name + "/" + b[0]
            bbox = [int(b[1]),int(b[2]),int(b[3]),int(b[4])]
            character = b[5][:-1]
            
            image_names.append(image_name)
            bboxes.append(bbox)
            characters.append(character)
            nb_examples = nb_examples + 1

    for idx, img_name in enumerate(image_names):
        img = cv.imread(img_name)
        bbox = bboxes[idx]
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        face = img[ymin:ymax,xmin:xmax]
        face_warped = cv.resize(face,(size, size))
        filename = POSITIVE + str(idx) + ".jpg"
        cv.imwrite(filename,face_warped)


# * GENERATE NEGATIVE EXAMPLES
def generate_negatives(size: int):
    print('Generez negative...')
    names  = ["bart", "homer", "lisa", "marge"]

    image_names = []
    bboxes = []
    characters = []
    nb_examples = 0

    for name in names:
        filename_annotations = TRAIN + name + ".txt"
        f = open(filename_annotations)
        for line in f:
            a = line.split(os.sep)[-1]
            b = a.split(" ")
            
            image_name = TRAIN + name + "/" + b[0]
            bbox = [int(b[1]),int(b[2]),int(b[3]),int(b[4])]
            character = b[5][:-1]
            
            image_names.append(image_name)
            bboxes.append(bbox)
            characters.append(character)
            nb_examples = nb_examples + 1
    
    current_boxes = []

    for idx, img_name in enumerate(image_names):
        img = cv.imread(img_name)
        num_rows, num_cols = img.shape[0], img.shape[1]
        min_size = min(num_rows, num_cols)
        current_boxes.append(bboxes[idx])

        if idx < len(image_names) - 1 and image_names[idx] != image_names[idx + 1]:
            ious = [0.05, 0.1, 0.3]
            for max_iou in ious:
                for i in range(2):
                    overall_iou = 1
                    while overall_iou >= max_iou:
                        # * generez un nou patch
                        patch_size = random.randint(30, min_size)
                        x = random.randint(0, img.shape[1] - patch_size)
                        y = random.randint(0, img.shape[0] - patch_size)
                        # * verific scorul de intersection over union
                        # forma: xmin ymin xmax ymax
                        patch_box = [x, y, x + patch_size, y + patch_size]
                        # * current boxes contine fetele curente cu care compar box-ul
                        overall_iou = 1
                        for box in current_boxes:
                            iou = intersection_over_union(patch_box, box)
                            overall_iou = min(iou, overall_iou)
                        if overall_iou < max_iou:
                            # * random patch box
                            pxmin, pymin, pxmax, pymax = patch_box[0], patch_box[1], patch_box[2], patch_box[3]
                            negative_example = img[pymin : pymax, pxmin : pxmax]
                            negative_warped = cv.resize(negative_example,(size, size))
                            filename = NEGATIVE + str(idx) + "_" + str(i) + '_' + str(int(max_iou * 10)) + ".jpg"
                            cv.imwrite(filename, negative_warped)
            current_boxes = []

# * functia de generare
def generate_if_necesarry(size: int):
    # * creeaza foldere
    dirs = [TASK1, POSITIVE, NEGATIVE]
    for DIR in dirs:
        if not os.path.exists(DIR):
            os.mkdir(DIR)

    # * genereaza pozitivele daca nu exista deja
    if not os.listdir(POSITIVE):
        generate_positives(size)
    if not os.listdir(NEGATIVE):
        generate_negatives(size)