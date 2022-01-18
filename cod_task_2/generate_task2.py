from params_task2 import *
import os
import cv2 as cv
import shutil
import random

def generate_faces(size: int):
    names = ['bart', 'homer', 'lisa', 'marge', 'unknown']

    # * creez foldere daca nu exista
    if not os.path.exists(DATASET):
        os.mkdir(DATASET)
    for DIR in [TRAIN_DS, VAL_DS]:
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        for name in names:
            if not os.path.exists(DIR + name):
                os.mkdir(DIR + name)
    
    faces = {'bart': [], 'homer': [], 'lisa': [], 'marge': [], 'unknown': []}
    bboxes = {'bart': [], 'homer': [], 'lisa': [], 'marge': [], 'unknown': []}

    # * generez train
    for name in names[ : -1]:
        annot = TRAIN + name + '.txt'
        f = open(annot)
        for line in f:
            info = line.split()
            image_name = TRAIN + name + '/' + info[0]
            bbox = [int(info[1]), int(info[2]), int(info[3]), int(info[4])]
            character = info[5]
            faces[character].append(image_name)
            bboxes[character].append(bbox)
    
    for name in names:
        for idx, img_name in enumerate(faces[name]):
            img = cv.imread(img_name)
            [xmin, ymin, xmax, ymax] = bboxes[name][idx]
            face = img[ymin : ymax, xmin : xmax]
            face_warped = cv.resize(face, (size, size))
            file_save = TRAIN_DS + name + '/' + name + '_' + str(idx) + '.jpg'
            cv.imwrite(file_save, face_warped)

    faces = {'bart': [], 'homer': [], 'lisa': [], 'marge': [], 'unknown': []}
    bboxes = {'bart': [], 'homer': [], 'lisa': [], 'marge': [], 'unknown': []}

    # * generez validare
    f = open(VAL_ANNOT)
    for line in f:
        info = line.split()
        name = info[5]
        image_name = VAL + info[0]
        bbox = [int(info[1]), int(info[2]), int(info[3]), int(info[4])]
        character = info[5]
        faces[character].append(image_name)
        bboxes[character].append(bbox)

    for name in names:
        for idx, img_name in enumerate(faces[name]):
            img = cv.imread(img_name)
            [xmin, ymin, xmax, ymax] = bboxes[name][idx]
            face = img[ymin : ymax, xmin : xmax]
            face_warped = cv.resize(face, (size, size))
            file_save = VAL_DS + name + '/' + name + '_' + str(idx) + '.jpg'
            cv.imwrite(file_save, face_warped)

    # * adaug negative
    DESTS = [TRAIN_DS + 'negative/', VAL_DS + 'negative/']

    for DIR in (DESTS):
        if not os.path.exists(DIR):
            os.mkdir(DIR)

    src_imgs = os.listdir(NEGATIVE)
    random.shuffle(src_imgs)
    tr_nr = int(0.8 * len(src_imgs))
    # * pun imagin in train
    for file_name in src_imgs[ : tr_nr]:
        src = NEGATIVE + file_name
        dest = DESTS[0]
        if os.path.isfile(src):
            shutil.copy(src, dest)
    # * pun restul imaginilor in val
    for file_name in src_imgs[tr_nr : ]:
        src = NEGATIVE + file_name
        dest = DESTS[1]
        if os.path.isfile(src):
            shutil.copy(src, dest)