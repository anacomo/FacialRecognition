import os
import glob
import numpy as np
import cv2 as cv
import pickle
from copy import deepcopy
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import ntpath
import timeit
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

# * other files
from params_task1 import *
from util_task1 import *

class FacialDetector:
    def __init__(self,) -> None:
        self.best_model = None

    # ? RESNET POSITIVE DESCRIPTORS
    def get_positive_descriptors_resnet(self):
        images_path = os.path.join(POSITIVE, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        model = models.resnet18(pretrained=True)#.cuda()
        layer = model._modules.get('avgpool')
        model.eval()
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        scaler = transforms.Resize((IMG_SIZE, IMG_SIZE))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        for i in range(num_images):
            # print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i])
            img = Image.fromarray(img)
            t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))#.cuda()
            my_embedding = torch.zeros(512)
            def copy_data(m, i, o):
                my_embedding.copy_(o.data.reshape(o.data.size(1)))
            h = layer.register_forward_hook(copy_data)
            model(t_img)
            h.remove()
            my_embedding = my_embedding.numpy()
            positive_descriptors.append(my_embedding)
        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    # ? RESNET NEGATIVE DESCRIPTORS
    def get_negative_descriptors_resnet(self):
        images_path = os.path.join(NEGATIVE, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []
        model = models.resnet18(pretrained=True)#.cuda()
        layer = model._modules.get('avgpool')
        model.eval()
        print('Calculam descriptorii pt %d imagini negative...' % num_images)
        scaler = transforms.Resize((IMG_SIZE, IMG_SIZE))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        for i in range(num_images):
            # print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i])
            img = Image.fromarray(img)
            t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))#.cuda()
            my_embedding = torch.zeros(512)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data.reshape(o.data.size(1)))

            h = layer.register_forward_hook(copy_data)
            model(t_img)
            h.remove()
            my_embedding = my_embedding.numpy()
            negative_descriptors.append(my_embedding)
        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def new_run(self):
        test_images_path = os.path.join(TEST, '*.jpg')
        threshold=-1
        test_files = glob.glob(test_images_path)
        detections = None 
        scores = np.array([])
        file_names = np.array([]) 
        num_test_images = len(test_files)
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0] 
        model = models.resnet18(pretrained=True)#.cuda()
        layer = model._modules.get('avgpool')
        model.eval()
        scaler = transforms.Resize((IMG_SIZE, IMG_SIZE))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()

        low_yellow = (15, 105, 105)
        high_yellow = (90, 255, 255)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i])
            num_rows, num_cols = img.shape[0], img.shape[1]
            img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            mask = cv.inRange(img_hsv, low_yellow, high_yellow)

            original_image = img.copy()
            image_scores = []
            image_detections = []
            
            max_square_size = min(num_rows, num_cols)
            patch_sizes = [round(i * max_square_size) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

            for patch_size in patch_sizes:
                for y in range(0, num_rows - patch_size, 10):
                    for x in range(0, num_cols - patch_size, 10):
                        mask_patch = mask[y : y + patch_size, x : x + patch_size]
                        no_zero = cv.countNonZero(mask_patch)

                        if no_zero > (patch_size ** 2) / 2:
                            bbox_curent = [x, y, x + patch_size, y + patch_size]
                            xmin, ymin, xmax, ymax = bbox_curent[0], bbox_curent[1], bbox_curent[2], bbox_curent[3]	
                            img_patch = img[ymin:ymax, xmin:xmax]

                            img_patch = Image.fromarray(img_patch)
                            t_img = Variable(normalize(to_tensor(scaler(img_patch))).unsqueeze(0))#.cuda()
                            my_embedding = torch.zeros(512)

                            def copy_data(m, i, o):
                                my_embedding.copy_(o.data.reshape(o.data.size(1)))

                            h = layer.register_forward_hook(copy_data)
                            model(t_img)
                            h.remove()
                            descr = my_embedding.numpy()
                            score = np.dot(descr, w)[0] + bias

                            if score > threshold:
                                image_detections.append(bbox_curent)
                                image_scores.append(score)

            if len(image_scores) > 0:
                s = timeit.default_timer()
                image_detections, image_scores = non_maximal_suppression(np.array(image_detections),
                                                                            np.array(image_scores),
                                                                            original_image.shape)
                e = timeit.default_timer()
                print ('nms sec', e - s)
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))

                scores = np.append(scores, image_scores)
                short_file_name = ntpath.basename(test_files[i])
                image_names = [short_file_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'% (i, num_test_images, end_time - start_time))
        return detections, scores, file_names


    def train_classifier(self, training_examples, train_labels, ignore_restore=True):
        svm_file_name = os.path.join(SAVED, 'best_model')
        if os.path.exists(svm_file_name) and ignore_restore:
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy , best_c, best_model  = 0, 0, None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2]
        for c in Cs:
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()