import numpy as np
from torchvision import transforms

# ? task 2 specific
IMG_SIZE = 224

DATA = '../data/'
TRAIN = DATA + 'antrenare/'
TASK1 = DATA + 'task1/'
POSITIVE = TASK1 + 'exemplePozitive/'
NEGATIVE = TASK1 + 'exempleNegative/'
VAL = DATA + 'validare/simpsons_validare/'
VAL_ANNOT = DATA + 'validare/simpsons_validare.txt'

DATASET = DATA + 'task2/'
TRAIN_DS = DATASET + 'train/'
VAL_DS = DATASET + 'val/'

TEST = None
TEST_ANNOT = False

BATCH_SIZE = 64
CLASSES = 6

MODEL_PATH = DATA
EPOCHS = 20

# * SAVE DETECTIONS TASK 2
SAVE_SOLUTION_TASK2 = '../evaluare/fisiere_solutie/Comorasu_Ana-Maria_334/task2/'

DET_BART = SAVE_SOLUTION_TASK2 + 'detections_bart.npy'
DET_HOMER = SAVE_SOLUTION_TASK2 + 'detections_homer.npy'
DET_LISA = SAVE_SOLUTION_TASK2 + 'detections_lisa.npy'
DET_MARGE = SAVE_SOLUTION_TASK2 + 'detections_marge.npy'

FILE_BART = SAVE_SOLUTION_TASK2 + 'file_names_bart.npy'
FILE_HOMER = SAVE_SOLUTION_TASK2 + 'file_names_bart.npy'
FILE_LISA = SAVE_SOLUTION_TASK2 + 'file_names_bart.npy'
FILE_MARGE = SAVE_SOLUTION_TASK2 + 'file_names_bart.npy'

SCORES_BART = SAVE_SOLUTION_TASK2 + 'scores_bart.npy'
SCORES_HOMER = SAVE_SOLUTION_TASK2 + 'scores_bart.npy'
SCORES_LISA = SAVE_SOLUTION_TASK2 + 'scores_bart.npy'
SCORES_MARGE = SAVE_SOLUTION_TASK2 + 'scores_bart.npy'

# * other network constants
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

