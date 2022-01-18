from params_task2 import MODEL_PATH
import torch
import numpy as np
from params_task2 import *
from util_task2 import *
import os
from tqdm import tqdm

# * face patches
def face_patches(path, model):
    # * gets detections, scores and filenames
    detections = []
    scores = []
    labels = []

    img_dataset = UnlabeledDataset(path, transform=TRANSFORM_IMG)
    test_loader = torch.utils.data.DataLoader(img_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for imgs, xmin, ymin, xmax, ymax in test_loader:
            imgs = imgs.cuda()
            predictions = model.forward(imgs)
            pred_size = len(predictions)

            for i in range(pred_size):
                coords = (int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i]))
                label = np.argmax(predictions[i].cpu())
                if label < 4:
                    detections.append(coords)
                    scores.append(float(predictions[i][label]))
                    labels.append(label)
    return non_maximal_suppression2(detections, scores, labels)


def task2_results():
    model = torch.load(MODEL_PATH)
    model.eval()

    res = []
    files = os.listdir(TEST)[:]
    for file in tqdm(files):
        x = face_patches(TEST + file, model)
        if x:
            det, sc, lab = x
            for idx, d in enumerate(det):
                res.append((d, sc[idx], lab[idx], file))
    return res