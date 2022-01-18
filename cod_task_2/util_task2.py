from posixpath import splitdrive
from cv2 import sort
from torch.utils.data import Dataset
from PIL import Image
from params_task2 import TRANSFORM_IMG


#  ? INTERSECTION OVER UNION
def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou
    
# * nub function -> get uniques from list
def nub(mylist):
    return list(set(mylist))

def same_center(bbox_a, bbox_b):
    ca_x = (bbox_a[0] + bbox_a[2]) // 2
    ca_y = (bbox_a[0] + bbox_a[2]) // 2
    if bbox_b[0] <= ca_x <= bbox_b[2] and bbox_b[1] <= ca_y <= bbox_b[3]:
        return True

# ? COMPUTE NON MAXIMAL SUPRESSION
def non_maximal_suppression2(image_detections, image_scores, image_labels):
    image_labels = [x.item() for x in image_labels]
    to_return = []
    iou_threshold = 0.3
    labels_set = nub(image_labels)
    for label in labels_set:
        best_bboxes = []
        zipall = list(zip(image_detections, image_scores, image_labels))
        # * filtrez dupa label
        filtered = list(filter(lambda x : x[2] == label , zipall))
        # * sortez dupa score
        sorted_data = sorted(filtered, key=lambda x: x[1], reverse=True)
        # * pastrez cele mai bune bounding boxes
        best_bboxes.append(sorted_data[0])
        if(label < 4):
            to_return += best_bboxes
            continue
        sorted_data.pop()
        for bbox in sorted_data:
            should_remove = False
            for best_box in best_bboxes:
                if intersection_over_union(bbox[0], best_box[0]) > iou_threshold or same_center(bbox[0], best_box[0]):
                    should_remove = True
            if should_remove == False:
                best_bboxes.append(bbox)
        to_return += best_bboxes
    unzipped = list(zip(*to_return))
    print('TO RETURN', unzipped)
    if len(unzipped) != 0:
        return unzipped[0], unzipped[1], unzipped[2]
    return None



label_map = {
    'bart' : 0,
    'homer' : 1,
    'lisa' : 2,
    'marge' : 3,
    'unknown' : 4,
    'negative' : 5,
}

class LabeledDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.paths[idx]).convert("RGB"))
        label = self.paths[idx].split('/')[-2]
        label = label_map[label]
        return img, label
        

# * for test
class UnlabeledDataset(Dataset):
    def __init__(self, img_path, transform):
        image = Image.open(img_path).convert("RGB")
        num_cols, num_rows = image.size
        min_size = min(num_cols, num_rows)
        self.images = []

        patch_sizes = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        patch_sizes = [int(x * min_size) for x in patch_sizes]
        
        for patch_len in patch_sizes:
            for y in range(0, num_rows - patch_len, 5):
                for x in range(0, num_cols - patch_len, 5):
                    subimg = transform(image.crop((x, y, x + patch_len, y + patch_len)))
                    self.images.append((subimg, x, y, x + patch_len, y + patch_len))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]