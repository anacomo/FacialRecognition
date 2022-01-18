from generate_task1 import *
from facialdetector import *
from params_task1 import *
from util_task1 import *
from visualize_task1 import *

def main():
    # * GENEREZ EXEMPLE POZITIVE SI NEGATIVE
    generate_if_necesarry(IMG_SIZE)
    # * PARAMETERS
    if not os.path.exists(SAVED):
        os.mkdir(SAVED)
    # * FACIAL DETECTOR
    fd : FacialDetector = FacialDetector()
    # * positive features
    positive_features_path = os.path.join(SAVED, 'descriptori_exemple_pozitive.npy')
    # * analyze features only if necessary
    if os.path.exists(positive_features_path):
        positive_features = np.load(positive_features_path)
        print('Incarcat descriptori pozitive')
    else:
        print('Construiesc descriptori pozitive - Resnet')
        positive_features = fd.get_positive_descriptors_resnet()
        np.save(positive_features_path, positive_features)

    # * negative features
    negative_features_path = os.path.join(SAVED, 'descriptori_exemple_negative.npy')
    if os.path.exists(negative_features_path):
        negative_features = np.load(negative_features_path)
        print('Incarcat descriptori negative')
    else:
        print('Construiesc descriptori negative - Resnet')
        negative_features = fd.get_negative_descriptors_resnet()
        np.save(negative_features_path, negative_features)

    # * clasificator
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
    fd.train_classifier(training_examples, train_labels)

    detections, scores, file_names = fd.new_run()
    
    if not os.path.exists(MY_DIR):
        os.mkdir(MY_DIR)
    if not os.path.exists(SAVE_SOLUTION_TASK1):
        os.mkdir(SAVE_SOLUTION_TASK1)

    np.save(SAVE_SOLUTION_TASK1 + 'detections_all_faces.npy', detections)
    np.save(SAVE_SOLUTION_TASK1 + 'scores_all_faces.npy', scores)
    np.save(SAVE_SOLUTION_TASK1 + 'file_names_all_faces.npy', file_names)

    if ANNOT:
        eval_detections(TEST_ANNOT, SAVED, detections, scores, file_names)
        # show_detections_with_ground_truth(detections, scores, file_names)
    else:
        show_detections_without_ground_truth(detections, scores, file_names)
        pass


if __name__ == '__main__':
    main()