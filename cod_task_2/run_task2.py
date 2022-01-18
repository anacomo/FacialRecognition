from facialrecognition import task2_results
from network import Network
from generate_task2 import *
from params_task2 import *

def main():
    # * generez 5 fete + negative
    # generate_faces(IMG_SIZE)
    if not os.path.exists(MODEL_PATH):
        # * train neural network
        network = Network()
        network.load_network()
        network.train_network(epochs=EPOCHS)
        network.save_model()

    # * submit
    results = task2_results()
    # * detections
    det_bart = []
    det_homer = []
    det_lisa = []
    det_marge = []
    # * file names
    files_bart = []
    files_homer = []
    files_lisa = []
    files_marge = []
    # * scores
    scores_bart = []
    scores_homer = []
    scores_lisa = []
    scores_marge = []

    for face in results:
        # ? detection, score, label, file
        # 0 for bart
        if face[2] == 0:
            det_bart.append(face[0])
            files_bart.append(face[3])
            scores_bart.append(face[1])
        # 1 for homer
        elif face[2] == 1:
            det_homer.append(face[0])
            files_homer.append(face[3])
            scores_homer.append(face[1])
        # 2 for lisa
        elif face[2] == 2:
            det_lisa.append(face[0])
            files_lisa.append(face[3])
            scores_lisa.append(face[1])
        # 3 for marge
        elif face[2] == 3:
            det_marge.append(face[0])
            files_marge.append(face[3])
            scores_marge.append(face[1])

    if not os.path.exists(SAVE_SOLUTION_TASK2):
        os.mkdir(SAVE_SOLUTION_TASK2)

    np.save(DET_BART, np.array(det_bart))
    np.save(DET_HOMER, np.array(det_homer))
    np.save(DET_LISA, np.array(det_lisa))
    np.save(DET_MARGE, np.array(det_marge))

    np.save(FILE_BART, np.array(files_bart))
    np.save(FILE_HOMER, np.array(files_homer))
    np.save(FILE_LISA, np.array(files_lisa))
    np.save(FILE_MARGE, np.array(files_marge))

    np.save(SCORES_BART, np.array(scores_bart))
    np.save(SCORES_HOMER, np.array(scores_homer))
    np.save(SCORES_LISA, np.array(scores_lisa))
    np.save(SCORES_MARGE, np.array(scores_marge))


if __name__ == '__main__':
    main()