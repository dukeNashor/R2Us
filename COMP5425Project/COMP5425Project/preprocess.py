import os
import sys
from PIL import Image
import numpy as np
import pickle as pkl

if __name__ == "__main__":

    # out path
    out_path = './data/val2017_preprocessed.pkl'

    if os.path.exists(out_path):
        print("Out file {} already exists, exiting.".format(out_path))
        sys.exit()

    # get image files:
    image_dir = "./data/val2017"
    image_file_names = [ os.path.join(image_dir, fn) for fn in os.listdir(image_dir) if os.path.splitext(fn)[1] == ".jpg" ]
    
    # process each file
    from algorithms.ObjectDetection import ObjectDetector

    result_dict = {}

    od = ObjectDetector()
    for fn in image_file_names:
        print("Processing #{}".format(len(result_dict)))
        image = np.asarray(Image.open(fn))
        boxes_out, labels_out, scores_out = od.Predict(image)
        #od.Plot(image, boxes_out, labels_out, scores_out)
        result_dict[fn] = {}
        result_dict[fn]["boxes"] = boxes_out
        result_dict[fn]["labels"] = labels_out
        result_dict[fn]["scores"] = scores_out

    
    with open(out_path, 'wb') as f:
        pkl.dump(result_dict, f)