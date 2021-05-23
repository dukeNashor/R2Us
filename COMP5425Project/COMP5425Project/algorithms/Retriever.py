import os
import sys
from operator import itemgetter

sys.path.insert(0, "../")

import numpy as np
from utils import GetBoxImage, PlotBoxes
from utils import COCO_INSTANCE_CATEGORY_NAMES as coco_labels


class Retriever():
    def __init__(self, data_dict, idx_to_class):
        self.data_dict = data_dict
        self.idx_to_class = idx_to_class
        self.CalculateCenters()

    def CalculateCenters(self):

        # initialize labels
        #self.mapped_dict = {}

        #for k, v in self.idx_to_class.items():
        #    if v not in coco_labels:
        #        continue

        #    idx = coco_labels.index(v)
        #    self.mapped_dict[v] = idx
        
        #pass

        for im, o  in self.data_dict.items():
            num_o = len(o["labels"])
            centers = np.zeros((num_o, 2))
            o["centers"] = centers
            # centers of boxes
            for i, boxes in enumerate(o["boxes"]):
                centers[i, 0] = round((boxes[0] + boxes[2]) * 0.5)
                centers[i, 1] = round((boxes[1] + boxes[3]) * 0.5)
                
        # GetBoxImage(im, o["boxes"], o["labels"], o["scores"])
        pass


    def Query(self, query):

        ######### find candidates
        qls = query["labels"]

        list_candidates = []
        for imp, t in self.data_dict.items():
            tls = t["labels"]
            intersect = np.intersect1d(qls, tls, assume_unique=False, return_indices=False)
            if len(intersect) == 0:
                continue

            list_candidates.append((len(intersect), imp))

        list_candidates = sorted(list_candidates, key = itemgetter(0))
        for i in list_candidates:
            print(str(i[0]), ":", i[1])

        print("found {} candidates:".format(len(list_candidates)))

        ####### calculate distances

        # center distance

        








        pass
