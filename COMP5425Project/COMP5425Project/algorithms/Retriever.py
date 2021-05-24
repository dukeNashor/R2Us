import os
import sys
from operator import itemgetter

sys.path.insert(0, "../")

import numpy as np
from utils import GetBoxImage, PlotBoxes
from utils import COCO_INSTANCE_CATEGORY_NAMES as coco_labels
from numpy import linalg

class Retriever():
    def __init__(self, data_dict, idx_to_class):
        self.data_dict = data_dict
        self.idx_to_class = idx_to_class
        self.CalculateCenters()
        self.aspect_ratio = np.array([600, 450])

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

        final_result = []
        # negative logarithm distance of scaled centers
        for _ , imp in list_candidates:
            t = self.data_dict[imp]
            tls = self.data_dict[imp]["labels"]
            qt, q_ind, t_ind = np.intersect1d(qls, tls, return_indices = True)
            print("in {}".format(imp))

            score = 0.0

            for i in range(len(q_ind)):
                l = qls[q_ind[i]]
                label = self.idx_to_class[l]
                print("found {}".format(label))

                # negative logarithm distance
                center_q = query["centers"][q_ind[i], :]
                center_t = t["scaled_centers"][t_ind[i], :]
                diff = center_q - center_t
                t_aspect_ratio = np.array([t["width"], t["height"]])
                scaling_factor = np.true_divide(t_aspect_ratio, self.aspect_ratio)
                dist = np.multiply(diff, scaling_factor)
                normalized_dist = np.true_divide(dist, t_aspect_ratio)

                nl_dist_score = -1.0 * np.log(linalg.norm(normalized_dist))
                score += nl_dist_score

            final_result.append((imp, len(q_ind), score))

        # sort by number of coincident items, then by score.
        final_result = sorted(final_result, key = lambda x: (x[1], x[2]), reverse = True)

        return final_result
