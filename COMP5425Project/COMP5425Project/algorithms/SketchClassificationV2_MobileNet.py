from PIL import Image, ImageOps
import numpy as np

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
import pickle
import os


class MobileNetSketchClassifier():
    def __init__(self, num_class = 340, input_size = 64):

        self.model = MobileNet(input_shape = (input_size, input_size, 1),
                               weights = None,
                               alpha = 1,
                               classes = num_class)

        self.num_class = num_class
        self.input_size = input_size

        def criterion_acc_top_3(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=3)

        self.model.compile(optimizer=Adam(lr=0.001),
                           loss='categorical_crossentropy',
                           metrics=[criterion_acc_top_3,
                                    categorical_crossentropy,
                                    categorical_accuracy,
                                    ]
                           )

        # load existing model weights
        self.model.load_weights('./data/MobileNet.h5')
        
        with open("./data/idx_to_class_340.pkl", 'rb') as f:
            self.class_idx_dict = pickle.load(f)
            self.idx_class_dict = dict((v,k) for k,v in self.class_idx_dict.items())

        self.model.summary()

    def Predict(self, image):
        pred = self.model.predict(image)
        idx = np.argmax(pred)
        return pred

    def GetLabel(self, idx):
        return self.idx_class_dict[idx]

    def Transform(self, image_pil):
        image_pil = ImageOps.invert(image_pil)#.transpose(Image.ROTATE_180)
        image_pil = image_pil.resize((self.input_size, self.input_size))
        image = np.expand_dims(np.asarray(image_pil)[:, :, np.newaxis], axis = 0)
        return image

    def TransformFromPath(self, image_path):
        image_pil = Image.open(image_path).convert('L')
        return self.Transform(image_pil)

def GetTrainedClassifier():
    mkc = MobileNetSketchClassifier()
    
    return mkc, mkc.idx_class_dict, mkc.class_idx_dict


if __name__ == "__main__":

    #classes_path = os.listdir(r"C:\Users\Sun\Desktop\train_simplified")
    #classes_path = sorted(classes_path, key=lambda s: s.lower())
    #class_dict = {x[:-4].replace(" ", "_"):i for i, x in enumerate(classes_path)}

    #import pickle
    #with open("./data/idx_to_class_340.pkl", 'wb') as f:
    #    pickle.dump(class_dict, f)


    image_dir = r"C:\Users\Sun\Desktop\key.png"
    mkc = MobileNetSketchClassifier()
    image = mkc.TransformFromPath(image_dir)
    pred = mkc.Predict(image)
    lab = mkc.GetLabel(np.argmax(pred))
    print("Predicted: {}".format(lab))


