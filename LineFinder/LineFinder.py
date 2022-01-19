import numpy as np
import cv2
from matplotlib import pyplot as plt

# io

def ReadImage(file_name):
    img = cv2.imread(file_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return bw


def GetConnectedComponents(img):

    output = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    [num_cc, labels, stats, centroids] = output
    return num_cc, labels, stats, centroids


def GetRightmostCoord(img_labels, label):
    points = np.where(img_labels == label)
    idx_rightmost = np.argmax(points[1])
    return points[0][idx_rightmost], points[1][idx_rightmost]



if __name__ == "__main__":
    
    img = ReadImage("test.jpg")
    print(img.shape)

    num_cc, labels, stats, centroids = GetConnectedComponents(img)
    print("found {} connected components.".format(num_cc))
    num_cc = num_cc - 1
    labels = labels[1:]
    stats = stats[1:]
    centroids = centroids[1:]

    print("Ignore the first one (background), we have {} lines in total.".format(num_cc))

    # display
    img_display = np.array(labels, copy = True)
    cv2.normalize(img_display, img_display, 0, 255, cv2.NORM_MINMAX)

    for i in range(num_cc):
        label = i + 1
        x, y = GetRightmostCoord(labels, label)
        img_display = cv2.circle(img_display, (y, x), radius = 10, color = (255, 255, 0), thickness = 2)

    
    plt.imshow(img_display)
    plt.show()

    pass



