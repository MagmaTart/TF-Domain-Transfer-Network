import numpy as np
import matplotlib.pyplot as plt
import cv2

class Tools:
    def __init__(self):
        pass

    def show_fashion(self, image):
        cv2.imshow('Image', image)
        cv2.waitKey(100000)
        cv2.destroyAllWindows()

    def show_mnist(self, image):
        plt.imshow(np.reshape(image, [64, 64]))
        plt.show()
