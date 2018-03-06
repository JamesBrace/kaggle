import numpy as np
from skimage.feature import canny
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import util


class PreProcessedData:

    def __init__(self):
        self.x = []
        self.y = []

        self.cannied_images = []
        self.filled_images = []
        self.cleaned_images = []
        self.processed_images = []

        print("Getting and reshaping data")
        self.get_and_reshape_data()
        print("Pre-processing data")
        self.pre_process()

    def get_and_reshape_data(self):
        self.x = np.loadtxt("../data/train_x.csv", delimiter=",")  # load from text
        self.y = np.loadtxt("../data/train_y.csv", delimiter=",")
        self.x = self.x.reshape(-1, 64, 64)  # reshape
        self.y = self.y.reshape(-1, 1)

        plt.imshow(self.x[0], cmap='gray')
        plt.show()


    def pre_process(self):
        print("Canny-ing images")
        self.canny_images()
        print("Filling images")
        self.fill_images()
        print("Cleaning images")
        self.clean_images()

    def canny_images(self):

        print(self.x[0])

        print("Inverting images")
        inverted_images = list(map(util.invert, self.x))

        plt.imshow(inverted_images[0], cmap='gray')
        plt.show()

        print("Cannying inverted images")
        self.cannied_images = list(map(canny, inverted_images))

        # Done for test purposes
        self.display_canny_image_example()

    def display_canny_image_example(self):
        plt.imshow(self.cannied_images[0], cmap='gray')
        plt.show()

        fig, ax = plt.subplots(figsize=(64, 64))
        ax.imshow(self.cannied_images[0], cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('Canny detector')
        ax.axis('off')
        ax.set_adjustable('box-forced')
        plt.show()

    def fill_images(self):
        self.filled_images = list(map(ndi.binary_fill_holes, self.cannied_images))
        self.cannied_images = []

        # Done for test purposes
        self.display_filled_image_example()

    def display_filled_image_example(self):
        plt.imshow(self.filled_images[0], cmap='gray')
        plt.show()

        fig, ax = plt.subplots(figsize=(64, 64))
        ax.imshow(self.filled_images[0], cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('filling the holes')
        ax.axis('off')
        plt.show()

    def clean_images(self):
        self.cleaned_images = morphology.remove_small_objects(self.filled_images, 21)
        self.filled_images = []

        self.display_clean_image_example()

    def display_clean_image_example(self):
        plt.imshow(self.cleaned_images[0], cmap='gray')
        plt.show()

        fig, ax = plt.subplots(figsize=(64, 64))
        ax.imshow(self.cleaned_images[0], cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('filling the holes')
        ax.axis('off')
        plt.show()


data = PreProcessedData()
