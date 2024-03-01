import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, zoom
from PIL import Image


class helper:
    @staticmethod
    def open_image(path):
        img_a = Image.open(path)
        return np.array(img_a.convert("L")).astype("float32") / 255

    @staticmethod
    def plot_pyramid(pyramid):
        """Plot the image pyramid"""
        plt.figure(figsize=(15, 15))
        for i in range(len(pyramid)):
            for j in range(pyramid[i].shape[2]):
                plt.subplot(len(pyramid), pyramid[i].shape[2], i * pyramid[i].shape[2] + j + 1)
                plt.imshow(pyramid[i][:, :, j], cmap="gray")
                plt.title(f"Octave {i}, Image {j}")
                plt.axis("off")
        plt.show()

    @staticmethod
    def imshow(img):
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()

    @staticmethod
    def plot_image_and_keypoints(image, keypoints, title=None):
        """Plot the image and the keypoints"""
        keypoints = np.array(keypoints)
        plt.figure(figsize=(10, 10))
        plt.imshow(image[::2, ::2], cmap="gray")
        plt.scatter(keypoints[:, 1], keypoints[:, 0], c="r", s=5, marker="x")
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def gaussian_filter(sigma):
        size = 2 * np.ceil(3 * sigma) + 1
        x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) / (2 * np.pi * sigma**2)
        return g / g.sum()

    @staticmethod
    def generate_base_image(image, sigma_in, sigma, upsample=2):
        """
        Generates the base image from the input image. This is done by upsampling the image by a factor of 2 and then applying a Gaussian blur
        """
        s0 = np.sqrt(max((sigma**2) - ((upsample * sigma_in) ** 2), 0.01))
        image = zoom(image, (upsample, upsample), order=1, mode="reflect")
        kernel = helper.gaussian_filter(s0)
        blurred_image = convolve(image, kernel)

        return blurred_image

    @staticmethod
    def open_base_image(path, sigma_in=0.5, sigma=1.6, upsample=2):
        """Opens the base image"""
        image = helper.open_image(path)
        return helper.generate_base_image(image, sigma_in, sigma, upsample)
