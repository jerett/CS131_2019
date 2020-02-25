import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io


def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = np.int64(0.5 * (image ** 2))
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2gray(image)
    # out = np.sum((np.array([[[0.299, 0.587, 0.114]]]) * image), axis=2)  # astype
    # out = (np.array([[[0.299, 0.587, 0.114]]]) * image)  # astype
    ### END YOUR CODE

    return out


def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None
    ### YOUR CODE HERE
    out = np.zeros(image.shape, dtype=np.int64)
    if channel == 'R':
        out[:, :, 1] = image[:, :, 1]
        out[:, :, 2] = image[:, :, 2]
    elif channel == 'G':
        out[:, :, 0] = image[:, :, 0]
        out[:, :, 2] = image[:, :, 2]
    elif channel == 'B':
        out[:, :, 0] = image[:, :, 0]
        out[:, :, 1] = image[:, :, 1]
    ### END YOUR CODE
    return out


def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    out = np.zeros(lab.shape)
    if channel == 'L':
        out[:, :, 0] = lab[:, :, 0]
    elif channel == 'A':
        out[:, :, 1] = lab[:, :, 1]
    elif channel == 'B':
        out[:, :, 2] = lab[:, :, 2]
    ### END YOUR CODE
    return out


def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    out = np.zeros(hsv.shape)
    if channel == 'H':
        out[:, :, 0] = hsv[:, :, 0]
    elif channel == 'S':
        out[:, :, 1] = hsv[:, :, 1]
    elif channel == 'V':
        out[:, :, 2] = hsv[:, :, 2]
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    out = np.zeros(image1.shape, dtype=np.int64)
    half_width = int(out.shape[1] / 2)
    out[:, :half_width] = rgb_decomposition(image1[:, :half_width], channel1)
    out[:, half_width:] = rgb_decomposition(image2[:, half_width:], channel2)
    ### END YOUR CODE

    return out
