import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    [U, S, V] = np.linalg.svd(image)
    compressed_image = np.zeros_like(image)
    # print(compressed_image.shape)
    # print(U.shape)
    # print(S.shape)
    # print(V.shape)
    for i in range(num_values):
        u = np.expand_dims(U[:, i], axis=1)
        s = S[i]
        v = np.expand_dims(V[i, :].transpose(), axis=0)
        compressed_image += (u * s).dot(v)
        compressed_size += u.size
        compressed_size += s.size
        compressed_size += v.size

    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
