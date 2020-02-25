import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(0, Hi):
        for n in range(0, Wi):
            # print('caculate (', m, ',', n, ')')
            for i in range(Hk):
                for j in range(Wk):
                    # map kernel index (0, 0) to (-1, 1)
                    x, y = (m + Hk // 2 - i, n + Wk // 2 - j)
                    val = 0
                    # print('kernel[', i, '][', j, ']*', 'image[', x, ']', '[', y, ']')
                    if x >= 0 and y >= 0 and x < Hi and y < Wi:
                        val = image[x][y] * kernel[i][j]
                    out[m][n] += val
    ### END YOUR CODE
    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    flip_kernel = np.flip(np.flip(kernel, 0), 1)
    padding_height = Hk // 2
    padding_width = Wk // 2
    padding_image = zero_pad(image, padding_height, padding_width)
    for m in range(padding_height, padding_height + Hi):
        for n in range(padding_width, padding_width + Wi):
            val = np.sum(
                padding_image[m - padding_height:m + (Hk - padding_height), n - padding_width:n + (Wk - padding_width)]
                * flip_kernel)
            out[m - padding_height, n - padding_width] = val
    ### END YOUR CODE
    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    flip_g = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, flip_g)
    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g_mean = np.mean(g)
    out = cross_correlation(f, g - g_mean)
    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))

    padding_height = Hg // 2
    padding_width = Wg // 2
    padding_f = zero_pad(f, padding_height, padding_width)

    g_mean = np.mean(g)
    g_std = np.std(g)
    g_nor = (g - g_mean) / g_std

    for m in range(padding_height, padding_height + Hf):
        for n in range(padding_width, padding_width + Wf):
            patch_f = padding_f[m - padding_height:m + (Hg - padding_height), n - padding_width:n + (Wg - padding_width)]
            patch_f_mean = np.mean(patch_f)
            patch_f_std = np.std(patch_f)
            patch_nor = (patch_f - patch_f_mean) / patch_f_std
            val = np.sum(patch_nor * g_nor)
            out[m - padding_height, n - padding_width] = val
    ### END YOUR CODE
    return out

