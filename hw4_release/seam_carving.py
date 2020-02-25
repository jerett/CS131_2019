import numpy as np
from skimage import color


def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: you can use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    out = np.zeros((H, W))

    ### YOUR CODE HERE
    gray_image = color.rgb2gray(image)
    gradients = np.gradient(gray_image)
    out = np.sum(np.abs(gradients), axis=0)
    # out = np.abs(gradients[0]) + np.abs(gradients[1])
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    # for i in range(1, H):
    #     for j in range(0, W):
    #         # left, up, right
    #         upper_cost = np.array([np.inf, np.inf, np.inf])
    #         if j != 0:
    #             upper_cost[0] = cost[i - 1, j - 1]
    #         upper_cost[1] = cost[i - 1, j]
    #         if j != W - 1:
    #             upper_cost[2] = cost[i - 1, j + 1]
    #         min_index = np.argmin(upper_cost)
    #         # path equal index-1
    #         paths[i][j] = (min_index - 1)
    #         cost[i][j] = upper_cost[min_index] + energy[i][j]

    # vectorized version
    for i in range(1, H):
        padded_up_cost = np.pad(cost[i-1], 1, mode='constant', constant_values=np.inf)
        cost_map = np.zeros((3, W))
        # left
        cost_map[0] = padded_up_cost[:W] + energy[i]
        # middle
        cost_map[1] = padded_up_cost[1:W+1] + energy[i]
        # right
        cost_map[2] = padded_up_cost[2:W+2] + energy[i]
        # min_index
        cost_min_index = np.argmin(cost_map, axis=0)
        paths[i] = cost_min_index - 1
        cost[i] = cost_map[cost_min_index, range(W)]
        # print('cost_map', cost_map)
        # print('min_cost', cost_map[range(3), cost_min_index])
        # print('min_index', cost_min_index)
    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
        "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    seam = np.zeros(H, dtype=np.int)

    # Initialization
    seam[H - 1] = end

    ### YOUR CODE HERE
    for row in range(H-2, -1, -1):
        path_val = paths[row + 1, seam[row + 1]]
        seam[row] = path_val + seam[row+1]
        # if path_val == -1:
        #     seam[row] = seam[row + 1] - 1
        # elif path_val == 0:
        #     seam[row] = seam[row + 1]
        # else:
        #     seam[row] = seam[row + 1] + 1
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE

    # out = np.zeros((H, W-1, C))
    # for i in range(H):
    #     for j in range(W-1):
    #         if j < seam[i]:
    #             out[i][j] = image[i][j]
    #         elif j >= seam[i]:
    #             out[i][j] = image[i][j+1]

    element_bool_indices = np.ones((H, W), dtype=bool)
    element_bool_indices[range(H), seam] = False
    out = image[element_bool_indices].reshape(H, W-1, C)
    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    for i in range(W-size):
        energy = efunc(out)
        vcost, vpaths = cfunc(out, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(out, seam)
    # print(out.shape)
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    for i in range(H):
        out[i] = np.insert(image[i], seam[i], image[i][seam[i]], axis=0)
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    for i in range(size-W):
        energy = efunc(out)
        vcost, vpaths = cfunc(out, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = duplicate_seam(out, seam)
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE

    seams = find_seams(out, size-W)
    seams = seams.reshape((H, W, 1))
    for i in range(size-W):
        seam = np.where(seams == i+1)[1]
        # for j in range(H):
        #     seam[j] += ((seams[j, :seam[j]] > 0) & (seams[j, :seam[j]] < (i+1))).sum()
        out = duplicate_seam(out, seam)
        seams = duplicate_seam(seams, seam)
    # for i in range(H):
    #     insert_indices = np.where(seams[i] > 0)[0]
    #     # print(insert_indices)
    #     out[i] = np.insert(image[i], insert_indices, image[i][insert_indices], axis=0)
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j + 1] - image[0, j - 1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for i in range(1, H):
        padded_up_cost = np.pad(cost[i-1], 1, mode='constant', constant_values=np.inf)
        cost_map = np.zeros((3, W))
        padded_m1 = np.pad(image[i-1], 1, mode='constant', constant_values=0)
        padded_m2 = np.pad(image[i], 1, mode='constant', constant_values=0)

        cv = np.abs(padded_m2[2:2+W] - padded_m2[:W])
        cv[0] = 0
        cv[-1] = 0

        # left
        cl = np.abs(padded_m1[1:1+W] - padded_m2[:W])
        cl[0] = 0
        cl += cv
        # cl[-1] = 0
        cost_map[0] = padded_up_cost[:W] + cl
        # middle
        cost_map[1] = padded_up_cost[1:W+1] + cv
        # right
        cr = np.abs(padded_m1[1:1+W] - padded_m2[2:2+W])
        cr[-1] = 0
        cr += cv
        # cr[0] = 0
        cost_map[2] = padded_up_cost[2:W+2] + cr

        cost_map += energy[i]
        # min_index
        cost_min_index = np.argmin(cost_map, axis=0)
        paths[i] = cost_min_index - 1
        cost[i] = cost_map[cost_min_index, range(W)]

    # for i in range(1, H):
    #     for j in range(0, W):
    #         # left, up, right
    #         upper_cost = np.array([np.inf, np.inf, np.inf])
    #         if j != 0:
    #             cl = np.abs(image[i-1][j] - image[i][j-1])
    #             if j < W -1:
    #                 cl += np.abs(image[i][j+1] - image[i][j-1])
    #             upper_cost[0] = cost[i - 1, j - 1] + cl
    #         cv = 0
    #         if j > 0 and j < W-1:
    #             cv = np.abs(image[i][j+1] - image[i][j-1])
    #         upper_cost[1] = cost[i - 1, j] + cv
    #         if j != W - 1:
    #             cr = 0
    #             if cr < W-1:
    #                 cr = np.abs(image[i][j+1] - image[i][j-1]) + np.abs(image[i-1][j] - image[i][j+1])
    #             upper_cost[2] = cost[i - 1, j + 1] + cr
    #         min_index = np.argmin(upper_cost)
    #         # path equal index-1
    #         paths[i][j] = (min_index - 1)
    #         cost[i][j] = upper_cost[min_index] + energy[i][j]
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
        "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Hint: do we really need to compute the whole cost map again at each iteration?

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    # def update_energy(energy, img, seam):
    #     W = energy.shape[1] + 1
    #     out_energy = np.copy(energy)
    #     seam_min_x = np.min(seam)
    #     seam_max_x = np.max(seam)
    #     # print(seam_min_x, seam_max_x, W)
    #     # range from seam_min_x to seam_max_x, but we sample from seam_min_x-1, seam_max_x+1 for border gradient problem
    #     if seam_min_x == 0 and seam_max_x < W-1:
    #         print('fuck0')
    #         out_energy[:, seam_min_x:seam_max_x+1] = efunc(img[:, seam_min_x:seam_max_x+2])[:, :seam_max_x-seam_min_x+1]
    #     elif seam_min_x > 0 and seam_max_x == W-1:
    #         print('fuck1')
    #         out_energy[:, seam_min_x:] = efunc(img[:, seam_min_x-1:])[:, 1:]
    #     else:
    #         # print(seam_min_x, seam_max_x+1, seam_min_x-1, seam_max_x+1)
    #         out_energy[:, seam_min_x-1:seam_max_x+1] = efunc(img[:, seam_min_x-1:seam_max_x+2])[:, 1:seam_max_x-seam_min_x+2]
    #
    #     # min_x = np.max([0, seam_min_x-1])
    #     # max_x = np.min([W, seam_max_x+2])
    #     # print(W, seam_min_x, seam_max_x, min_x, max_x)
    #     # # print('update energy', min_x, max_x)
    #     # out_energy[:, min_x:max_x+1] = efunc(img[:, min_x:max_x])[:, 1:max_x-min_x-1+3]
    #     return out_energy

    # energy = efunc(out)
    # # vcost, vpaths = cfunc(out, energy)
    #
    # for i in range(W-size):
    #     # energy = efunc(out)
    #     # vcost, vpaths = cfunc(out, energy)
    #     vcost, vpaths = cfunc(out, energy)
    #     end = np.argmin(vcost[-1])
    #     seam = backtrack_seam(vpaths, end)
    #     out = remove_seam(out, seam)
    #     energy = remove_seam(energy, seam)
    #     energy = update_energy(energy, out, seam)
    pass
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    out = np.copy(image)

    ### YOUR CODE HERE
    H, W, _ = out.shape
    w = np.where(mask == 1)
    min_x = np.min(w[1])
    max_x = np.max(w[1])
    min_y = np.min(w[0])
    max_y = np.max(w[0])
    print(min_x, max_x, min_y, max_y)
    size = (W - (max_x - min_x + 1))
    assert size < W, "size must be greather than %d" % W

    for i in range(W-size):
        energy = energy_function(out)
        energy[mask] = -1000
        vcost, vpaths = compute_cost(out, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(out, seam)
        mask = remove_seam(mask, seam)
    out = enlarge(out, W)
    ### END YOUR CODE


    return out
