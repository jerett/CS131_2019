
from linalg import *

v1 = np.array([[1, 2],
               [3, 4]])
v2 = np.array([[1, 2],
               [3, 4]])
print('dot_product:\n', dot_product(v1, v2))

m = np.array([[1, 2],
              [3, 4],
              [5, 6]])
v2 = np.array([1, 2])
v1 = v2.T
print('matrix_mult:\n', matrix_mult(m, v1, v2))

array = np.array([[0, 0, 0, 2, 2],
                  [0, 0, 0, 3, 3],
                  [0, 0, 0, 1, 1],
                  [1, 1, 1, 0, 0],
                  [2, 2, 2, 0, 0],
                  [5, 5, 5, 0, 0],
                  [1, 1, 1, 0, 0]])
u, s, v = svd(array)
print('u:\n', u, 's:\n', s, 'v:\n', v)
print('get_singular_value:\n', get_singular_values(array, 2))

array2 = np.array([[1, 2, 3],
                   [4, 5 ,6],
                   [7, 8, 9]])
w, v = eigen_decomp(array2)
print('w:\n', w)
print('v:\n', v)

eigen_value, eigen_vectors = get_eigen_values_and_vectors(array2, 2)
print('w:\n', eigen_value)
print('v:\n', eigen_vectors)
print('shape:\n', eigen_vectors.shape)

