import numpy as np
import scipy.sparse.linalg as ssl
from scipy.sparse import csc_matrix

import time
import sys

A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)
b = np.array([1., 0.01, -1.], dtype=float)

start = time.time()
res = ssl.lsmr__b__(A, b)
end = time.time()
print("Time taken with Numba compilation:", end - start, "seconds.")

start = time.time()
res = ssl.lsmr__b__(A, b)
end = time.time()
print("Time taken AFTER Numba compilation:", end - start, "seconds.")

start = time.time()
res = ssl.lsmr(A, b)
end = time.time()
print("Time taken WITHOUT Numba (DEFAULT):", end - start, "seconds.")

print(res)