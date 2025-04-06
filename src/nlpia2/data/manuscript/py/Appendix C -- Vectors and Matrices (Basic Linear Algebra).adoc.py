import numpy as np

np.array(range(4))

np.arange(4)

x = np.arange(0.5, 4, 1)

x

x[1] = 2

x

x.shape

x.T.shape

np.array([range(4), range(4)])

array([[0, 1, 2, 3],

X = np.array([range(4), range(4)])

X.shape

X.T.shape

A = np.array([[1, 2, 3], [4, 5, 6]])

A

A.T

A

A[0]

A[1]

np.diff(A, axis=0)

A[1] - A[0]

A.diff()

import numpy as np

vector_query = np.array([1, 1])

vector_tc = np.array([1, 0])

vector_wired = np.array([5, 6])

normalized_query = vector_query / np.linalg.norm(vector_query)

normalized_tc = vector_tc / np.linalg.norm(vector_tc)

normalized_wired = vector_wired / np.linalg.norm(vector_wired)

normalized_query

normalized_tc

normalized_wired

np.dot(normalized_query, normalized_tc)  # cosine similarity

np.dot(normalized_query, normalized_wired)  # cosine similarity

1 - np.dot(normalized_query, normalized_tc)  # cosine distance

1 - np.dot(normalized_query, normalized_wired)  # cosine distance

vector_tc = np.array([1, 0])

vector_wired = np.array([5, 6])

np.abs(vector_tc - vector_wired).sum()

normalized_tc = vector_tc / np.linalg.norm(vector_tc)

normalized_wired = vector_wired / np.linalg.norm(vector_wired)

np.abs(normalized_tc - normalized_wired).sum()
