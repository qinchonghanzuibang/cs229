import numpy as np
from util import add_intercept

_ = "-------------------------------"
t = np.array([
    [1, 2, 4],
    [3, 4, 2]
])

print(t)
print(t.shape)

print(t[0])
print(t[0].shape)
print(t[1].shape)

# print one column of t
print(t[:, 0])
print(t[:, 0].shape)

t1 = np.array([1, 2, 3])
print(t1)
print(t1.shape)

t1 += t[0]
print(t1)

# Example data
x = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])

y = np.array([0, 0, 1, 1])

# Calculate means for each class
mu_0 = np.mean(x[y == 0], axis=0)
mu_1 = np.mean(x[y == 1], axis=0)

print("Mean for class 0:")
print(mu_0)
print("Mean for class 1:")
print(mu_1)
# Number of samples
m = x.shape[0]

# Calculate the covariance matrix
sigma_0 = (x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0)
sigma_1 = (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)

print("Covariance matrix for class 0:")
print(sigma_0)
print("Covariance matrix for class 1:")
print(sigma_1)
sigma = (sigma_0 + sigma_1) / m

print("Covariance matrix \(\Sigma\):")
print(sigma)

# help me test the function of np.hstack

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.hstack([a, b])
print(c)

# test add_intercept


x = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])
print(x)
print(x.shape)
x = add_intercept(x)
print(x)
print(x.shape)

a = np.array([[1, 2], [3, 4]])
b = np.array([[2, 0], [1, 2]])
print(_)
print(a)
print(b)
print(a * b)  # Output: [[2, 0], [3, 8]]

print(_)
a = np.array([[1, 2], [3, 4]])
b = np.array([[2, 0], [1, 2]])
print(np.dot(a, b))  # Output: [[4, 4], [10, 8]]

print(_)
x = np.array([1, 2, 3])  # This is actually a 1D array with shape (3,)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Using np.dot
result_dot = np.dot(x, A)
print("Using np.dot:", result_dot)  # Output: [22 28]

# Using np.matmul
result_matmul = np.matmul(x, A)
print("Using np.matmul:", result_matmul)  # Output: [22 28]
