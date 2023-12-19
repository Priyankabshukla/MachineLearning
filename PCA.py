import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

rng = np.random.default_rng(0)

def center_X(X:np.ndarray) -> np.ndarray:
    """
    Create a new matrix X' by subtracting the mean of rows of X from every rows of X, such that the mean of rows of X' is zero. 

    Param
        X: a 2D np.ndarray with shape (n, d). n is the number of data points and d is the dimension of the data. 

    Return: 
        X': a 2D np.ndarray of shape (n, d), such that the mean of the rows of X' is zero

    """
    mean=[]
    for i in range(len(X[0])):
        mean.append(np.average(X[:,i]))
    X_prime=X-np.array(mean)
    return X_prime


def pca(X:np.ndarray, k:int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center the input matrix X, compute the leading k right singular vectors and the leading k singular values of X. 

    Param
        X: a 2D np.ndarray with shape (n,d)
        k: an integer such that 1 <= k <= d

    Return: a tuple, where the first entry of the tuple is a numpy array of shape (k, d) and contains the leading k right singular vectors of X, and the second entry of the tuple is a numpy array of shape (k,) that contains the leading k singular values. The singular values should be sorted in descending order, and the singular vectors correspond to this order.

    """
    X_prime=center_X(X)
    u, s, vh = np.linalg.svd(X_prime) 
    cut_vh=vh[0:k,:]
    cut_s = s[0:k]
    return (cut_vh,cut_s)



def project(X:np.ndarray, right_singular_vecs:np.ndarray) -> np.ndarray:
    """
    Center the input matrix X, then project each row of X onto the subspace spanned by the rows of right_singular_vecs.

    Param
        X: a 2D np.ndarray with shape (n, d)
        right_singular_vecs: a 2D np.ndarray with shape (k, d). k is the number of right singular vectors, also the number of principle components. 

    Return: a 2D np.ndarray with shape (n, k), which contains the coordinates of these projections.
    """
    X_prime=center_X(X)
    mat=np.matmul(X_prime,right_singular_vecs.T)
    return mat


def draw_PCA_on_data(X_centered: np.ndarray, right_singular_vecs: np.ndarray) -> None:
	"""

	Plot the centered data X in its first two dimensions, and plot the unit vectors along the directions of the first and second principle component of X. 

	Param 
		X_centered: a 2D np.ndarray with shape (n, d)
		right_singular_vecs: a 2D np.ndarray with shape (k, d). k is the number of right singular vectors, also the number of principle components. 
	"""
	
	plt.figure()
	plt.scatter(X_centered[:, 0], X_centered[:, 1],label="original data")
	# plt.scatter(coordinates[:, 0], coordinates[:, 1],label="PCA")
	unit_right_singular_vecs = right_singular_vecs / np.linalg.norm(right_singular_vecs)
	plt.quiver([0], [0], unit_right_singular_vecs[0,0], unit_right_singular_vecs[0,1], color='r', units='xy', scale=1,label="1st PC")
	plt.quiver([0], [0], unit_right_singular_vecs[1,0], unit_right_singular_vecs[1,1], color='b', units='xy', scale=1, label="2nd PC")
	plt.xlim([-5, 5])
	plt.ylim([-5, 5])
	plt.legend()
	plt.show()





"""

Perform centering and PCA on synthetic data generated from a bivariate Gaussian distribution. Generate a plot with unit vectors along the directions of the first and second principle components on the original data. 

In this part, you should create a matrix right_singular_vecs, which is a 2D np.ndarray with shape (2, 2). The first row is the leading right singular vector (principle component) of X, the second row is the second principle component. 
"""

# Generate data
d = 2
n = 1000
mean = np.zeros(d)
cov = np.diag([3, 0.5])
X = rng.multivariate_normal(mean, cov, size = n)

right_singular_vecs,lam=pca(X,k=2)

# Generate plot 
X_centered = center_X(X)
draw_PCA_on_data(X_centered, right_singular_vecs)


""" 

Perform centering and PCA on synthetic data generated from a bivariate Gaussian distribution. Generate a plot with unit vectors along the directions of the first and second principle components on the original data. 
"""
# Generate data
d = 2
n = 1000
mean = np.zeros(d)
cov = np.array([[3, -1], [-1, 0.5]])
X = rng.multivariate_normal(mean, cov, size = n)

right_singular_vecs,lam=pca(X,k=2) 

# Generate plot. You DO NOT need to edit this part. 
X_centered = center_X(X)
draw_PCA_on_data(X_centered, right_singular_vecs)


"""

In this section, you will (i) read real-world data with four features (A, B, C, D) from a CSV file we provide and save it in a numpy array X; (this step is already done for you) (ii) perform PCA on X and computes the percentage of the variance explained with the top k leading principal components, for each
k = 1, 2, 3, 4. 
"""
# Read the data. DO NOT edit this part.
df = pd.read_csv("data.csv")
X = np.array(df[["A", "B", "C", "D"]])


right_singular_vecs,lam=pca(X,k=4)
eigen=lam**2
total_eig=np.sum(eigen)
perc=[]

for i in range(1,5):
    perc.append((np.sum(eigen[0:i])/total_eig)*100)
#     print(np.sum(eigen[0:i])/total_eig)
print(f"Percentage %: {perc}")

"""

In this part, you should create a matrix named coordinates, which is a 2D np.ndarray with shape (150, 4). The ith row in coordinates is the ith row of X projected onto the subspace spanned by the leading two principle components. 

The code to plot the data in these subspace coordinates has already been written for you. 
"""


coordinates=project(X,right_singular_vecs)

plt.figure()
plt.scatter(coordinates[:50, 0], coordinates[:50, 1], label="Class 0")
plt.scatter(coordinates[50:100, 0], coordinates[50:100, 1], label="Class 1")
plt.scatter(coordinates[100:150, 0], coordinates[100:150, 1], label="Class 2")
plt.title("Scatter plot with first 2 principle components of data.")
plt.legend()
plt.show()

