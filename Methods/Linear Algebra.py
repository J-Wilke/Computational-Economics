import numpy as np

# Vector Definition
v1 = np.array([1, 2])
v2 = np.array([4, 5])
print("Vektor v1:", v1)
print("Vektor v2:", v2)

# Vector Addition
v3 = v1 + v2
print("Vektor v3:", v3)

#Vector Multiplication
v4 = v1 * 2
print("Vektor v4:", v4)

v5 = v1 * v2
print("Vektor v5:", v5)

# Vectors Dot Product
dot_product = np.dot(v1, v2)
print("Dot Product:", dot_product)

# Vectors Cross Product
cross_product = np.cross(v1, v2)
print("Cross Product:", cross_product)

# Define Matrix
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Matrix A:", A)
print("Matrix B:", B)

# Matrix Addition
C = A + B
print("Matrix C:", C)

# Matrix Multiplication
D = A @ B               # or np.dot(A, B)
print("Matrix   D:", D)

# Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)

# Determinant
determinant = np.linalg.det(A)
print("Determinant:", determinant)

# Inverse
inverse = np.linalg.inv(A)
print("Inverse:", inverse)

# Transpose
transpose = np.transpose(A) # or A.T
print("Transpose:", transpose)

# Solve linear system A*x = b
b = np.array([5, 7])
x = np.linalg.solve(A,b)
print("Solution:", x)




