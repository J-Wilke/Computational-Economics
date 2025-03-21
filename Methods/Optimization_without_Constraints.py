import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols and function
x, y = sp.symbols('x y')
f =  x**2 - 3*x + y**2 - 3*y
print("f(x,y) =", f)

# Compute the gradient (partial derivatives)
gradient = [sp.diff(f, x), sp.diff(f, y)]
print("Gradient:", gradient)

# Solve for stationary points (where the gradient equals zero)
stationary_points = sp.solve(gradient, (x, y), dict=True)  # using dict=True to get dictionaries
print("Stationary Points:", stationary_points)

# Filter only real stationary points
points = [pt for pt in stationary_points if all(val.is_real for val in pt.values())]

# Compute the Hessian matrix (second partial derivatives)
Hessian = [[sp.diff(f, x, x), sp.diff(f, x, y)],
           [sp.diff(f, y, x), sp.diff(f, y, y)]]
print("Hessian Matrix:", Hessian)

def classify_stationary_point(Hessian, point):
    # If the point is not a dictionary (e.g., it is a tuple), convert it to a dictionary mapping symbols to values.
    if not isinstance(point, dict):
        point = dict(zip([x, y], point))
    
    # Evaluate the Hessian matrix at the stationary point
    H_eval = [[expr.subs(point) for expr in row] for row in Hessian]
    # Calculate the determinant: D = f_xx * f_yy - (f_xy)^2
    D = H_eval[0][0] * H_eval[1][1] - H_eval[0][1] * H_eval[1][0]
    f_xx = H_eval[0][0]
    
    # Use numerical evaluation with chop=True to remove negligible imaginary parts
    D_val = float(sp.N(D, chop=True))
    f_xx_val = float(sp.N(f_xx, chop=True))
    
    # Classify the stationary point based on the determinant and f_xx
    if D_val > 0:
        if f_xx_val > 0:
            local_ext = "Local Minimum"
            convexity = "Convex (locally)"
        elif f_xx_val < 0:
            local_ext = "Local Maximum"
            convexity = "Concave (locally)"
        else:
            local_ext = "Test inconclusive (f_xx = 0)"
            convexity = "Indeterminate"
    elif D_val < 0:
        local_ext = "Saddle Point"
        convexity = "Indefinite"
    else:
        local_ext = "Test inconclusive (D = 0)"
        convexity = "Indeterminate"
        
    return local_ext, convexity

# Classify all stationary points and print results
for point in points:
    classification = classify_stationary_point(Hessian, point)
    print(f"\nStationary point {point}:")
    print("  Local Classification:", classification[0])
    print("  Convexity/Concavity:", classification[1])

# --------------------- 3D Plot Section ---------------------
# Create a meshgrid for x and y values
x_vals = np.linspace(-3, 6, 100)
y_vals = np.linspace(-3, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Convert the sympy function f into a numpy-compatible function using lambdify
f_lambdified = sp.lambdify((x, y), f, 'numpy')
Z = f_lambdified(X, Y)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the function
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mark the stationary points on the surface
for point in points:
    # Ensure the point is a dictionary and extract its x and y values
    if not isinstance(point, dict):
        point = dict(zip([x, y], point))
    x_val = float(point[x])
    y_val = float(point[y])
    z_val = f_lambdified(x_val, y_val)
    ax.scatter(x_val, y_val, z_val, color='red', s=100, marker='o')

# Set plot labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('3D Surface Plot of f(x,y) with Stationary Points')

plt.show()
