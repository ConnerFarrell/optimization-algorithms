# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Define the objective, gradient, and Hessian
# --------------------------
def f_obj(x):
    """Evaluate the objective function at x = [x1, x2]."""
    return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)

def grad_f(x):
    """Compute the gradient of f_obj at x = [x1, x2]."""
    exp1 = np.exp(x[0] + 3*x[1] - 0.1)
    exp2 = np.exp(x[0] - 3*x[1] - 0.1)
    exp3 = np.exp(-x[0] - 0.1)
    grad1 = exp1 + exp2 - exp3
    grad2 = 3*exp1 - 3*exp2
    return np.array([grad1, grad2])

def hessian_f(x):
    """Compute the Hessian of f_obj at x = [x1, x2]."""
    exp1 = np.exp(x[0] + 3*x[1] - 0.1)
    exp2 = np.exp(x[0] - 3*x[1] - 0.1)
    exp3 = np.exp(-x[0] - 0.1)
    f11 = exp1 + exp2 + exp3
    f12 = 3*exp1 - 3*exp2
    f22 = 9*exp1 + 9*exp2
    return np.array([[f11, f12],
                     [f12, f22]])

# --------------------------
# (a) Gradient Descent
# --------------------------
def gd(f, gradf, x0, tol):
    """Gradient Descent with fixed step size alpha=0.05.
       Terminates when ||grad f(x)||_2 <= tol or after max_iter iterations."""
    alpha = 0.05
    max_iter = 10000
    x = x0.copy()
    trajectory = [x.copy()]
    iters = 0
    while np.linalg.norm(gradf(x)) > tol and iters < max_iter:
        x = x - alpha * gradf(x)
        trajectory.append(x.copy())
        iters += 1
    return np.array(trajectory), iters

# --------------------------
# (b) Heavy Ball Method
# --------------------------
def heavyball(f, gradf, x0, tol):
    """Heavy Ball method with fixed step size alpha=0.05 and momentum beta=0.7.
       Terminates when ||grad f(x)||_2 <= tol or after max_iter iterations."""
    alpha = 0.05
    beta = 0.7
    max_iter = 10000
    trajectory = [x0.copy()]
    # First step: a simple gradient descent step
    x_prev = x0.copy()
    x = x0 - alpha * gradf(x0)
    trajectory.append(x.copy())
    iters = 1
    while np.linalg.norm(gradf(x)) > tol and iters < max_iter:
        x_new = x - alpha * gradf(x) + beta * (x - x_prev)
        trajectory.append(x_new.copy())
        x_prev = x.copy()
        x = x_new.copy()
        iters += 1
    return np.array(trajectory), iters

# --------------------------
# (c) Nesterov’s Accelerated Gradient Method
# --------------------------
def nesterov(f, gradf, x0, tol):
    """Nesterov's method with step size alpha=0.05 and momentum beta_k = (k-1)/(k+2).
       Terminates when ||grad f(x)||_2 <= tol or after max_iter iterations."""
    alpha = 0.05
    max_iter = 10000
    trajectory = [x0.copy()]
    x_prev = x0.copy()
    x = x0.copy()
    iters = 0
    while np.linalg.norm(gradf(x)) > tol and iters < max_iter:
        beta = iters / (iters + 2)  # for iters=0, beta = 0.
        y = x + beta * (x - x_prev)
        x_new = y - alpha * gradf(y)
        trajectory.append(x_new.copy())
        x_prev = x.copy()
        x = x_new.copy()
        iters += 1
    return np.array(trajectory), iters

# --------------------------
# (d) Newton's Method
# --------------------------
def newton(f, gradf, hessianf, x0, tol):
    """Newton's method with pure Newton step (step size=1).
       Terminates when ||grad f(x)||_2 <= tol or after max_iter iterations."""
    max_iter = 10000
    trajectory = [x0.copy()]
    x = x0.copy()
    iters = 0
    while np.linalg.norm(gradf(x)) > tol and iters < max_iter:
        H = hessianf(x)
        # Solve H p = -grad f(x)
        p = np.linalg.solve(H, -gradf(x))
        x = x + p
        trajectory.append(x.copy())
        iters += 1
    return np.array(trajectory), iters

# --------------------------
# Create Level Set Plot of f_obj
# --------------------------
num_gridpoint = 300
xx = np.linspace(-1.0, 0.5, num_gridpoint)
# Use -xx for the second coordinate (as in your provided code)
X1, X2 = np.meshgrid(xx, -xx)
fx = np.zeros_like(X1)
for i in range(num_gridpoint):
    for j in range(num_gridpoint):
        fx[i, j] = f_obj(np.array([X1[i, j], X2[i, j]]))

plt.figure(figsize=(12, 10))
CS = plt.contour(X1, X2, fx, levels=[2.56, 2.6, 2.7, 2.8, 3.0, 3.3], colors='black')
plt.xlim([-1.0, 0.5])
plt.ylim([-0.5, 0.5])
# Revised code: use CS.collections if available to set dashed linestyles.
if hasattr(CS, 'collections'):
    for coll in CS.collections:
         coll.set_linestyle('--')
else:
    print("CS.collections not found; skipping linestyle customization.")
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Level Sets of f(x) and Optimization Trajectories')

# --------------------------
# Run Algorithms on f_obj
# --------------------------
x0 = np.array([-0.45, 0.35])
tol1 = 1e-5

# (a) Gradient Descent
traj_gd, iters_gd = gd(f_obj, grad_f, x0, tol1)
print("Gradient Descent iterations (tol=1e-5):", iters_gd)
plt.plot(traj_gd[:,0], traj_gd[:,1], 'ro-', label='Gradient Descent')

# (b) Heavy Ball Method
traj_hb, iters_hb = heavyball(f_obj, grad_f, x0, tol1)
print("Heavy Ball iterations (tol=1e-5):", iters_hb)
plt.plot(traj_hb[:,0], traj_hb[:,1], 'bo-', label='Heavy Ball')

# (c) Nesterov's Method
traj_nes, iters_nes = nesterov(f_obj, grad_f, x0, tol1)
print("Nesterov iterations (tol=1e-5):", iters_nes)
plt.plot(traj_nes[:,0], traj_nes[:,1], 'go-', label='Nesterov')

# (d) Newton's Method with tol=1e-5
traj_newton, iters_newton = newton(f_obj, grad_f, hessian_f, x0, tol1)
print("Newton iterations (tol=1e-5):", iters_newton)
plt.plot(traj_newton[:,0], traj_newton[:,1], 'mo-', label="Newton (tol=1e-5)")

plt.legend()
plt.show()

# --------------------------
# Extra run: Newton's Method with tol=1e-10
# --------------------------
tol2 = 1e-10
traj_newton2, iters_newton2 = newton(f_obj, grad_f, hessian_f, x0, tol2)
print("Newton iterations (tol=1e-10):", iters_newton2)

plt.figure(figsize=(12, 10))
CS2 = plt.contour(X1, X2, fx, levels=[2.56, 2.6, 2.7, 2.8, 3.0, 3.3], colors='black')
plt.xlim([-1.0, 0.5])
plt.ylim([-0.5, 0.5])
if hasattr(CS2, 'collections'):
    for coll in CS2.collections:
         coll.set_linestyle('--')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Newton Trajectory')
plt.plot(traj_newton2[:,0], traj_newton2[:,1], 'mo-', label="Newton (tol=1e-10)")
plt.legend()
plt.show()
