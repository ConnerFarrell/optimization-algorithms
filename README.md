# Project Title: Optimization Algorithms

## Overview
This project implements classical optimization algorithms:
Gradient Descent, Heavy Ball, Nesterov Accelerated Gradient,
and Newton’s Method, and visualizes their trajectories on a convex test function.

## Key Features
- Implements multiple optimization methods from scratch in Python.
- Plots convergence trajectories and compares iteration counts.
- Educational example for convex optimization coursework.

## Example Output
Gradient Descent iterations (tol=1e-5): 78

Heavy Ball iterations (tol=1e-5): 62

Nesterov iterations (tol=1e-5): 51

Newton iterations (tol=1e-5): 4

![All Trajectories](images/optimization_trajectories.png)
Newton iterations (tol=1e-10): 4
![Newton Trajectory](images/optimization_newton_trajectory.png)

## Skills Demonstrated
- Numerical optimization (GD, NAG, Newton)
- Python (NumPy, Matplotlib)
- Convex analysis

## How to Run
```bash
pip install -r requirements.txt
python optimization_methods.py
