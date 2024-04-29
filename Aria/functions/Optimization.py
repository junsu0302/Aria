import numpy as np

def sphere(x: float, y: float) -> float:
  return x ** 2 + y ** 2

def matyas(x: float, y: float) -> float:
  return 0.26 * (x**2 + y**2) - 0.48 * x * y

def goldstein(x: float, y: float) -> float:
  return (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))

def rosenbrock(x0: float, x1: float) -> float:
  return 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2