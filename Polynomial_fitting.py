# importing the required module
import matplotlib.pyplot as plt
import numpy as np
import functions as f
import random


# Generating N input samples
random.seed(1)
N = 20
U = np.linspace(0, 10, N)  # (N*1)

# Getting coefficients of polynomial from user
Coefficients = [int(item) for item in input("Enter the coefficient of polynomial with space between them and without commas in the decreasing order of their degree : ").split()]
# Calculating values of polynomial for input U
y = f.polynomial(Coefficients, U)

# Creating a noise from random normal distribution and adding it to output
mean = float(input("Enter the mean of noise: "))
std = float(input("Enter the standard deviation of noise: "))
noise = np.random.normal(mean, std, y.shape)
Y = y + noise           # (N*1)
n = int(input("Enter the order of polynomial that you would like to fit to given x and y data: "))

# calculating solution for 0th to nth order
for i in range(n+1):
    # calling regressor function from function.py to create regression
    X = f.regressor(U, i)    # (N*n+1)
    # calling least_sqr function from function.py
    Coefficients, NRMSE = f.least_sqr(X, Y, N)
    print('\n', f"Coefficients of polynomial of {i} order")
    print(" Coefficients: ", Coefficients,)

plt.plot(U, Y, "*", label='Given data')
plt.plot(U, f.polynomial(Coefficients, U), label='Fitted polynomial curve')
plt.xlabel("Values of X")
plt.ylabel("Values of Y")
plt.legend()
plt.show()
