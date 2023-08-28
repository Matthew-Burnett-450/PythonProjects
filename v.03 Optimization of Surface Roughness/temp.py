import matplotlib.pyplot as plt
import numpy as np

# Define the grid and constant K
x, y = np.linspace(-5, 5, 20), np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)
K = 1




u = K * X
v = -K * Y


plt.streamplot(X, Y, u, v, density=1.5)
plt.title('Streamlines of the Flow K=1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
