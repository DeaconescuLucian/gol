import numpy as np
import matplotlib.pyplot as plt


def slime_activation(x, b):
    return 1 / np.power(2, np.power(x - b, 2))

b = 0.2

x_values = np.linspace(0, 1, 500)

y_values = slime_activation(x_values, b)

plt.plot(x_values, y_values, label=f'b = {b}')
plt.title('Slime Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()