# Núria Casals 950801-T740
# Robin de Groot 981116-T091


# Load packages
import numpy as np
import gym
import torch
from tqdm import trange
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Load model
try:
    model = torch.load('neural-network-1.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)

y_range = np.arange(0., 1.5, 0.01)
omg_range = np.arange(-np.pi, np.pi, 0.0314)

results_q = list()

for y in y_range:
    #print("testing value y: ", y)
    for omg in omg_range:
        state = np.array([0.,y,0.,0.,omg,0.,0.,0.])
        state = torch.tensor([state], dtype=torch.float32)
        q_values = model(state)
        max_q, action = torch.max(q_values, axis=1)

        results_q.append([y, omg, max_q.item(), action.item()])

results_q = np.array(results_q)

#print(results_q)
#print(results_q.shape)
print(np.unique(results_q[:, 3]))

fig = plt.figure()
ax = plt.axes(projection='3d')

x = results_q[:, 0]
y = results_q[:, 1]
z = results_q[:, 2]

ax.set_title("maxQ(s,a) for different value of y and omega")
ax.set_xlabel("y")
ax.set_ylabel("Omega")

ax.scatter3D(x, y, z, c=z, cmap='viridis');

plt.show()
plt.gcf().clear()
plt.clf()

fig = plt.figure()
ax = plt.axes(projection='3d')

#x = results_q[:, 0]
#y = results_q[:, 1]
z2 = results_q[:, 3]

ax.set_title("argmaxQ(s,a) for different y and omega")
ax.set_xlabel("y")
ax.set_ylabel("Omega")

ax.scatter3D(x, y, z2, c=z2, cmap='viridis');

plt.savefig('actions')
plt.show()