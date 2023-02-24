import numpy as np

frame = []

for j in range (0,5):
    for i in range(0, 4):
        for k in range(0, 5):
            frame.append(np.array([j,i,k]))

print(np.shape(frame))