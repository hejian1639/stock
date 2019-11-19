# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import numpy as np

length = 100
# create data
x = np.random.rand(length).astype(np.float32)
deviation = np.random.rand(length).astype(np.float32)
y = x * 0.1 + 0.3 + (deviation - 0.5) * 0.01

one = np.ones(length)

xMat = np.mat((x, one))

print(np.linalg.inv((xMat * xMat.T))*xMat*np.mat(y).T)
