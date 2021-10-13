import numpy as np

def addMatrixToTarget(target_matrix, target_dim, added_matrix):
    if target_dim[1] <= target_matrix.shape[1]:
        target_matrix[:,target_dim[0]-1:target_dim[1]] = added_matrix
    else:
        if target_dim[0]-1 == target_matrix.shape[1]:
            target_matrix = np.concatenate((target_matrix, added_matrix),1)
        else:
            target_matrix = np.concatenate((target_matrix, np.zeros((target_matrix.shape[0], target_dim[0]-1 - target_matrix.shape[1])), added_matrix),1)

    return target_matrix

a = np.ones([1,5])*3
b = np.ones([1,2])*7
print(a,b)
c = addMatrixToTarget(a, [7,8], b)
print("result", c)