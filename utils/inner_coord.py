import numpy as np
np.random.seed(1)
EPSILON = 1e-08

import pandas as pd

def get_vec(crd):
    """ Get the vector of the sequential coordinate.
    """
    # (B, A, D)
    crd_ = np.roll(crd, -1, axis=-2)
    vec = crd_ - crd
    # (B, A-1, D)
    return vec[:, :-1, :]
 
def get_dis(crd):
    """ Get the distance of the sequential coordinate.
    """
    # (B, A-1, D)
    vec = get_vec(crd)
    # (B, A-1, 1)
    dis = np.linalg.norm(vec, axis=-1, keepdims=True)
    return dis, vec
 
def get_angle(crd):
    """ Get the bond angle of the sequential coordinate.
    """
    # (B, A-1, 1), (B, A-1, D)
    dis, vec = get_dis(crd)
    vec_ = np.roll(vec, -1, axis=-2)
    dis_ = np.roll(dis, -1, axis=-2)
    # (B, A-1, 1)
    angle = np.einsum('ijk,ijk->ij', vec, vec_)[..., None] / (dis * dis_ + EPSILON)
    # (B, A-2, 1), (B, A-1, 1), (B, A-1, D)
    return np.arccos(angle[:, :-1, :]), dis, vec
 
def get_dihedral(crd):
    """ Get the dihedrals of the sequential coordinate.
    """
    # (B, A-2, 1), (B, A-1, 1), (B, A-1, D)
    angle, dis, vec_0 = get_angle(crd)
    # (B, A-1, D)
    vec_1 = np.roll(vec_0, -1, axis=-2)
    vec_2 = np.roll(vec_1, -1, axis=-2)
    vec_01 = np.cross(vec_0, vec_1)
    vec_12 = np.cross(vec_1, vec_2)
    vec_01 /= np.linalg.norm(vec_01, axis=-1, keepdims=True) + EPSILON
    vec_12 /= np.linalg.norm(vec_12, axis=-1, keepdims=True) + EPSILON
    # (B, A-1, 1)
    dihedral = np.einsum('ijk,ijk->ij', vec_01, vec_12)[..., None]
    # (B, A-3, 1), (B, A-2, 1), (B, A-1, 1)
    return np.arccos(dihedral[:, :-2, :]), angle, dis
 
def get_inner_crd(crd):
    """ Concat the distance, angles and dihedrals to get the inner coordinate.
    """
    # (B, A-3, 1), (B, A-2, 1), (B, A-1, 1)
    dihedral, angle, dis = get_dihedral(crd)
    # (B, A, 1)
    dihedral_ = np.pad(dihedral, ((0, 0), (3, 0), (0, 0)), mode='constant', constant_values=0)
    angle_ = np.pad(angle, ((0, 0), (2, 0), (0, 0)), mode='constant', constant_values=0)
    dis_ = np.pad(dis, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    # (B, A, 3)
    inner_crd = np.concatenate((dis_, angle_, dihedral_), axis=-1)
    return inner_crd
 
if __name__ == '__main__':
    B = 1
    A = 6
    D = 3
    # (B, A, D)
    # origin_crd = np.random.random((B, A, D))
    coord_data = pd.read_csv("/home/yanggk/Data/CO_Bert/Structure/3DCoord/CO-Ag-bridge-1-1.csv",dtype='float32').values
    # print(coord_data)
    coord_data = np.array(coord_data)
    # a = int(len(coord_data)/3)
    # print(coord_data)
    coord_data = coord_data.reshape(1, 14, 3)
    # print(coord_data)

    # (B, A, 3)
    # origin_crd = []
    # for i in range(0,len(coord_data),3):
    #     origin_crd.append((coord_data[i],coord_data[i+1],coord_data[i+2]))
    # origin_crd = np.array(origin_crd)
    # print(origin_crd.shape)
    icrd = get_inner_crd(coord_data)
    print(icrd)
