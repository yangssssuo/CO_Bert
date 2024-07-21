import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd

def cartesian_to_internal(coordinates):
    # 计算所有原子之间的距离，得到键长
    distances = pdist(coordinates)
    
    # 计算键角
    angles = []
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            for k in range(j+1, len(coordinates)):
                ba = coordinates[i] - coordinates[j]
                bc = coordinates[k] - coordinates[j]
                
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                
                angles.append(angle)
    
    # 计算二面角
    dihedrals = []
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            for k in range(j+1, len(coordinates)):
                for l in range(k+1, len(coordinates)):
                    b1 = -1.0*(coordinates[j] - coordinates[i])
                    b2 = coordinates[k] - coordinates[j]
                    b3 = coordinates[l] - coordinates[k]
                    
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    
                    n1_x_n2 = np.cross(n1, n2)
                    
                    u1 = n1 - np.dot(n1, b2)*b2/np.linalg.norm(b2)**2
                    u3 = n2 - np.dot(n2, b2)*b2/np.linalg.norm(b2)**2
                    
                    cosine_angle = np.dot(u1, u3)
                    sine_angle = np.dot(n1_x_n2, b2)*1.0/np.linalg.norm(b2)
                    
                    dihedral = np.arctan2(sine_angle, cosine_angle)
                    
                    dihedrals.append(dihedral)
    
    return distances, np.array(angles), np.array(dihedrals)

coord_data = pd.read_csv("/home/yanggk/Data/CO_Bert/Structure/3DCoord/CO-Ag-bridge-1-1.csv",dtype='float32').values
    # print(coord_data)
coordinates = np.array(coord_data)
print(coord_data)
coordinates = coordinates.reshape(14, 3)
print(coordinates)
distances, angles, dihedrals = cartesian_to_internal(coordinates)

print("键长: ", distances,'数量：',len(distances))
print("键角: ", angles,'数量：',len(angles))
print("二面角: ", dihedrals,'数量：',len(dihedrals))


def internal_to_cartesian(distances, angles, dihedrals):
    # 初始化坐标数组
    coordinates = np.zeros((14, 3))
    
    # 第一个原子位于原点
    coordinates[0] = np.array([0, 0, 0])
    
    # 第二个原子位于x轴上，距离第一个原子为键长
    coordinates[1] = np.array([distances[0], 0, 0])
    
    # 第三个原子位于x-y平面上，通过键长和键角确定位置
    coordinates[2] = np.array([
        distances[1]*np.cos(angles[0]),
        distances[1]*np.sin(angles[0]),
        0
    ])
    
    # 其他原子的位置通过键长、键角和二面角确定
    for i in range(3, 14):
        coordinates[i] = np.array([
            distances[i]*np.cos(angles[i]),
            distances[i]*(np.cos(dihedrals[i])*np.sin(angles[i])),
            distances[i]*(np.sin(dihedrals[i])*np.sin(angles[i]))
        ])
    
    return coordinates

# 假设你的14个原子的键长、键角和二面角存储在numpy数组distances, angles, dihedrals中
# 这里只是一个示例，你需要用你的实际键长、键角和二面角替换这里的随机数

coordinates = internal_to_cartesian(distances, angles, dihedrals)

print("三维坐标: ", coordinates)
