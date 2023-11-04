import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def euler_angles_to_rotation_matrix(
    alpha1: float, alpha2: float, alpha3: float
) -> np.array:
    '''
    Get Euler angles from rotation matrix.

    From simpleICP https://github.dev/pglira/simpleICP/blob/master/python/simpleicp/mathutils.py
    '''

    R = np.array(
        [
            [
                np.cos(alpha2) * np.cos(alpha3),
                -np.cos(alpha2) * np.sin(alpha3),
                np.sin(alpha2),
            ],
            [
                np.cos(alpha1) * np.sin(alpha3)
                + np.sin(alpha1) * np.sin(alpha2) * np.cos(alpha3),
                np.cos(alpha1) * np.cos(alpha3)
                - np.sin(alpha1) * np.sin(alpha2) * np.sin(alpha3),
                -np.sin(alpha1) * np.cos(alpha2),
            ],
            [
                np.sin(alpha1) * np.sin(alpha3)
                - np.cos(alpha1) * np.sin(alpha2) * np.cos(alpha3),
                np.sin(alpha1) * np.cos(alpha3)
                + np.cos(alpha1) * np.sin(alpha2) * np.sin(alpha3),
                np.cos(alpha1) * np.cos(alpha2),
            ],
        ]
    )

    return R


df = pd.read_pickle('reference_calculation_results_2023-11-04_15-26-58.pickle')
print(f'Загружено {df.shape[0]} измерений, невязка min - {df.std_residuals.min()}, max - {df.std_residuals.max()}, std - {df.std_residuals.std()}')

print(f'Фильтруем по величине невязки')
df = df[df.std_residuals < 0.2]
print(f'Осталось {df.shape[0]} измерений')
# print(f'Максимальная невязка {df.std_residuals.iloc[-1]: .3f}, {df.stereo_std_reproj2.iloc[-1]: .3f}')

df_filtered = df.sort_values('std_residuals')[:100]

plt.subplot(231)
plt.hist(df_filtered.alpha1)
plt.xlabel('Alpha1, deg')
plt.ylabel('Count')

plt.subplot(232)
plt.hist(df_filtered.alpha2)
plt.xlabel('Alpha2, deg')
plt.ylabel('Count')

plt.subplot(233)
plt.hist(df_filtered.alpha3)
plt.xlabel('Alpha3, deg')
plt.ylabel('Count')

plt.subplot(234)
plt.hist(df_filtered.tx)
plt.xlabel('Tx, deg')
plt.ylabel('Count')

plt.subplot(235)
plt.hist(df_filtered.ty)
plt.xlabel('Ty, deg')
plt.ylabel('Count')

plt.subplot(236)
plt.hist(df_filtered.tz)
plt.xlabel('Alpha3, deg')
plt.ylabel('Tz')
plt.show()

result = (df_filtered.alpha1.mean(),
       df_filtered.alpha2.mean(),
       df_filtered.alpha3.mean(),
       df_filtered.tx.mean(),
       df_filtered.ty.mean(),
       df_filtered.tz.mean()
    )

print(result)

rot_mtx = euler_angles_to_rotation_matrix(np.deg2rad(result[0]), np.deg2rad(result[1]), np.deg2rad(result[2]))
trns_vec = np.array([[result[3]], [result[4]], [result[5]]])

H = np.vstack((np.hstack((rot_mtx, trns_vec)), np.array([0, 0, 0 ,1])))

with np.printoptions(suppress=True):
    print(H)
