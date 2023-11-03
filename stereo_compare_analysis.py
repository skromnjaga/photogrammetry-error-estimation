import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt


df = pd.read_pickle('compare_stereo_2023-11-03_10-50-09.pickle')

print('Выбираем 100 измерений с минимальной ошибкой репроекции')

# df = df.assign(sum=df.loc[:,df.columns[4:6]].sum(axis=1)).sort_values(by='sum').iloc[:100, :-1]
df = df[df.stereo_std_reproj1 < 0.5]

print(f'Максимальная ошибка репроекции {df.stereo_std_reproj1.iloc[-1]: .3f}, {df.stereo_std_reproj2.iloc[-1]: .3f}')

x = np.linspace(100, 420, 25)
y = np.linspace(50, 380, 25)
xx, yy = np.meshgrid(x, y)

z_list = []
dz_list = []
corel_list= []

k = 0

for z, dz, corel_xy, corel in zip(df.phase_points, df.distance_errors, df.stereo_points, df.corelation_coef):
    # Интерполируем трехмерные точки из данных модели
    interp_z = interpolate.LinearNDInterpolator(z[:,:2], z[:,2])
    interp_dz = interpolate.LinearNDInterpolator(dz[:,:2], dz[:,2])
    interp_corel = interpolate.LinearNDInterpolator(corel_xy[:,:2], corel)

    # Получаем интерполяцию из модельных данных для координат измеренных точек
    z = interp_z(xx, yy)
    dz = interp_dz(xx, yy)
    corel = interp_corel(xx, yy)

    z_list.append(z)
    dz_list.append(dz)
    corel_list.append(corel)

    print(f'{k}')
    k = k + 1

mean_z = np.mean(np.array(z_list), axis=0)
std_dz = np.std(np.array(dz_list), axis=0)
mean_dz = np.mean(np.abs(np.array(dz_list)), axis=0)
std_corel = np.std(np.array(corel_list), axis=0)
corel_list = np.mean(np.array(corel_list), axis=0)

ax = plt.subplot(231)
cs = plt.contourf(xx, yy, mean_dz, cmap='jet', levels=40)
plt.colorbar(cs)
ax.invert_yaxis()
plt.grid()

ax = plt.subplot(232)
cs = plt.contourf(xx, yy, std_dz, cmap='jet', levels=40)
plt.colorbar(cs)
ax.invert_yaxis()
plt.grid()

ax = plt.subplot(233)
cs = plt.contourf(xx, yy, mean_z, cmap='jet', levels=40)
plt.colorbar(cs)
ax.invert_yaxis()
plt.grid()

ax = plt.subplot(234)
cs = plt.contourf(xx, yy, corel_list, cmap='jet', levels=40)
plt.colorbar(cs)
ax.invert_yaxis()
plt.grid()

ax = plt.subplot(235)
cs = plt.contourf(xx, yy, std_corel, cmap='jet', levels=40)
plt.colorbar(cs)
ax.invert_yaxis()
plt.grid()

plt.show()