import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt


df = pd.read_pickle('compare_stereo_2023-11-04_16-36-16.pickle')
print(f'Загружено {df.shape[0]} измерений, ошибка репроекции min - {df.stereo_std_reproj1.min()}, max - {df.stereo_std_reproj1.max()}, std - {df.stereo_std_reproj1.std()}')

print('Выбираем 100 измерений с минимальной ошибкой репроекции')
df = df.assign(sum=df.loc[:,df.columns[4:6]].sum(axis=1)).sort_values(by='sum').iloc[:100, :-1]
# df = df[df.stereo_std_reproj1 < 0.5]
print(f'Осталось {df.shape[0]} измерений, максимальная ошибка репроекции {df.stereo_std_reproj1.iloc[-1]: .3f}, {df.stereo_std_reproj2.iloc[-1]: .3f}')

x = np.linspace(100, 420, 25)
y = np.linspace(50, 380, 25)
xx, yy = np.meshgrid(x, y)

z_list = []
dz_list = []
corel_list= []
reproj1_list = []
reproj2_list = []
z_points = []
dz_points = []
min_max_z_list = []

k = 0

for z, dz, corel_xy, corel, rpj1, rpj2 in zip(df.phase_points, df.distance_errors, df.stereo_points, df.corelation_coef, df.reproj1, df.reproj2):
    min_max_z_list.append((np.min(z[:,2]), np.max(z[:,2])))
    
    # Интерполируем трехмерные точки из данных модели
    interp_z = interpolate.LinearNDInterpolator(z[:,:2], z[:,2])
    interp_dz = interpolate.LinearNDInterpolator(dz[:,:2], dz[:,2])
    interp_corel = interpolate.LinearNDInterpolator(corel_xy[:,:2], corel)
    interp_reproj1 = interpolate.LinearNDInterpolator(corel_xy[:,:2], rpj1)
    interp_reproj2 = interpolate.LinearNDInterpolator(corel_xy[:,:2], rpj2)

    # Получаем интерполяцию из модельных данных для координат измеренных точек
    z = interp_z(xx, yy)
    dz = interp_dz(xx, yy)
    corel = interp_corel(xx, yy)
    rpj1 = interp_reproj1(xx, yy)
    rpj2 = interp_reproj2(xx, yy)

    z_list.append(z)
    dz_list.append(dz)
    corel_list.append(corel)
    reproj1_list.append(rpj1)
    reproj2_list.append(rpj2)

    print(f'{k}')
    k = k + 1

mean_z = np.mean(np.array(z_list), axis=0)
std_dz = np.std(np.array(dz_list), axis=0)
mean_dz = np.mean(np.abs(np.array(dz_list)), axis=0)
std_corel = np.std(np.array(corel_list), axis=0)
corel_list = np.mean(np.array(corel_list), axis=0)
reproj1_list = np.mean(np.array(reproj1_list), axis=0)
reproj2_list = np.mean(np.array(reproj2_list), axis=0)

min_max_z = np.array(min_max_z_list)
print(f'Размах высот в эксперименте от {np.min(min_max_z[:,0])} мм до {np.max(min_max_z[:,1])} мм')

reproj1_list[reproj1_list > 1.5] = np.NaN
reproj2_list[reproj1_list > 1.5] = np.NaN


plt.rcParams.update({'font.size': 14})

ax = plt.subplot(231)
cs = plt.contourf(xx, yy, mean_dz, cmap='jet', levels=40)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel(r'Mean $\left| \Delta Z \right|$, mm')
ax.invert_yaxis()
plt.xlabel('X, mm')
plt.ylabel('Y, mm')
plt.grid()

ax = plt.subplot(232)
cs = plt.contourf(xx, yy, std_dz, cmap='jet', levels=40)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel(r'RMS $\left| \Delta Z \right|$, mm')
ax.invert_yaxis()
plt.xlabel('X, mm')
plt.ylabel('Y, mm')
plt.grid()

ax = plt.subplot(233)
cs = plt.contourf(xx, yy, mean_z, cmap='jet', levels=40)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel('Mean Z, mm')
ax.invert_yaxis()
plt.xlabel('X, mm')
plt.ylabel('Y, mm')
plt.grid()

ax = plt.subplot(234)
cs = plt.contourf(xx, yy, corel_list, cmap='jet', levels=40)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel('Correlation coefficient, rel. units')
ax.invert_yaxis()
plt.xlabel('X, mm')
plt.ylabel('Y, mm')
plt.grid()

ax = plt.subplot(235)
cs = plt.contourf(xx, yy, reproj1_list, cmap='jet', levels=40)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel('Reprojection error, pxls')
ax.invert_yaxis()
plt.xlabel('X, mm')
plt.ylabel('Y, mm')
plt.grid()

ax = plt.subplot(236)
cs = plt.contourf(xx, yy, reproj2_list, cmap='jet', levels=40)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel('Reprojection error, pxls')
ax.invert_yaxis()
plt.xlabel('X, mm')
plt.ylabel('Y, mm')
plt.grid()

plt.show()