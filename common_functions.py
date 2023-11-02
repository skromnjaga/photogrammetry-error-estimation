import json
import pathlib
import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, linalg
from simpleicp import PointCloud, SimpleICP


def get_now_time(fmt='%Y-%m-%d_%H-%M-%S') -> str:
    '''
    Возвращает текущую дату и время в заданном формате (по умолчанию Y-m-d_H-M-S) 
    '''
    return datetime.datetime.now().strftime(fmt)


def get_measurements_paths(measurements_path: str, skip_folders_with_result: bool = True, result_name: str=None) -> List[pathlib.Path]:
    '''
    Возвращает список путей к папкам с результатами измерений
    ''' 
    # Get all folders with measurements
    path_to_measurements_folder = pathlib.Path(measurements_path)

    measurements_paths = []

    for date_folder in path_to_measurements_folder.iterdir():
        if  date_folder.is_dir():
            for measurement_folder in date_folder.iterdir():
                if measurement_folder.is_dir():
                    if not skip_folders_with_result or len(list(measurement_folder.glob(f'{result_name}*'))) == 0:
                        measurements_paths.append(measurement_folder)
    
    return measurements_paths


def load_json_file(file_name: str, exit_on_except: bool=True) -> Dict:
    '''
    Загружает файл JSON по указаному пути
    '''
    try:
        data = json.load(open(file_name))
    except:
        print(f'Указанный файл {file_name} не найден!')
        if exit_on_except:
            exit(1)
    return data


def fit_to_plane(x, y, z):
    # From https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    tmp_A = []
    tmp_b = []

    for i in range(z.shape[0]):
        tmp_A.append([x[i], y[i], 1])
        tmp_b.append(z[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit, residual, rnk, s = linalg.lstsq(A, b)
    return fit


def calculate_ICP(points1: np.ndarray, points2: np.ndarray, initial_params:Tuple[float] = None, max_overlap_distance: float = 1000):
    '''
    Calculate ICP for two point clouds
    '''
    X_mov = pd.DataFrame({
        'x': points2[:,0],
        'y': points2[:,1],
        'z': points2[:,2],
    })

    X_fix = pd.DataFrame({
        'x': points1[:,0],
        'y': points1[:,1],
        'z': points1[:,2],
    })

    pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
    pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

    # Create simpleICP object, add point clouds, and run algorithm!
    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)

    if initial_params is None:
        initial_params = (0, 0, 0, 0, 0, 0)

    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(rbp_observed_values=initial_params, max_overlap_distance=max_overlap_distance)

    return H, X_mov_transformed, rigid_body_transformation_params, distance_residuals


def filter_outliers(data):
    upper_quartile = np.percentile(data[:,2], 75)
    lower_quartile = np.percentile(data[:,2], 25)
    outlierConstant = 1.5
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    resultList = []
    for i in range(data.shape[0]):
        if data[i, 2] >= quartileSet[0] and data[i, 2] <= quartileSet[1]:
            resultList.append(data[i])

    return np.array(resultList)


def calculate_distance_difference(points1, points2):

    if points1.shape[0] < points2.shape[0]:
        points_to_interpolate = points1
        points_for_interpolation = points2
    else:
        points_to_interpolate = points2
        points_for_interpolation = points1

    # Интерполируем трехмерные точки из данных модели
    interp = interpolate.LinearNDInterpolator(points_for_interpolation[:,:2], points_for_interpolation[:,2])

    # Получаем интерполяцию из модельных данных для координат измеренных точек
    phase_interpolated_for_ls_points = interp(points_to_interpolate[:,0], points_to_interpolate[:,1])

    # # Формируем данные для фильтрации на выбросы
    # process_data = list(zip(x2, y2, z2))

    # Рассчитываем отличие модельных точек от измеренных
    distance_errors = points_to_interpolate[:,2] - phase_interpolated_for_ls_points

    condition = ~np.isnan(distance_errors)
    distance_errors_filtered = distance_errors[condition]

    distance_errors = np.zeros((distance_errors_filtered.shape[0], 3))
    distance_errors = points1[condition,:]
    distance_errors[:,2] = distance_errors_filtered

    # Фильтруем выбросы по интерквартильному размаху IQR
    distance_errors_filtered = filter_outliers(distance_errors)

    # # Получаем интерполяцию из модельных данных без координат выбросов
    # x3 = [point[0] for point in process_data_filtered]
    # y3 = [point[1] for point in process_data_filtered]
    # z3 = [point[2] for point in process_data_filtered]
    # Z3 = interp(x3, y3)

    # # Интерполируем зависимость ошибки по координатам
    # distance_errors_interp = interpolate.LinearNDInterpolator(list(zip(x3, y3)), distance_errors)

    # x = np.linspace(min(x3), max(x3))
    # y = np.linspace(min(y3), max(y3))
    # X3, Y3 = np.meshgrid(x, y)

    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Error for Z estimation, mm')
    # ax1.boxplot([z2-Z2, z3-Z3])
    
    return distance_errors_filtered
