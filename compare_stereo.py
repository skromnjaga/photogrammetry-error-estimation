from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors, pyplot as plt

from common_functions import calculate_ICP, calculate_distance_difference, get_now_time, load_json_file


def draw_surface_by_contours(ls_sensor_points, phase_points, distance_errors):
    # Create a normalization object
    norm = colors.Normalize(
        vmin=np.min([np.min(ls_sensor_points[:,2]), np.min(phase_points[:,2])]), 
        vmax=np.max([np.max(ls_sensor_points[:,2]), np.max(phase_points[:,2])])
    )

    plt.rcParams.update({'font.size': 16})

    fig = plt.figure()
    ax = fig.add_subplot(131)
    cs = ax.tricontourf(ls_sensor_points[:,0], ls_sensor_points[:,1], ls_sensor_points[:,2], cmap='jet', levels=40, norm=norm)
    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs, ax=ax)
    cbar.ax.set_ylabel('Z, mm')
    ax.invert_yaxis()
    plt.grid()

    ax2 = fig.add_subplot(132, sharex=ax, sharey=ax)
    cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phase_points[:,2], cmap='jet', levels=40, norm=norm)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs, ax=ax2)
    cbar.ax.set_ylabel('Z, mm')
    plt.grid()

    if distance_errors.shape[0] > 3:
        ax2 = fig.add_subplot(133, sharex=ax, sharey=ax)
        cs = plt.tricontourf(distance_errors[:,0], distance_errors[:,1], np.abs(distance_errors[:,2]), cmap='jet', levels=40)
        ax2.set_xlabel('X, mm')
        ax2.set_ylabel('Y, mm')
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('dZ, mm')
        plt.grid()

    plt.show()


def compare_stereo_pair(phaso_path, stereo_path, calibration_data):
    # Загружаем данные фазограмметрического измерения
    phaso_measurement_path = Path(phaso_path)
    
    phase_meas = load_json_file(list(phaso_measurement_path.glob('phasogrammetry_measurement.json'))[0])

    # Get X, Y, Z data from file
    phase_points_2d1 = np.array([[point['r1'], point['c1']] for point in phase_meas['measured_points']])
    phase_points_2d2 = np.array([[point['r2'], point['c2']] for point in phase_meas['measured_points']])
    phase_points_3d = np.array([[point['x'], point['y'], point['z']] for point in phase_meas['measured_points']])
    phase_errors = np.array([[point['reproj_err1'], point['reproj_err2'], point['phase_err1'], point['phase_err2']] for point in phase_meas['measured_points']])

    # Загружаем данные стерео измерения
    data = load_json_file(stereo_path)

    stereo_points_2d1 = np.array([[point[0], point[1]] for point in data['PointsOnFirstImage']])
    stereo_points_2d2 = np.array([[point[0], point[1]] for point in data['PointsOnSecondImage']])
    stereo_points_3d = np.array([[point[0], point[1], point[2]] for point in data['TriangulatedPointsArray']])
    reproj_errors1 = np.array([value for value in data['FirstImageReprojectionErrors']])
    reproj_errors2 = np.array([value for value in data['SecondImageReprojectionErrors']])

    print(f'\nЗагружено {phase_points_3d.shape[0]} точек измерения фазограмметрической системы')
    phaso_std_rprj1 = np.std(phase_errors[:,0])
    phaso_std_rprj2 = np.std(phase_errors[:,1])
    print(f'STD ошибки репроекции {phaso_std_rprj1: .3f} и {phaso_std_rprj2: .3f}')   
    
    print(f'\nЗагружено {stereo_points_3d.shape[0]} точек измерения стерео системы')
    stero_std_rprj1 = np.std(reproj_errors1)
    stero_std_rprj2 = np.std(reproj_errors2)
    print(f'STD ошибки репроекции {stero_std_rprj1: .3f} и {stero_std_rprj2: .3f}')

    print('\nОтфильтровываем выбросы в фазограмметрии по величине ошибки репроекции...')
    filter_condition = (phase_errors[:,0] < 3*phaso_std_rprj1) & (phase_errors[:,1] < 3*phaso_std_rprj2)
    phase_points_3d = phase_points_3d[filter_condition,:]
    phase_errors = phase_errors[filter_condition,:]
    print(f'{phase_points_3d.shape[0]} точек после фильтрации...')

    # Загружаем параметры для перевода в мировую систему координат (датчика)
    R = np.array(calibration_data['RWorld'])
    t = np.array(calibration_data['TWorld']).reshape((3,1))

    # Преобразуем измеренные точки с фазограмметрии в мировую систему координат
    phase_points_transformed = np.hstack((R, t)).dot(np.hstack((phase_points_3d, np.ones((phase_points_3d.shape[0], 1)))).T)
    phase_points_transformed = phase_points_transformed.T

    print('\nОтфильтровываем точки в фазограмметрии по пределам измерений...')
    filter_condition = (phase_points_transformed[:,1] > 20) & (phase_points_transformed[:,2] > 10) & (phase_points_transformed[:,2] < 60)
    phase_points_transformed = phase_points_transformed[filter_condition,:]
    phase_errors = phase_errors[filter_condition,:]
    print(f'{phase_points_transformed.shape[0]} точек после фильтрации...')

    print('\nОтфильтровываем выбросы в стерео по величине ошибки репроекции...')
    condition = (reproj_errors1 < 6*stero_std_rprj1) & (reproj_errors2 < 6*stero_std_rprj2)
    stereo_points_2d1 = stereo_points_2d1[condition, :]
    stereo_points_2d2 = stereo_points_2d2[condition, :]
    stereo_points_3d = stereo_points_3d[condition, :]
    reproj_errors1 = reproj_errors1[condition]
    reproj_errors2 = reproj_errors2[condition]
    print(f'{stereo_points_3d.shape[0]} точек после фильтрации...')

    if len(stereo_points_3d) == 0:
        return phase_points_transformed, phase_errors, None, reproj_errors1, reproj_errors2

    if CALCULATE_ICP:
        _, stereo_points_3d, _, _ = calculate_ICP(phase_points_transformed, stereo_points_3d, initial_params=INIT_PARAMS, max_overlap_distance=2000.0)
    else:
        # Преобразуем измеренные точки ссо стереосистемы в мировую систему координат
        stereo_points_3d = H.dot(np.hstack((stereo_points_3d, np.ones((stereo_points_3d.shape[0], 1)))).T)
        stereo_points_3d = stereo_points_3d[:3,:].T

    return phase_points_transformed, phase_errors, stereo_points_3d, reproj_errors1, reproj_errors2


def compare_stereo():    
    path_to_phaso_date_folder = Path(PATH_TO_PHASO_MEASUREMENT)

    phaso_calibration = load_json_file(PATH_TO_PHASO_CALIBRATION)

    phaso_measurements_paths = []

    for measurement_folder in path_to_phaso_date_folder.iterdir():
        if measurement_folder.is_dir():
            phaso_measurements_paths.append(measurement_folder)

    path_to_stereo_data_folder = Path(PATH_TO_STEREO_MEASUREMENT)

    stereo_measurements_paths = list(path_to_stereo_data_folder.glob('result_*.json'))

    loaded_data = []

    k = 0

    for i in range(len(phaso_measurements_paths)):
        stereo_path = stereo_measurements_paths[k]

        stereo_number = int(stereo_path.name.split('.')[0].split('_')[1])
        if stereo_number != i:
            print(f'Не найден стерео результат #{i}, переходим к следующему...')
            continue

        phaso_path = phaso_measurements_paths[i]

        print(f'Обрабатываем поверхность {i+1} из {len(phaso_measurements_paths)}')
        
        try:
            phase_points_transformed, phase_errors, stereo_points_3d, reproj_errors1, reproj_errors2 = compare_stereo_pair(phaso_path, stereo_path, phaso_calibration)

            if stereo_points_3d is not None:
                distance_errors = calculate_distance_difference(stereo_points_3d, phase_points_transformed)
                        
                loaded_data.append([
                    len(phase_points_transformed),
                    len(stereo_points_3d),
                    np.std(phase_errors[:,0]),
                    np.std(phase_errors[:,1]),
                    np.std(reproj_errors1), 
                    np.std(reproj_errors2),
                    np.std(distance_errors[:,2]),
                    phase_points_transformed,
                    phase_errors,
                    stereo_points_3d,
                    reproj_errors1,
                    reproj_errors2,
                    distance_errors,
                ])

                # draw_surface_by_contours(stereo_points_3d, phase_points_transformed, distance_errors)
        except Exception as ex:
            print(ex)
            raise ex
        
        k = k + 1

    df = pd.DataFrame(loaded_data, columns=(
        'phase_points_num',
        'stereo_points_num', 
        'phase_std_reproj1', 
        'phase_std_reproj2',
        'stereo_std_reproj1', 
        'stereo_std_reproj2',
        'distance_std',
        'phase_points',
        'phase_errors', 
        'stereo_points', 
        'reproj1', 
        'reproj2',
        'distance_errors'
    ))

    df.to_pickle(f'compare_stereo_{get_now_time()}.pickle')


if __name__ == '__main__':

    PATH_TO_STEREO_MEASUREMENT = r'experimental_results\stereo5'

    PATH_TO_PHASO_MEASUREMENT = r'experimental_results\2023-11-01'

    PATH_TO_PHASO_CALIBRATION = r'experimental_results\calibrated_data_phase5.json'

    INIT_PARAMS = (62.580111439506275, -11.976132973600278, -3.207140605473713, 496.319310041048, 1259.5447980833978, -480.61173541932527)

    H = np.array([[   0.97670201,    0.05472822,   -0.20750422,  496.31931004],
                  [  -0.20966734,    0.44948188,   -0.86833498, 1259.54479808],
                  [   0.04574695,    0.89161139,    0.45048458, -480.61173542],
                  [   0.,            0.,            0.,            1.,        ]])
    
    CALCULATE_ICP = False

    compare_stereo()
