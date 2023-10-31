from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors, pyplot as plt

from common_functions import calculate_ICP, calculate_distance_difference, load_json_file


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
        cs = plt.tricontourf(distance_errors[:,0], distance_errors[:,1], distance_errors[:,2], cmap='jet', levels=40)
        ax2.set_xlabel('X, mm')
        ax2.set_ylabel('Y, mm')
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('dZ, mm')
        plt.grid()

    plt.show()


def draw_surfaces_in_3D(ls_sensor_points, phase_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ls_sensor_points[:,0], ls_sensor_points[:,1], ls_sensor_points[:,2])
    ax.scatter(phase_points[:,0], phase_points[:,1], phase_points[:,2])

    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    ax.set_zlabel('Z, mm')

    ax.view_init(elev=-75, azim=-89)

    plt.show()


def compare_stereo(phaso_path, stereo_path, calibration_data):
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
    print(f'\nЗагружено {stereo_points_3d.shape[0]} точек измерения стерео системы')

    # Загружаем параметры для перевода в мировую систему координат (датчика)
    R = np.array(calibration_data['RWorld'])
    t = np.array(calibration_data['TWorld']).reshape((3,1))

    print('\nОтфильтровываем выбросы в фазограмметрии по величине ошибки репроекции...')
    std_rprj1 = np.std(phase_errors[:,0])
    std_rprj2 = np.std(phase_errors[:,1])
    filter_condition = (phase_errors[:,0] < 3*std_rprj1) & (phase_errors[:,1] < 3*std_rprj2)
    phase_points_3d = phase_points_3d[filter_condition,:]
    phase_errors = phase_errors[filter_condition,:]
    print(f'\n{phase_points_3d.shape[0]} точек после фильтрации...')

    print('\nОтфильтровываем выбросы в стерео по величине ошибки репроекции...')
    std_rprj1 = np.std(reproj_errors1)
    std_rprj2 = np.std(reproj_errors2)
    condition = (reproj_errors1 < 3*std_rprj1) & (reproj_errors2 < 3*std_rprj2)
    stereo_points_2d1 = stereo_points_2d1[condition, :]
    stereo_points_2d2 = stereo_points_2d2[condition, :]
    stereo_points_3d = stereo_points_3d[condition, :]
    reproj_errors1 = reproj_errors1[condition]
    reproj_errors2 = reproj_errors2[condition]
    print(f'\n{stereo_points_3d.shape[0]} точек после фильтрации...')

    # Преобразуем измеренные точки с фазограмметрии в мировую систему координат
    phase_points_transformed = np.hstack((R, t)).dot(np.hstack((phase_points_3d, np.ones((phase_points_3d.shape[0], 1)))).T)
    phase_points_transformed = phase_points_transformed.T

    H = np.array([[0.978260, 0.050839, -0.201054, 489.272564],
                   [-0.203827, 0.414448, -0.886954, 1252.394387],
                   [0.038235, 0.908652, 0.415800, -469.539673],
                   [0.000000, 0.000000, 0.000000, 1.000000]])
    
    # stereo_points_3d_transformed = H1.dot(np.hstack((stereo_points_3d, np.ones((stereo_points_3d.shape[0], 1)))).T)
    # stereo_points_3d_transformed = stereo_points_3d_transformed.T

    # R, t, H = calculate_ICP(phase_points_transformed, stereo_points_3d_transformed, 100.0)

    # H = H @ H1

    # Преобразуем измеренные точки с фазограмметрии в мировую систему координат
    stereo_points_3d = H.dot(np.hstack((stereo_points_3d, np.ones((stereo_points_3d.shape[0], 1)))).T)
    stereo_points_3d = stereo_points_3d[:3,:].T

    return phase_points_transformed, phase_errors, stereo_points_3d, reproj_errors1, reproj_errors2


def draw_many_surfaces(ls_measures, phaso_measures, differences):
    meas_count = len(ls_measures)

    fig = plt.figure()

    for i in range(meas_count):
        phase_points = phaso_measures[i]
        ls_sensor_points = ls_measures[i]
        distance_errors = differences[i]

        # Create a normalization object
        norm = colors.Normalize(
            vmin=np.min([np.min(ls_sensor_points[:,2]), np.min(phase_points[:,2])]), 
            vmax=np.max([np.max(ls_sensor_points[:,2]), np.max(phase_points[:,2])])
        )

        ax = fig.add_subplot(meas_count, 3, 1 + i*3)
        cs = ax.tricontourf(ls_sensor_points[:,0], ls_sensor_points[:,1], ls_sensor_points[:,2], cmap='jet', levels=40, norm=norm)
        ax.set_xlabel('X, mm')
        ax.set_ylabel('Y, mm')
        cbar = fig.colorbar(cs, ax=ax)
        cbar.ax.set_ylabel('Z, mm')
        ax.invert_yaxis()
        plt.grid()

        ax2 = fig.add_subplot(meas_count, 3, 2 + i*3, sharex=ax, sharey=ax)
        cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phase_points[:,2], cmap='jet', levels=40, norm=norm)
        ax2.set_xlabel('X, mm')
        ax2.set_ylabel('Y, mm')
        cbar = fig.colorbar(cs, ax=ax2)
        cbar.ax.set_ylabel('Z, mm')
        plt.grid()

        ax2 = fig.add_subplot(meas_count, 3, 3 + i*3, sharex=ax, sharey=ax)
        cs = plt.tricontourf(distance_errors[:,0], distance_errors[:,1], distance_errors[:,2], cmap='jet', levels=40)
        ax2.set_xlabel('X, mm')
        ax2.set_ylabel('Y, mm')
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('dZ, mm')
        plt.grid()

    plt.show()


if __name__ == '__main__':

    PATH_TO_STEREO_MEASUREMENT = r'experimental_results\stereo'

    PATH_TO_PHASO_MEASUREMENT = r'experimental_results\2023-10-23'

    PATH_TO_PHASO_CALIBRATION = r'experimental_results\calibrated_data_phase4.json'

    path_to_phaso_date_folder = Path(PATH_TO_PHASO_MEASUREMENT)
    phaso_calibration = load_json_file(PATH_TO_PHASO_CALIBRATION)

    phaso_measurements_paths = []

    for measurement_folder in path_to_phaso_date_folder.iterdir():
        if measurement_folder.is_dir():
            phaso_measurements_paths.append(measurement_folder)

    path_to_stereo_data_folder = Path(PATH_TO_STEREO_MEASUREMENT)

    stereo_measurements_paths = list(path_to_stereo_data_folder.glob('result_*.json'))

    loaded_data = []

    for i, (phaso_path, stereo_path)  in enumerate(zip(phaso_measurements_paths, stereo_measurements_paths)):
        print(f'Обрабатываем поверхность {i+1} из {len(stereo_measurements_paths)}')
        
        try:
            phase_points_transformed, phase_errors, stereo_points_3d, reproj_errors1, reproj_errors2 = compare_stereo(phaso_path, stereo_path, phaso_calibration)

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

            # if len(reproj_errors1) > 3000 and np.std(reproj_errors1) < 1:
            #     plt.hist(reproj_errors1)
            #     plt.show()
            #     ax = plt.subplot(121)
            #     cs = plt.tricontourf(stereo_points_2d1[:,0], stereo_points_2d1[:,1], reproj_errors1, levels=40)
            #     plt.colorbar(cs)
            #     plt.xlim(0, 2048)
            #     plt.ylim(0, 1536)
            #     ax.invert_yaxis()
            #     ax = plt.subplot(122)
            #     cs = plt.tricontourf(stereo_points_2d2[:,0], stereo_points_2d2[:,1], reproj_errors2, levels=40)
            #     plt.colorbar(cs)
            #     plt.xlim(0, 2048)
            #     plt.ylim(0, 1536)
            #     ax.invert_yaxis()
            #     plt.show()

            # draw_surface_by_contours(stereo_points_3d, phase_points_transformed, distance_errors)
        except Exception as ex:
            print(ex)
            raise ex

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

    # df = df[(df.std_reproj1 < 1) & (df.std_reproj2 < 1)]

    df.distance_std.hist()
    plt.show()
