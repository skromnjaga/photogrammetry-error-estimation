from pathlib import Path

import numpy as np
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

def compare_imitator(path_to_measurement, calibration_data):
    
    measurement_path = Path(path_to_measurement)
    
    ls_meas = load_json_file(list(measurement_path.glob('ls_measurement*.json'))[0])

    # Get X, Y, Z data from file
    ls_sensor_points = np.array([[point['x'], point['y'], point['z']] for point in ls_meas['measured_points'] if point['z'] != -1])

    phase_meas = load_json_file(list(measurement_path.glob('phasogrammetry_measurement.json'))[0])

    # Get X, Y, Z data from file
    phase_points_2d1 = np.array([[point['r1'], point['c1']] for point in phase_meas['measured_points']])
    phase_points_2d2 = np.array([[point['r2'], point['c2']] for point in phase_meas['measured_points']])
    phase_points_3d = np.array([[point['x'], point['y'], point['z']] for point in phase_meas['measured_points']])
    phase_errors = np.array([[point['reproj_err1'], point['reproj_err2'], point['phase_err1'], point['phase_err2']] for point in phase_meas['measured_points']])

    # Загружаем параметры для перевода в мировую систему координат (датчика)
    R = np.array(calibration_data['RWorld'])
    t = np.array(calibration_data['TWorld']).reshape((3,1))

    # Преобразуем измеренные точки с фазограмметрии в мировую систему координат
    phase_points_transformed = np.hstack((R, t)).dot(np.hstack((phase_points_3d, np.ones((phase_points_3d.shape[0], 1)))).T)
    phase_points_transformed = phase_points_transformed.T

    H1 = np.vstack([np.hstack((R, t)), np.array([0, 0, 0, 1])])

    R, t, H = calculate_ICP(ls_sensor_points, phase_points_transformed, 10.0)

    H = H @ H1

    # Преобразуем измеренные точки с фазограмметрии в мировую систему координат
    phase_points_transformed = H.dot(np.hstack((phase_points_3d, np.ones((phase_points_3d.shape[0], 1)))).T)
    phase_points_transformed = phase_points_transformed.T

    print(f'\nLoaded {ls_sensor_points.shape[0]} points from LS')
    print(f'\nLoaded {phase_points_transformed.shape[0]} points from PHASO')

    print('\nTry to filter outliers with reprojection error...')
    filter_condition = (phase_errors[:,0] < 3*np.std(phase_errors[:,0])) & (phase_errors[:,1] < 3*np.std(phase_errors[:,1]))
    phase_points_transformed = phase_points_transformed[filter_condition,:]
    phase_errors = phase_errors[filter_condition,:]
    print(f'\n{phase_points_transformed.shape[0]} points after filtration...')

    return ls_sensor_points, phase_points_transformed, phase_errors


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

    PATH_TO_MEASUREMENT = r'experimental_results\2023-10-20'

    PATH_TO_PHASO_CALIBRATION = r'experimental_results\calibrated_data_phase4.json'

    path_to_date_folder = Path(PATH_TO_MEASUREMENT)
    phaso_calibration = load_json_file(PATH_TO_PHASO_CALIBRATION)

    measurements_paths = []

    for measurement_folder in path_to_date_folder.iterdir():
        if measurement_folder.is_dir():
            measurements_paths.append(measurement_folder)
            
    ls_measures = []
    phaso_measures = []
    differences = []

    for i, path in enumerate(measurements_paths[5:]):
        print(f'Обрабатываем поверхность {i+1} из {len(measurements_paths)}')
        try:
            ls_sensor_points, phase_points_transformed, phase_errors = compare_imitator(path, phaso_calibration)

            distance_errors = calculate_distance_difference(ls_sensor_points, phase_points_transformed)

            ls_measures.append(ls_sensor_points)
            phaso_measures.append(phase_points_transformed)
            differences.append(distance_errors)

            # draw_surface_by_contours(ls_sensor_points, phase_points_transformed, distance_errors)
        except Exception as ex:
            print(ex)


    draw_many_surfaces(ls_measures, phaso_measures, differences)

    differences = np.array([el[:,2] for el in differences])

    differences = differences[1:] * 10**3 # в мкм

    plt.rcParams.update({'font.size': 16})
    plt.boxplot(differences)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    plt.xlabel('Measurement number')
    plt.ylabel('Distance to fit plane, $ \mu $m')
    plt.grid()
    plt.show()

    # [82.31858388021796, 72.25510026317396, 121.56024372866983, 107.82389553519879, 80.33831907688214, 79.89408024983487, 122.08212530950004, 138.17813952887934, 101.12381229173893]
    