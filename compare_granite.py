from pathlib import Path

import numpy as np
from matplotlib import colors, pyplot as plt

from common_functions import calculate_ICP, calculate_distance_difference, fit_to_plane, load_json_file


def draw_surface_by_contours(ls_sensor_points, phase_points, ls_distance_to_plane, phaso_distance_to_plane, phase_errors, distance_errors):
# Create a normalization object
    norm = colors.Normalize(
        vmin=np.min([np.min(ls_sensor_points[:,2]), np.min(phase_points[:,2])]), 
        vmax=np.max([np.max(ls_sensor_points[:,2]), np.max(phase_points[:,2])])
    )

    plt.rcParams.update({'font.size': 12})

    fig = plt.figure()
    ax = fig.add_subplot(241)
    cs = ax.tricontourf(ls_sensor_points[:,0], ls_sensor_points[:,1], ls_sensor_points[:,2], cmap='jet', levels=40, norm=norm)
    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs, ax=ax)
    cbar.ax.set_ylabel('Z, mm')
    ax.invert_yaxis()
    plt.grid()

    ax2 = fig.add_subplot(242, sharex=ax, sharey=ax)
    cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phase_points[:,2], cmap='jet', levels=40, norm=norm)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs, ax=ax2)
    cbar.ax.set_ylabel('Z, mm')
    plt.grid()

    ax2 = fig.add_subplot(245, sharex=ax, sharey=ax)
    cs = plt.tricontourf(distance_errors[:,0], distance_errors[:,1], distance_errors[:,2], cmap='jet', levels=40)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('dZ, mm')
    ax2.invert_yaxis()
    plt.grid()

    ax2 = fig.add_subplot(246, sharex=ax, sharey=ax)
    cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phaso_distance_to_plane, cmap='jet', levels=40)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('dZ, mm')
    ax2.invert_yaxis()
    plt.grid()

    ax2 = fig.add_subplot(243, sharex=ax, sharey=ax)
    cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phase_errors[:,2], cmap='jet', levels=40)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Phase error, radians')
    ax2.invert_yaxis()
    plt.grid()

    ax2 = fig.add_subplot(247, sharex=ax, sharey=ax)
    cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phase_errors[:,3], cmap='jet', levels=40)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Phase error, radians')
    ax2.invert_yaxis()
    plt.grid()

    ax2 = fig.add_subplot(244, sharex=ax, sharey=ax)
    cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phase_errors[:,0], cmap='jet', levels=40)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Reprojection error, pxls')
    ax2.invert_yaxis()
    plt.grid()

    ax2 = fig.add_subplot(248, sharex=ax, sharey=ax)
    cs = plt.tricontourf(phase_points[:,0], phase_points[:,1], phase_errors[:,1], cmap='jet', levels=40)
    ax2.set_xlabel('X, mm')
    ax2.set_ylabel('Y, mm')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Reprojection error, pxls')
    ax2.invert_yaxis()
    plt.grid()

    plt.show()


def draw_rms_distance_to_plane(std_ls, std_phaso):
    rng = np.array(list(range(len(std_ls)))) + 1

    plt.rcParams.update({'font.size': 12})

    [plt.bar(rng, std_ls, align='edge', width=-0.3, label='Laser sensor'), plt.bar(rng, std_phaso, align='edge', width=0.3, label='Phasogrammetry')]

    plt.xlabel('Measurement number')
    plt.yticks(range(1, 10))
    plt.ylabel('RMS distance to fit plane, $ \mu $m')
    plt.yticks(range(0, 65, 5))
    plt.grid()
    plt.legend()
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


def compare_granite(path_to_measurement, calibration_data):
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

    # R, t = calculate_ICP(ls_sensor_points, phase_points_transformed, 2.0)

    # # Преобразуем измеренные точки с фазограмметрии в мировую систему координат
    # phase_points_transformed = np.hstack((R, t)).dot(np.hstack((phase_points_transformed, np.ones((phase_points_transformed.shape[0], 1)))).T)
    # phase_points_transformed = phase_points_transformed.T

    print(f'\nFit {ls_sensor_points.shape[0]} points from LS to plane')
    fit2 = fit_to_plane(ls_sensor_points[:,0], ls_sensor_points[:,1], ls_sensor_points[:,2])
    ls_distance_to_plane = np.abs(ls_sensor_points[:,2] - (fit2[0] * ls_sensor_points[:,0] + fit2[1] * ls_sensor_points[:,1] + fit2[2]))
    print(f'Fitting deviation mean = {np.mean(ls_distance_to_plane):.4f} mm')
    print(f'Fitting deviation max = {np.max(ls_distance_to_plane):.4f} mm')
    print(f'Fitting deviation std = {np.std(ls_distance_to_plane):.4f} mm')

    print(f'\nFit {phase_points_transformed.shape[0]} points from PHASO to plane')
    fit2 = fit_to_plane(phase_points_transformed[:,0], phase_points_transformed[:,1], phase_points_transformed[:,2])
    phaso_distance_to_plane = np.abs(phase_points_transformed[:,2] - (fit2[0] * phase_points_transformed[:,0] + fit2[1] * phase_points_transformed[:,1] + fit2[2]))
    print(f'Fitting deviation mean = {np.mean(phaso_distance_to_plane):.4f} mm')
    print(f'Fitting deviation max = {np.max(phaso_distance_to_plane):.4f} mm')
    print(f'Fitting deviation std = {np.std(phaso_distance_to_plane):.4f} mm')

    print('\nTry to filter outliers with reprojection error...')
    filter_condition = (phase_errors[:,0] < 3*np.std(phase_errors[:,0])) & (phase_errors[:,1] < 3*np.std(phase_errors[:,1]))
    phase_points_transformed = phase_points_transformed[filter_condition,:]
    phase_errors = phase_errors[filter_condition,:]

    print(f'\nFit {phase_points_transformed.shape[0]} points to plane without outliers')
    fit2 = fit_to_plane(phase_points_transformed[:,0], phase_points_transformed[:,1], phase_points_transformed[:,2])
    phaso_distance_to_plane = np.abs(phase_points_transformed[:,2] - (fit2[0] * phase_points_transformed[:,0] + fit2[1] * phase_points_transformed[:,1] + fit2[2]))
    print(f'Fitting deviation mean = {np.mean(phaso_distance_to_plane):.4f} mm')
    print(f'Fitting deviation max = {np.max(phaso_distance_to_plane):.4f} mm')
    print(f'Fitting deviation std = {np.std(phaso_distance_to_plane):.4f} mm\n')
    
    print('\nTry to filter outliers with distance to fitted surface...')
    filter_condition = phaso_distance_to_plane < 3*np.std(phaso_distance_to_plane)
    phase_points_transformed = phase_points_transformed[filter_condition,:]
    phase_errors = phase_errors[filter_condition,:]

    print(f'\nFit {phase_points_transformed.shape[0]} points to plane without outliers')
    fit2 = fit_to_plane(phase_points_transformed[:,0], phase_points_transformed[:,1], phase_points_transformed[:,2])
    phaso_distance_to_plane = np.abs(phase_points_transformed[:,2] - (fit2[0] * phase_points_transformed[:,0] + fit2[1] * phase_points_transformed[:,1] + fit2[2]))
    print(f'Fitting deviation mean = {np.mean(phaso_distance_to_plane):.4f} mm')
    print(f'Fitting deviation max = {np.max(phaso_distance_to_plane):.4f} mm')
    print(f'Fitting deviation std = {np.std(phaso_distance_to_plane):.4f} mm\n')

    return ls_sensor_points, phase_points_transformed, ls_distance_to_plane, phaso_distance_to_plane, phase_errors


if __name__ == '__main__':

    PATH_TO_MEASUREMENT = r'experimental_results\2023-10-19'

    PATH_TO_PHASO_CALIBRATION = r'experimental_results\calibrated_data_phase4.json'

    phaso_calibration = load_json_file(PATH_TO_PHASO_CALIBRATION)
    
    path_to_date_folder = Path(PATH_TO_MEASUREMENT)

    measurements_paths = []

    for measurement_folder in path_to_date_folder.iterdir():
        if measurement_folder.is_dir():
            measurements_paths.append(measurement_folder)
            
    std_ls = []
    std_phaso = []
    differences = []

    for i, path in enumerate(measurements_paths):
        print(f'Обрабатываем поверхность {i+1} из {len(measurements_paths)}')
        try:
            ls_sensor_points, phase_points_transformed, ls_distance_to_plane, phaso_distance_to_plane, phase_errors = compare_granite(path, phaso_calibration)

            distance_errors = calculate_distance_difference(ls_sensor_points, phase_points_transformed)

            print(f'Distance deviation mean = {np.mean(distance_errors[:,2]):.4f} mm')
            print(f'Distance deviation max = {np.max(distance_errors[:,2]):.4f} mm')
            print(f'Distance deviation std = {np.std(distance_errors[:,2]):.4f} mm\n')

            std_ls.append(ls_distance_to_plane)
            std_phaso.append(phaso_distance_to_plane)
            differences.append(distance_errors)

            # draw_surface_by_contours(ls_sensor_points, phase_points_transformed, ls_distance_to_plane, phaso_distance_to_plane, phase_errors, distance_errors)
        except Exception as ex:
            print(ex)

    std_ls = np.array([np.std(el) for el in std_ls])
    std_phaso = np.array([np.std(el) for el in std_phaso])
    differences = [el[:,2] for el in differences]
    std_dif = np.array([np.std(el) for el in differences])

    # std_ls = 10**3 * np.array([0.04154948373051659, 0.04093847799902943, 0.04047180455900982, 0.0397561060186825, 0.03878952790256108, 0.03971447983898574, 0.039164994718468925, 0.03875863027479726, 0.03952797826345427, 0.03852526148755849])
    # std_phaso = 10**3 * np.array([0.04529797349631641, 0.046264603765972856, 0.04623559404760039, 0.04660295780556167, 0.04695086276563886, 0.04671086983830089, 0.04662471865351157, 0.04758088632103628, 0.047386267234966035, 0.0464112211122935])
    # differences = 10**3 * np.array([0.1185620736003126, 0.09953139560533236, 0.09401298212294475, 0.09989253274995802, 0.09314984074933819, 0.08293346773764969, 0.0908879814625155, 0.08104240046731885, 0.08342360315531197, 0.0908507353148522])

    draw_rms_distance_to_plane(std_ls * 10**3, std_phaso * 10**3)
