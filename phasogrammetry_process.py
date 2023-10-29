import sys
import json
import pathlib

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

from common_functions import get_measurements_paths, get_now_time, load_json_file
from config import STRUCTURE_LIGHT_PYTHON_PATH, RESULTS_PATH

sys.path.append(STRUCTURE_LIGHT_PYTHON_PATH)

from fpp_structures import FPPMeasurement, PhaseShiftingAlgorithm
from processing import calculate_phase_for_fppmeasurement, create_polygon, process_fppmeasurement_with_phasogrammetry, get_phase_field_ROI, get_phase_field_LUT, triangulate_points
from utils import get_images_from_config, load_fpp_measurements


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


def process_with_phasogrammetry(
        measurement_folder: pathlib.Path = None, 
        measurement: FPPMeasurement = None,
        show_results: bool = False
    ):
    '''
    Calculate phasogrammetry measurement
    '''
    if measurement is None:
        # Load FPPMeasurements from files
        print('Load FPPMeasurements from files...', end='', flush=True)
        for folder in measurement_folder.iterdir():
            if folder.is_dir():
                inner_measurement_folder = folder
                break
        try:
            measurement = load_fpp_measurements(inner_measurement_folder.joinpath('fpp_measurement.json'))
        except Exception as ex:
            print(f'Ошибка загрузки файла: {ex}')
            return
        print('Done')

    # If there are no images in measurement - load them
    if len(measurement.camera_results[0].imgs_list) == 0:
        print('Load images from files...', end='', flush=True)
        for cam_result in measurement.camera_results:
            cam_result.imgs_list = get_images_from_config(cam_result.imgs_file_names)
        print('Done')
    
    # Display FPPMeasurement parameters
    if measurement.phase_shifting_type == PhaseShiftingAlgorithm.n_step:
        algortihm_type = f'{len(measurement.shifts)}-step' 
    elif measurement.phase_shifting_type == PhaseShiftingAlgorithm.double_three_step:
        algortihm_type = 'double 3-step'
    print(f'\nPhase shift algorithm: {algortihm_type}')
    print(f'Phase shifts: {measurement.shifts}')
    print(f'Frequencies: {measurement.frequencies}\n')

    # Load calibration data for cameras stero system
    print('Load calibration data for cameras stereo system...', end='', flush=True)
    calibration_data = load_json_file(PATH_TO_CALIBRATION)
    print('Done')

    # Calculate phase fields
    print('Calculate phase fields for first and second cameras...', end='', flush=True)
    calculate_phase_for_fppmeasurement(measurement)
    print('Done')

    if show_results:
        plt.rcParams.update({'font.size': 16})

        # Plot unwrapped phases
        ax = plt.subplot(111)
        cs = plt.imshow(measurement.camera_results[0].unwrapped_phases[-1], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixels')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Phase, radians')
        plt.show()
        ax = plt.subplot(111)
        cs = plt.imshow(measurement.camera_results[1].unwrapped_phases[-1], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixels')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Phase, radians')
        plt.show()
        ax = plt.subplot(111)
        cs = plt.imshow(measurement.camera_results[2].unwrapped_phases[-1], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixels')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Phase, radians')
        plt.show()
        ax = plt.subplot(111)
        cs = plt.imshow(measurement.camera_results[3].unwrapped_phases[-1], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixels')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Phase, radians')
        plt.show()

    print('Determine phase fields ROI...', end='', flush=True)
    get_phase_field_ROI(measurement)
    print('Done')

    # Set ROI manually for test plate
    measurement.camera_results[0].ROI = ROI
    ROI_mask = create_polygon(measurement.camera_results[0].imgs_list[0][0].shape, measurement.camera_results[0].ROI)
    measurement.camera_results[0].ROI_mask = ROI_mask
    measurement.camera_results[2].ROI_mask = ROI_mask

    if show_results:
        # Plot signal to noise ration
        plt.subplot(221)
        cs = plt.imshow(measurement.camera_results[0].modulated_intensities[-2]/measurement.camera_results[0].average_intensities[-2], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixles')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Modulation coefficient, relative')
        # # Draw ROI
        # ax = plt.plot(measurement.camera_results[0].ROI[:, 0], measurement.camera_results[0].ROI[:, 1], 'r-')
        # ax = plt.plot([measurement.camera_results[0].ROI[-1, 0], measurement.camera_results[0].ROI[0, 0]],
        #               [measurement.camera_results[0].ROI[-1, 1], measurement.camera_results[0].ROI[0, 1]], 'r-')
        plt.subplot(222)
        cs = plt.imshow(measurement.camera_results[1].modulated_intensities[-1]/measurement.camera_results[1].average_intensities[-1], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixels')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Modulation coefficient, relative')
        plt.subplot(223)
        cs = plt.imshow(measurement.camera_results[2].modulated_intensities[-1]/measurement.camera_results[2].average_intensities[-1], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixels')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Modulation coefficient, relative')
        plt.subplot(224)
        cs = plt.imshow(measurement.camera_results[3].modulated_intensities[-1]/measurement.camera_results[3].average_intensities[-1], cmap='gray')
        ax.set_xlabel('X, pixels')
        ax.set_ylabel('Y, pixels')
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Modulation coefficient, relative')
        plt.show()

    print('Calculate phase fields LUT...', end='', flush=True)
    LUT = get_phase_field_LUT(measurement)
    print('Done')

    # Process FPPMeasurements with phasogrammetry approach
    print('Calculate 2D corresponding points with phasogrammetry approach...')
    points_2d_1, points_2d_2, errs = process_fppmeasurement_with_phasogrammetry(measurement, PHASE_PROCESS_STEP_X, PHASE_PROCESS_STEP_Y, LUT)
    print(f'Found {points_2d_1.shape[0]} corresponding points')
    print('Done')

    print('\nCalculate 3D points with triangulation...')
    points_3d, rms1, rms2, reproj_err1, reproj_err2 = triangulate_points(calibration_data, points_2d_1, points_2d_2)
    print(f'Reprojected RMS for camera 1 = {rms1:.3f}')
    print(f'Reprojected RMS for camera 2 = {rms2:.3f}')
    print('Done')

    # Filter outliers by reprojection error
    print('\nTry to filter outliers with reprojection error threshold...')
    filter_condition = (reproj_err1 < REPROJECTION_ERROR) & (reproj_err2 < REPROJECTION_ERROR)
    x = points_3d[filter_condition, 0].astype(float)
    y = points_3d[filter_condition, 1].astype(float)
    z = points_3d[filter_condition, 2].astype(float)
    points_2d_1 = points_2d_1[filter_condition,:]
    points_2d_2 = points_2d_2[filter_condition,:]
    errs = errs[filter_condition,:]
    print(f'Found {points_3d.shape[0] - x.shape[0]} outliers')

    print('\nCalculate 3D points after filtration...')
    points_3d, rms1, rms2, reproj_err1, reproj_err2 = triangulate_points(calibration_data, points_2d_1, points_2d_2)
    print(f'Reprojected RMS for camera 1 = {rms1:.3f}')
    print(f'Reprojected RMS for camera 2 = {rms2:.3f}')
    print('Done')

    # Plot 3D point cloud
    if show_results:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z)

        plt.gca().invert_yaxis()

        ax.set_xlabel('X, mm')
        ax.set_ylabel('Y, mm')
        ax.set_zlabel('Z, mm')

        ax.view_init(elev=-75, azim=-89)

        plt.show()

    store_path = measurement_folder.joinpath(f'{RESULT_NAME}.json')

    # Создаем словарь для хранения результатов измерений
    data = {}
    data['type'] = RESULT_NAME
    data['phase_shifting_type'] = algortihm_type
    data['frequencies'] = measurement.frequencies
    data['shifts'] = measurement.shifts
    data['prcoessing_time'] = get_now_time()

    points = []

    for rc1, rc2, xx, yy, zz, re1, re2, err in zip(points_2d_1, points_2d_2, x, y, z, reproj_err1, reproj_err2, errs):
        points.append({
            'r1': float(rc1[0]),
            'c1': float(rc1[1]),
            'r2': float(rc2[0]),
            'c2': float(rc2[1]),
            'x': xx, 
            'y': yy, 
            'z': zz, 
            'reproj_err1': float(re1), 
            'reproj_err2': float(re2), 
            'phase_err1': float(err[0]), 
            'phase_err2': float(err[1])
        })
    
    data['measured_points'] = points

    with open(store_path, 'w') as outfile:
        print("Cохранение файла с результатами измерений лазерного датчика...")
        json.dump(data, outfile, indent=4)


if __name__ == '__main__':

    PATH_TO_CALIBRATION = r'experemental_results/calibrated_data_phase4.json'

    RESULT_NAME = 'phasogrammetry_measurement'

    SKIP_FOLDERS_WITH_RESULTS = True
    SHOW_RESULT = True

    PHASE_PROCESS_STEP_X = 3
    PHASE_PROCESS_STEP_Y = 3

    # ROI = np.array([[390, 90], [1420, 270], [1580, 1380], [360, 1415]])
    ROI = np.array([[462, 88], [1487, 216], [1690, 1323], [490, 1395]])
    #ROI = np.array([[380, 980], [960, 1000], [1000, 1400], [360, 1415]])

    REPROJECTION_ERROR = 1.0 # pixel

    measurements_paths = get_measurements_paths(RESULTS_PATH, SKIP_FOLDERS_WITH_RESULTS, RESULT_NAME)

    print(f'Найдено {len(measurements_paths)} папок с измерениями...')

    for i, path in enumerate(measurements_paths):
        print(f'\nОбработка папки "{path.parts[-1]}" # {i+1} из {len(measurements_paths)}...')
        process_with_phasogrammetry(path, show_results=SHOW_RESULT)