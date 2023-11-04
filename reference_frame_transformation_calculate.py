from pathlib import Path

import numpy as np
import pandas as pd

from common_functions import calculate_ICP, get_now_time, load_json_file


def calculate_transform_for_pair(phaso_path, stereo_path, calibration_data):
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
    condition = (reproj_errors1 < 3*stero_std_rprj1) & (reproj_errors2 < 3*stero_std_rprj2)
    stereo_points_2d1 = stereo_points_2d1[condition, :]
    stereo_points_2d2 = stereo_points_2d2[condition, :]
    stereo_points_3d = stereo_points_3d[condition, :]
    reproj_errors1 = reproj_errors1[condition]
    reproj_errors2 = reproj_errors2[condition]
    print(f'{stereo_points_3d.shape[0]} точек после фильтрации c STD {np.std(reproj_errors1): .3f} и {np.std(reproj_errors2): .3f}...')

    if len(stereo_points_3d) == 0:
        return None, None, stereo_points_3d.shape[0]
    
    if np.std(reproj_errors1) > 0.6 or np.std(reproj_errors2) > 0.6:
        return None, None, stereo_points_3d.shape[0]

    _, _, params, residuals = calculate_ICP(
        phase_points_transformed,
        stereo_points_3d,
        initial_params=INIT_PARAMS,
        max_overlap_distance=5000.0
    )

    return params, residuals, stereo_points_3d.shape[0]


def calculate_reference_frame_transform():
    path_to_phaso_date_folder = Path(PATH_TO_PHASO_MEASUREMENT)
    phaso_calibration = load_json_file(PATH_TO_PHASO_CALIBRATION)

    phaso_measurements_paths = []

    for measurement_folder in path_to_phaso_date_folder.iterdir():
        if measurement_folder.is_dir():
            phaso_measurements_paths.append(measurement_folder)

    path_to_stereo_data_folder = Path(PATH_TO_STEREO_MEASUREMENT)

    stereo_measurements_paths = list(path_to_stereo_data_folder.glob('*result_*.json'))

    loaded_data = []

    k = 0

    for i in range(len(phaso_measurements_paths)):
        stereo_path = stereo_measurements_paths[k]

        stereo_number = int(stereo_path.name.split('.')[0].split('_')[-1])
        if stereo_number != i:
            print(f'Не найден стерео результат #{i}, переходим к следующему...')
            continue

        phaso_path = phaso_measurements_paths[i]

        print(f'Обрабатываем поверхность {i+1} из {len(phaso_measurements_paths)}')
        
        try:
            params, residuals, points_num = calculate_transform_for_pair(phaso_path, stereo_path, phaso_calibration)

            if params is not None:
                loaded_data.append((
                    params.alpha1.estimated_value_scaled,
                    params.alpha2.estimated_value_scaled,
                    params.alpha3.estimated_value_scaled,
                    params.tx.estimated_value,
                    params.ty.estimated_value,
                    params.tz.estimated_value,
                    params.alpha1.estimated_uncertainty_scaled,
                    params.alpha2.estimated_uncertainty_scaled,
                    params.alpha3.estimated_uncertainty_scaled,
                    params.tx.estimated_uncertainty_scaled,
                    params.ty.estimated_uncertainty_scaled,
                    params.tz.estimated_uncertainty_scaled,
                    np.mean(residuals),
                    np.std(residuals),
                    points_num
                ))
        except Exception as ex:
            print(ex)
            raise ex
        
        k = k + 1

    df = pd.DataFrame(loaded_data, columns=(
        'alpha1',
        'alpha2',
        'alpha3',
        'tx',
        'ty',
        'tz',
        'alpha1_uncert',
        'alpha2_uncert',
        'alpha3_uncert',
        'tx_uncert',
        'ty_uncert',
        'tz_uncert',
        'mean_residuals',
        'std_residuals',
        'points_num'
    ))

    df.to_pickle(f'reference_calculation_results_{get_now_time()}.pickle')


if __name__ == '__main__':

    PATH_TO_STEREO_MEASUREMENT = r'experimental_results\stereo6'

    PATH_TO_PHASO_MEASUREMENT = r'experimental_results\2023-11-01'

    PATH_TO_PHASO_CALIBRATION = r'experimental_results\calibrated_data_phase5.json'

    INIT_PARAMS = (64.920109, -11.462937, -3.051579, 487.680680, 1253.083990, 469.350445)

    calculate_reference_frame_transform()
