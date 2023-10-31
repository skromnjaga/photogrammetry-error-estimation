import os
import sys
import json
import time
from typing import List, Dict

import cv2

from common_functions import get_now_time
from config2 import IPCT_TEST_SETUP_PATH, STRUCTURE_LIGHT_PYTHON_PATH, RESULTS_PATH, SERVO_PORT, SENSOR_PORT, PHASE_CAMS_TYPE, PHASE_CAMS_SERIAL_NUMBERS, STEREO_CAMS_TYPE, STEREO_CAMS_SERIAL_NUMBERS

sys.path.append(IPCT_TEST_SETUP_PATH)

from ipcttestsetup import IPCTTestSetup, make_point_list

sys.path.append(STRUCTURE_LIGHT_PYTHON_PATH)

import config as phase_meas_system_config
from camera import Camera
from projector import Projector
from main import initialize_cameras, capture_measurement_images
from fpp_structures import PhaseShiftingAlgorithm


def init_experemental_setup():
    '''
    Инициализируем оборудование для проведения измерений
    '''
    print('Инициализация сервомашин и лазерного датчика...')
    ipct_setup = IPCTTestSetup(SERVO_PORT, SENSOR_PORT)

    ipct_setup.set_position(512)

    print('Рассчитываем координаты точек для измерения лазерным датчиком...')
    measure_points = make_point_list(
        START_X,
        END_X,
        STEPS_X,
        START_Y,
        END_Y,
        STEPS_Y,
        START_S,
        END_S,
        STEPS_S,
        rnd=False,
    )

    print('Инициализация проектора и камер для фазограмметрии...')
    projector = Projector(
        phase_meas_system_config.PROJECTOR_WIDTH,
        phase_meas_system_config.PROJECTOR_HEIGHT,
        phase_meas_system_config.PROJECTOR_MIN_BRIGHTNESS,
        phase_meas_system_config.PROJECTOR_MAX_BRIGHTNESS,
    )

    try:
        phase_cameras = initialize_cameras(
            PHASE_CAMS_TYPE, 
            cam_to_found_number=2, 
            cameras_serial_numbers=PHASE_CAMS_SERIAL_NUMBERS
        )            
    except Exception as ex:
        print(f'Ошибка поиска камер для фазограмметрии: {ex}')
        exit(1)

    for i in range(len(PHASE_CAMS_EXPOSURES)):
        phase_cameras[i].exposure = PHASE_CAMS_EXPOSURES[i]

    for i in range(len(PHASE_CAMS_GAINS)):
        phase_cameras[i].gain = PHASE_CAMS_GAINS[i]

    print('Инициализация камер стереосистемы...')
    try:
        stereo_cameras = initialize_cameras(
            STEREO_CAMS_TYPE,
            cam_to_found_number=len(STEREO_CAMS_SERIAL_NUMBERS),
            cameras_serial_numbers=STEREO_CAMS_SERIAL_NUMBERS
        ) 
    except Exception as ex:
        print(f'Ошибка поиска камер для стереосистемы: {ex}')
        exit(1)

    for i in range(len(STEREO_CAMS_SERIAL_NUMBERS)):
        stereo_cameras[i].exposure = STEREO_CAMS_EXPOSURES[i]

    for i in range(len(STEREO_CAMS_SERIAL_NUMBERS)):
        stereo_cameras[i].gain = STEREO_CAMS_GAINS[i]

    # Включаем сохранение результатов в файл для фазограмметрической системы
    phase_meas_system_config.SAVE_MEASUREMENT_IMAGE_FILES = True

    return ipct_setup, measure_points, projector, phase_cameras, stereo_cameras


def measure_surface(setup: IPCTTestSetup, measure_points: List, ls_meas_folder: str) -> Dict:
    '''
    Проводит измерение лазерным датчиком расстояния по заданным
    точкам и сохраняет реузльтат в указанную папку
    '''
    # Создаем словарь для хранения результатов измерений
    data = {}
    data["type"] = RESULTS_NAME
    data["start_x"] = START_X
    data["start_y"] = START_Y
    data["end_x"] = END_X
    data["end_y"] = END_Y
    data["steps_x"] = STEPS_X
    data["steps_y"] = STEPS_Y

    data["servo_positions"] = setup.servos_positions

    data["measuring_time_start"] = get_now_time()

    # Измеряем поверхность в заданных точках
    try:
        measured_points = setup.measure_points(
            measure_points,
            use_calibration=False,
            use_cameras=False,
            use_servos=False,
            image_path=None,
            time_before_measure=1,
        )
    except:
        setup.home_sensor()
        exit()

    data["measuring_time_end"] = get_now_time()

    data.update(measured_points)

    with open(os.path.join(ls_meas_folder, f'{RESULTS_NAME}_{get_now_time()}.json'), 'w') as outfile:
        print("Cохранение файла с результатами измерений лазерного датчика...")
        json.dump(data, outfile, indent=4)

    return data


def get_stereo_measurement(cameras: List[Camera], stereo_images_folder: str):
    '''
    Получает и сохраняет изображения с видеокамер стереосистемы в указанную папку
    '''
    if len(cameras) > 0:
        cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cam1', 800, 600)
        cv2.resizeWindow('cam2', 800, 600)

        img1 = cameras[0].get_image()
        img2 = cameras[1].get_image()

        cv2.imshow('cam1', img1)
        cv2.imshow('cam2', img2)

        cv2.waitKey(1000)

        cv2.imwrite(os.path.join(stereo_images_folder, f'img1{IMAGE_FORMAT}'), img1)
        cv2.imwrite(os.path.join(stereo_images_folder, f'img2{IMAGE_FORMAT}'), img2)

        cv2.destroyWindow('cam1')
        cv2.destroyWindow('cam2')


def get_phasogrammetry_measurement(projector: Projector, cameras: List[Camera], phase_result_folder: str):
    '''
    Выполняет фазограмметрическое измерение и сохраняет результат в указанную папку  
    '''
    # Указываем через конфиг фазограмметрической системы папку для сохранения результата
    phase_meas_system_config.DATA_PATH = phase_result_folder

    capture_measurement_images(cameras, projector, phase_shift_type=PhaseShiftingAlgorithm.n_step)


def main_measurement_loop(
        ipct_setup: IPCTTestSetup,
        measure_points: List, 
        projector: Projector, 
        phase_cameras: List[Camera],
        stereo_cameras: List[Camera]):
    '''
    Основное тело цикла для проведения измерений
    '''
    # Создаем папки для хранения результатов измерений
    measurement_day_folder = os.path.join(RESULTS_PATH, get_now_time('%Y-%m-%d'))

    # Проводим заданное количество измерений
    for meas_num in range(MEASUREMENT_REPEAT_COUNT):
        print(f'Начинаем измерение #{meas_num}...')

        measurement_folder = os.path.join(measurement_day_folder, get_now_time())
        os.makedirs(measurement_folder, exist_ok=True)

        if meas_num > 0 or not FIRST_MEAS_WITHOUT_SERVO_MOVEMENT:
            print(f'Устанавливаем новое положение поверхности на имитаторе...')
            ipct_setup.set_not_so_random_position(
                MAXIMUM_DIFFERENCE_POSITION, 
                min=MINIMUM_POSITION, 
                max=MAXIMUM_POSITION
            )

        projector.project_white_background()

        time.sleep(5)

        print(f'Получаем изображения с камер для IPCT и DIC...')
        get_stereo_measurement(stereo_cameras, measurement_folder)
        
        time.sleep(5)

        print(f'Получаем изображения для фазограмметрии...')
        get_phasogrammetry_measurement(projector, phase_cameras, measurement_folder)

        projector.project_black_background()

        time.sleep(5)

        print(f'Измеряем поверхность лазерным датчиком...')
        measure_surface(ipct_setup, measure_points, measurement_folder)

        time.sleep(5)

    

if __name__ == "__main__":
    # Тип результата измерения для лазерного датчика
    RESULTS_NAME = "ls_measurement"

    # Формат сохранения изображений
    IMAGE_FORMAT = ".png"

    STEREO_CAMS_EXPOSURES = [100_000, 100_000] # us
    STEREO_CAMS_GAINS = [1.3, 1.3]

    PHASE_CAMS_EXPOSURES = [100_000, 100_000] # us
    PHASE_CAMS_GAINS = [2.0, 2.0]

    # Параметры для расчета точек в которых будут производится измерения лазерным датчиком
    # Для имитатора
    # START_X = 87
    # START_Y = 17
    # START_S = 0
    # END_X = 452
    # END_Y = 385
    # END_S = 1

    # Для поверочной плиты
    START_X = 58
    START_Y = 17
    START_S = 0
    END_X = 450
    END_Y = 380
    END_S = 1

    STEPS_X = 12
    STEPS_Y = 12
    STEPS_S = 0

    # Параметры для формирования случайной поверхности
    MAXIMUM_DIFFERENCE_POSITION = 100
    MINIMUM_POSITION = 230 #390
    MAXIMUM_POSITION = 800

    # Количество повторений измерений
    MEASUREMENT_REPEAT_COUNT = 10

    # Первое измерение на плоской поверхности
    FIRST_MEAS_WITHOUT_SERVO_MOVEMENT = False

    # Инициализируем устройства установки
    setup = init_experemental_setup()
        
    main_measurement_loop(*setup)
