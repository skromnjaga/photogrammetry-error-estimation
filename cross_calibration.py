import os
import sys
import json
import datetime
from typing import List, Dict

from functools import partial

import cv2
import numpy as np
from common_functions import get_now_time

IPCT_TEST_SETUP_PATH = r"D:\ipct_test_setup\IPCT_test_setup" 
#IPCT_TEST_SETUP_PATH = r"C:\Users\Caesar\source\repos\ipct_test_setup\IPCT_test_setup"
STRUCTURE_LIGHT_PYTHON_PATH = r"C:\Users\User\source\repos\helleb0re\structured-light-python"
#STRUCTURE_LIGHT_PYTHON_PATH = r"C:\Users\Caesar\source\repos\helleb0re\structured-light-python"

sys.path.append(IPCT_TEST_SETUP_PATH)

from ipcttestsetup import IPCTTestSetup, make_point_list

sys.path.append(STRUCTURE_LIGHT_PYTHON_PATH)

from camera import Camera
from main import initialize_cameras


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

    return ipct_setup, measure_points, phase_cameras


def detect_laser_spot(cameras: List[Camera], data: Dict, img1, img2):
    
    im1, im2 = get_images(cameras, wait_key_time=500)
    
    rc_data = {}

    all_imgs_valid  = True

    if im1 is not None and im2 is not None:
        images = [im1, im2]

        k = 0

        for im in images:
            
            if im is not None:

                image = np.array(im)

                #ret, thresh = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)        
                sx = 0
                sy = 0
                si = 0

                INTENSITY_THRESHOLD = 40
                PIXELS_THRESHOLD = 20
                AVG_INTENSITY_THRESHOLD = 100
                pixels_count = 0

                #find indexes of maximum element
                ymax, xmax = np.unravel_index(image.argmax(), image.shape)

                #calculate center of light spot in area of 50x50 near maximum
                for y in range(ymax - 25, ymax + 25):
                    for x in range(xmax - 25, xmax + 25):
                        if image[y, x] > INTENSITY_THRESHOLD:
                            sx = sx + x * image[y, x]
                            sy = sy + y * image[y, x]
                            si = si + image[y, x]
                            pixels_count = pixels_count + 1

                if pixels_count > 0:  
                    r = sx / si
                    c = sy / si
                    i_avg = si / pixels_count
                else:
                    r = c = i_avg = -1
                
                rc_data['r' + str(k + 1)] = r
                rc_data['c' + str(k + 1)] = c            

                spot_valid = pixels_count > PIXELS_THRESHOLD and i_avg > AVG_INTENSITY_THRESHOLD
                
                #if show_log:
                print(f'Camera #{k+1} -- r = {r:.3f}, c = {c:.3f}, pixs = {pixels_count},  i_max = {image[ymax, xmax]}, i_mean = {i_avg:.3f}')
                    
                if SHOW_DETECTED_SPOTS:
                    if spot_valid:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    if r != -1:
                        cv2.rectangle(image, (int(r) - 50, int(c) - 50), (int(r) + 50, int(c) + 50), color, 3)
                    cv2.putText(image, f'r = {r:.3f}, c = {c:.3f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

                    winname = 'image'

                    cv2.namedWindow(winname, cv2.WINDOW_NORMAL) 
                    cv2.resizeWindow(winname, 800, 600)
                    cv2.imshow(winname, image)
                    cv2.waitKey(500)

                k = k + 1
                all_imgs_valid = all_imgs_valid and spot_valid

        if all_imgs_valid:
            data.update(rc_data)


def measure_surface(setup: IPCTTestSetup, measure_points: List, ls_meas_folder: str, cameras: List[Camera]) -> Dict:
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

    callback_funct = partial(detect_laser_spot, cameras)

    # Измеряем поверхность в заданных точках
    try:
        measured_points = setup.measure_points(
            measure_points,
            use_calibration=False,
            use_cameras=False,
            use_servos=True,
            image_path=None,
            time_before_measure=1,
            callback=callback_funct
        )
    except Exception as ex:
        setup.home_sensor()
        raise ex

    data["measuring_time_end"] = get_now_time()

    data.update(measured_points)

    with open(os.path.join(ls_meas_folder, f'{RESULTS_NAME}_{get_now_time()}.json'), 'w') as outfile:
        print("Cохранение файла с результатами измерений лазерного датчика...")
        json.dump(data, outfile, indent=4)

    return data


def get_images(cameras: List[Camera], wait_key_time: int=0, close_windows =False) -> List[np.ndarray]:
    '''
    Получает изображения с указанных видеокамер 
    '''
    cameras_num = len(cameras)

    for i in range(cameras_num):
        cv2.namedWindow(f'cam{i}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'cam{i}', 800, 600)

    imgs = []

    for i in range(cameras_num):
        img = cameras[i].get_image()
        imgs.append(img)
        cv2.imshow(f'cam{i}', img)

    cv2.waitKey(wait_key_time)

    if close_windows:
        for i in range(cameras_num):
            cv2.destroyWindow(f'cam{i}')

    return imgs


def main_measurement_loop(
        ipct_setup: IPCTTestSetup,
        measure_points: List, 
        phase_cameras: List[Camera]
        ):
    '''
    Основное тело цикла для проведения измерений
    '''
    # Создаем папки для хранения результатов измерений
    measurement_day_folder = os.path.join(RESULTS_PATH, get_now_time('%Y-%m-%d'))

    measurement_folder = os.path.join(measurement_day_folder, get_now_time())
    os.makedirs(measurement_folder, exist_ok=True)

    print(f'Начинаем кросскалибровку...')
    measure_surface(ipct_setup, measure_points, measurement_folder, phase_cameras)

    

if __name__ == "__main__":
    # Папка для сохранения результатов
    RESULTS_PATH = "experemental_results"

    # Тип результата измерения для лазерного датчика
    RESULTS_NAME = "cross_calibration_measurement"

    # Формат сохранения изображений
    IMAGE_FORMAT = ".png"

    # Имена портов для подключения к сервомашинам и лазерному датчику
    SERVO_PORT = "COM3"
    SENSOR_PORT = "COM4"

    # Тип камер стереосистемы и фазограмметрической системы
    STEREO_CAMS_TYPE = 'baumer'
    PHASE_CAMS_TYPE = 'baumer'

    # Серийные номера камер стереосистемы и фазограмметрической системы
    STEREO_CAMS_SERIAL_NUMBERS = ['700005466452', '700005466457']
    PHASE_CAMS_SERIAL_NUMBERS = ['700007464735', '700007464734']

    PHASE_CAMS_EXPOSURES = [5000, 5000] # us
    PHASE_CAMS_GAINS = [1.0, 1.0]

    # Параметры для расчета точек в которых будут производится измерения лазерным датчиком
    START_X = 230
    START_Y = 14
    START_S = 230
    END_X = 330
    END_Y = 374
    END_S = 800

    STEPS_X = 10
    STEPS_Y = 10
    STEPS_S = 10

    # Параметры для формирования случайной поверхности
    MAXIMUM_DIFFERENCE_POSITION = 100
    MINIMUM_POSITION = 230 #390
    MAXIMUM_POSITION = 800

    SHOW_DETECTED_SPOTS = True

        # Инициализируем устройства установки
    setup = init_experemental_setup()
        
    main_measurement_loop(*setup)
