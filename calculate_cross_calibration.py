import sys
import json

import numpy as np
from matplotlib import pyplot as plt

from config import STRUCTURE_LIGHT_PYTHON_PATH

sys.path.append(STRUCTURE_LIGHT_PYTHON_PATH)

from processing import triangulate_points

from common_functions import load_json_file


def rigid_transform_3D(A: np.ndarray, B: np.ndarray):

    A = np.transpose(A, (1, 0))
    B = np.transpose(B, (1, 0))

    num_rows1, num_cols1 = A.shape
    num_rows2, num_cols2 = B.shape
   
     # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R @ centroid_A + centroid_B

    return R, t


if __name__ == '__main__':

    PATH_TO_CAMERA_CALIBRATION = r'experimental_results\calibrated_data_phase4.json'

    PATH_TO_CROSS_CALIBRATION = r'experimental_results\2023-10-19\2023-10-19_15-29-09\cross_calibration_measurement_2023-10-19_17-29-21.json'

    cams_calibration = load_json_file(PATH_TO_CAMERA_CALIBRATION)

    cross_calibration = load_json_file(PATH_TO_CROSS_CALIBRATION)

    # Get X, Y, Z data from file
    ls_sensor_points = np.array([[point['x'], point['y'], point['z']] for point in cross_calibration['measured_points'] if 'r1' in point])

    # Get r1, c1, r2, c2 data from file
    cam1_2d_points = np.array([[point['r1'], point['c1']] for point in cross_calibration['measured_points'] if 'r1' in point])
    cam2_2d_points = np.array([[point['r2'], point['c2']] for point in cross_calibration['measured_points'] if 'r1' in point])

    k = 1
    error = 1000
    ERROR_THRESHOLD = 0.3
    MAX_ITER_NUMBER = 20

    while error > ERROR_THRESHOLD:

        if k > MAX_ITER_NUMBER:
            break

        if k > 1:
            indices = (reproj_err1 < 3*np.std(reproj_err1)) & (reproj_err2 < 3*np.std(reproj_err2))
            cam1_2d_points = cam1_2d_points[indices]
            cam2_2d_points = cam2_2d_points[indices]
            ls_sensor_points = ls_sensor_points[indices]

        cams_triang_points, rms1, rms2, reproj_err1, reproj_err2 = triangulate_points(cams_calibration, cam1_2d_points, cam2_2d_points)
        print(f'Reprojected RMSs {rms1:.3f} - {rms2:.3f}')
        print(f'Итерация #{k} - Количество точек {len(cam1_2d_points)} - Средняя ошибка {np.mean(rms1)} - Максимальная ошибка {np.max(reproj_err1)} - Стд. отклонение {np.std(reproj_err1)}')
        print(f'Итерация #{k} - Количество точек {len(cam2_2d_points)} - Средняя ошибка {np.mean(rms2)} - Максимальная ошибка {np.max(reproj_err2)} - Стд. отклонение {np.std(reproj_err2)}')

        k = k + 1
        error = np.max((rms1, rms2))

    R, t = rigid_transform_3D(cams_triang_points, ls_sensor_points)

    # # Create point cloud objects
    # X_mov = pd.DataFrame({
    #     'x': phase_points_3d[:,0],
    #     'y': phase_points_3d[:,1],
    #     'z': phase_points_3d[:,2],
    # })

    # X_fix = pd.DataFrame({
    #     'x': ls_sensor_points[:,0],
    #     'y': ls_sensor_points[:,1],
    #     'z': ls_sensor_points[:,2],
    # })

    # pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
    # pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

    # # Create simpleICP object, add point clouds, and run algorithm!
    # icp = SimpleICP()
    # icp.add_point_clouds(pc_fix, pc_mov)
    # H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1000)
    # R = H[:3,:3]
    # t = H[:3,3].reshape((3,1))

    H = np.hstack((R, t))

    cams_triang_transformed_points = H.dot(np.hstack((cams_triang_points, np.ones((cams_triang_points.shape[0], 1)))).T)

    # k = 1
    # error = 1000
    # ERROR_THRESHOLD = 0.5
    # MAX_ITER_NUMBER = 20

    # while error > ERROR_THRESHOLD:

    #     if k > MAX_ITER_NUMBER:
    #         break

    #     if k > 1:
    #         cams_triang_points = cams_triang_points[err < 3*np.std(err)]
    #         ls_sensor_points = ls_sensor_points[err < 3*np.std(err)]

    #     R, t = rigid_transform_3D(cams_triang_points, ls_sensor_points)
        
    #     cams_triang_transformed_points = np.hstack((R, t)).dot(np.hstack((cams_triang_points, np.ones((cams_triang_points.shape[0], 1)))).T)

    #     err = np.sum((cams_triang_transformed_points - ls_sensor_points.T) ** 2, axis=0) ** 0.5

    #     print(f'Итерация #{k} - Количество точек {len(cams_triang_points)} - Средняя ошибка {np.mean(err)} - Максимальная ошибка {np.max(err)} - Стд. отклонение {np.std(err)}')
        
    #     k = k + 1
    #     error = np.std(err)

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ls_sensor_points[:,0], ls_sensor_points[:,1], ls_sensor_points[:,2])
    ax.scatter(cams_triang_transformed_points[0,:], cams_triang_transformed_points[1,:], cams_triang_transformed_points[2,:])

    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    ax.set_zlabel('Z, mm')

    ax.view_init(elev=-75, azim=-89)

    plt.show()

    cams_calibration['RWorld'] = R.tolist()
    cams_calibration['TWorld'] = t.tolist()

    with open(PATH_TO_CAMERA_CALIBRATION, 'w') as outfile:
        print("Cохранение результатов кросскалибровки...")
        json.dump(cams_calibration, outfile, indent=4)