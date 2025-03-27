from typing import Optional
import cv2
import numpy as np
import os
import dlib
import scipy.io as scio
from tools.frame_preprocessors import FramePreprocessor


class ThreeD_Model:
    def __init__(self, path, name):
        self.load_model(path, name)

    def load_model(self, path, name):
        model = scio.loadmat(path)[name]
        self.out_A = np.asmatrix(model["outA"][0, 0], dtype="float32")  # 3x3
        self.size_U = model["sizeU"][0, 0][0]  # 1x2
        self.model_TD = np.asarray(model["threedee"][0, 0], dtype="float32")  # 68x3
        self.indbad = model["indbad"][0, 0]  # 0x1
        self.ref_U = np.asarray(model["refU"][0, 0])


def shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append(
            (
                shape.part(i).x,
                shape.part(i).y,
            )
        )
    xy = np.asarray(xy, dtype="float32")
    return xy


def frontalize(do_calculate_symmetry, img, proj_matrix, ref_U, eyemask):
    ACC_CONST = 800
    img = img.astype("float32")

    bgind = np.sum(np.abs(ref_U), 2) == 0
    # count the number of times each pixel in the query is accessed
    threedee = np.reshape(ref_U, (-1, 3), order="F").transpose()
    temp_proj = proj_matrix * np.vstack((threedee, np.ones((1, threedee.shape[1]))))
    temp_proj2 = np.divide(temp_proj[0:2, :], np.tile(temp_proj[2, :], (2, 1)))

    bad = np.logical_or(temp_proj2.min(axis=0) < 1, temp_proj2[1, :] > img.shape[0])
    bad = np.logical_or(bad, temp_proj2[0, :] > img.shape[1])
    bad = np.logical_or(bad, bgind.reshape((-1), order="F"))
    bad = np.asarray(bad).reshape((-1), order="F")

    temp_proj2 -= 1

    badind = np.nonzero(bad > 0)[0]
    temp_proj2[:, badind] = 0

    ind = np.ravel_multi_index(
        (
            np.asarray(temp_proj2[1, :].round(), dtype="int64"),
            np.asarray(temp_proj2[0, :].round(), dtype="int64"),
        ),
        dims=img.shape[:-1],
        order="F",
    )

    synth_frontal_acc = np.zeros(ref_U.shape[:-1])
    ind_frontal = np.arange(0, ref_U.shape[0] * ref_U.shape[1])

    c, ic = np.unique(ind, return_inverse=True)
    bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
    count, bin_edges = np.histogram(ind, bin_edges)
    synth_frontal_acc = synth_frontal_acc.reshape(-1, order="F")
    synth_frontal_acc[ind_frontal] = count[ic]
    synth_frontal_acc = synth_frontal_acc.reshape((320, 320), order="F")
    synth_frontal_acc[bgind] = 0
    synth_frontal_acc = cv2.GaussianBlur(
        synth_frontal_acc, (15, 15), 30.0, borderType=cv2.BORDER_REPLICATE
    )

    # remap
    mapX = temp_proj2[0, :].astype(np.float32)
    mapY = temp_proj2[1, :].astype(np.float32)

    mapX = np.reshape(mapX, (-1, 320), order="F")
    mapY = np.reshape(mapY, (-1, 320), order="F")

    frontal_raw = cv2.remap(img, mapX, mapY, cv2.INTER_CUBIC)

    frontal_raw = frontal_raw.reshape((-1, 3), order="F")
    frontal_raw[badind, :] = 0
    frontal_raw = frontal_raw.reshape((320, 320, 3), order="F")

    if not do_calculate_symmetry:
        return frontal_raw.copy(order="C").astype(np.uint8)

    # which side has more occlusions?
    midcolumn = int(np.round(ref_U.shape[1] / 2))
    sumaccs = synth_frontal_acc.sum(axis=0)
    sum_left = sumaccs[0:midcolumn].sum()
    sum_right = sumaccs[midcolumn + 1:].sum()
    sum_diff = sum_left - sum_right

    if np.abs(sum_diff) > ACC_CONST:  # one side is ocluded
        ones = np.ones((ref_U.shape[0], midcolumn))
        zeros = np.zeros((ref_U.shape[0], midcolumn))
        if sum_diff > ACC_CONST:  # left side of face has more occlusions
            weights = np.hstack((zeros, ones))
        else:  # right side of face has more occlusions
            weights = np.hstack((ones, zeros))
        weights = cv2.GaussianBlur(
            weights, (33, 33), 60.5, borderType=cv2.BORDER_REPLICATE
        )

        # apply soft symmetry to use whatever parts are visible in ocluded side
        synth_frontal_acc /= synth_frontal_acc.max()
        weight_take_from_org = 1.0 / np.exp(0.5 + synth_frontal_acc)
        weight_take_from_sym = 1 - weight_take_from_org

        weight_take_from_org = np.multiply(weight_take_from_org, np.fliplr(weights))
        weight_take_from_sym = np.multiply(weight_take_from_sym, np.fliplr(weights))

        weight_take_from_org = np.tile(
            weight_take_from_org.reshape(320, 320, 1), (1, 1, 3)
        )
        weight_take_from_sym = np.tile(
            weight_take_from_sym.reshape(320, 320, 1), (1, 1, 3)
        )
        weights = np.tile(weights.reshape(320, 320, 1), (1, 1, 3))

        denominator = weights + weight_take_from_org + weight_take_from_sym
        frontal_sym = (
            np.multiply(frontal_raw, weights)
            + np.multiply(frontal_raw, weight_take_from_org)
            + np.multiply(np.fliplr(frontal_raw), weight_take_from_sym)
        )
        frontal_sym = np.divide(frontal_sym, denominator)

        # exclude eyes from symmetry
        frontal_sym = np.multiply(frontal_sym, 1 - eyemask) + np.multiply(
            frontal_raw, eyemask
        )
        frontal_raw[frontal_raw > 255] = 255
        frontal_raw[frontal_raw < 0] = 0
        frontal_raw = frontal_raw.astype(np.uint8)
        frontal_sym[frontal_sym > 255] = 255
        frontal_sym[frontal_sym < 0] = 0
        frontal_sym = frontal_sym.astype(np.uint8)
    else:  # both sides are occluded pretty much to the same extent -- do not use symmetry
        frontal_sym = frontal_raw.astype(np.uint8)

    return frontal_sym.astype(np.uint8)


def estimate_camera(model3D, fidu_XY):
    rmat, tvec = calib_camera(model3D, fidu_XY)
    RT = np.hstack((rmat, tvec))
    projection_matrix = model3D.out_A * RT
    return projection_matrix, model3D.out_A, rmat, tvec


def calib_camera(model3D, fidu_XY):
    # compute pose using refrence 3D points + query 2D points
    ret, rvecs, tvec = cv2.solvePnP(
        model3D.model_TD, fidu_XY, model3D.out_A, None, None, None, False
    )
    rmat, jacobian = cv2.Rodrigues(rvecs, None)

    inside = calc_inside(
        model3D.out_A,
        rmat,
        tvec,
        model3D.size_U[0],
        model3D.size_U[1],
        model3D.model_TD,
    )
    if inside == 0:
        tvec = -tvec
        t = np.pi
        RRz180 = np.asmatrix(
            [np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]
        ).reshape((3, 3))
        rmat = RRz180 * rmat
    return rmat, tvec


def get_opengl_matrices(camera_matrix, rmat, tvec, width, height):
    projection_matrix = np.asmatrix(np.zeros((4, 4)))
    near_plane = 0.0001
    far_plane = 10000

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    px = camera_matrix[0, 2]
    py = camera_matrix[1, 2]

    projection_matrix[0, 0] = 2.0 * fx / width
    projection_matrix[1, 1] = 2.0 * fy / height
    projection_matrix[0, 2] = 2.0 * (px / width) - 1.0
    projection_matrix[1, 2] = 2.0 * (py / height) - 1.0
    projection_matrix[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    projection_matrix[3, 2] = -1
    projection_matrix[2, 3] = -2.0 * far_plane * near_plane / (far_plane - near_plane)

    deg = 180
    t = deg * np.pi / 180.0
    RRz = np.asmatrix(
        [np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]
    ).reshape((3, 3))
    RRy = np.asmatrix(
        [np.cos(t), 0, np.sin(t), 0, 1, 0, -np.sin(t), 0, np.cos(t)]
    ).reshape((3, 3))
    rmat = RRz * RRy * rmat

    mv = np.asmatrix(np.zeros((4, 4)))
    mv[0:3, 0:3] = rmat
    mv[0, 3] = tvec[0]
    mv[1, 3] = -tvec[1]
    mv[2, 3] = -tvec[2]
    mv[3, 3] = 1.0
    return mv, projection_matrix


def extract_frustum(camera_matrix, rmat, tvec, width, height):
    mv, proj = get_opengl_matrices(camera_matrix, rmat, tvec, width, height)
    clip = proj * mv
    frustum = np.asmatrix(np.zeros((6, 4)))
    # /* Extract the numbers for the RIGHT plane */
    frustum[0, :] = clip[3, :] - clip[0, :]
    # /* Normalize the result */
    v = frustum[0, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[0, :] = frustum[0, :] / t

    # /* Extract the numbers for the LEFT plane */
    frustum[1, :] = clip[3, :] + clip[0, :]
    # /* Normalize the result */
    v = frustum[1, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[1, :] = frustum[1, :] / t

    # /* Extract the BOTTOM plane */
    frustum[2, :] = clip[3, :] + clip[1, :]
    # /* Normalize the result */
    v = frustum[2, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[2, :] = frustum[2, :] / t

    # /* Extract the TOP plane */
    frustum[3, :] = clip[3, :] - clip[1, :]
    # /* Normalize the result */
    v = frustum[3, :3]
    t = np.sqrt(np.sum(np.multiply(v, v))) 
    frustum[3, :] = frustum[3, :] / t

    # /* Extract the FAR plane */
    frustum[4, :] = clip[3, :] - clip[2, :]
    # /* Normalize the result */
    v = frustum[4, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[4, :] = frustum[4, :] / t

    # /* Extract the NEAR plane */
    frustum[5, :] = clip[3, :] + clip[2, :]
    # /* Normalize the result */
    v = frustum[5, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[5, :] = frustum[5, :] / t
    return frustum


def calc_inside(camera_matrix, rmat, tvec, width, height, obj_points):
    frustum = extract_frustum(camera_matrix, rmat, tvec, width, height)
    inside = 0
    for point in obj_points:
        if point_in_frustum(point[0], point[1], point[2], frustum) > 0:
            inside += 1
    return inside


def point_in_frustum(x, y, z, frustum):
    for p in range(0, 3):
        if (
            frustum[p, 0] * x + frustum[p, 1] * y + frustum[p, 2] + z + frustum[p, 3]
            <= 0
        ):
            return False
    return True


class TextureFrontalizationPreprocessor(FramePreprocessor):
    PREDICTOR_MODEL = "shape_predictor_68_face_landmarks.dat"
    THREE_D_MODEL = "model3Ddlib.mat"
    EYEMASK_MODEL = "eyemask.mat"

    def __init__(self, models_path: str, do_calculate_symmetry: bool = True):
        self._models_path = models_path
        self._do_calculate_symmetry = do_calculate_symmetry

        self._init_detector_and_predictor_for_landmarks()
        self._init_3d_model()
        self._init_eyemask()

    def _init_detector_and_predictor_for_landmarks(self):
        predictor_path = os.path.join(self._models_path, self.PREDICTOR_MODEL)
        self._landmark_detector = dlib.get_frontal_face_detector()
        self._landmark_predictor = dlib.shape_predictor(predictor_path)

    def _init_3d_model(self):
        self._model3D = ThreeD_Model(
            os.path.join(self._models_path, self.THREE_D_MODEL), "model_dlib"
        )

    def _init_eyemask(self):
        self._eyemask = np.asarray(
            scio.loadmat(os.path.join(self._models_path, "eyemask.mat"))["eyemask"]
        )

    def _get_landmarks(self, img):
        lmarks = []
        dets = self._landmark_detector(img, 1)
        shapes = []
        for k, det in enumerate(dets):
            shape = self._landmark_predictor(img, det)
            shapes.append(shape)
            xy = shape_to_np(shape)
            lmarks.append(xy)

        lmarks = np.asarray(lmarks, dtype="float32")
        return lmarks

    def preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        landmarks = self._get_landmarks(frame)
        if len(landmarks) == 0:
            return None

        proj_matrix, _, _, _ = estimate_camera(self._model3D, landmarks[0])
        out = frontalize(
            self._do_calculate_symmetry,
            frame,
            proj_matrix,
            self._model3D.ref_U,
            self._eyemask,
        )

        return out
