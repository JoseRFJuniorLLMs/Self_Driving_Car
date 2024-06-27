import cv2
import numpy as np
import time
from collections import deque
import concurrent.futures


def undistort(img, cal_dir='camera    import pickle
    with


open(cal_dir, mode='rb') as f:
file = pickle.load(f)
mtx = file['mtx']
dist = file['dist']
return cv2.undistort(img, mtx, dist, None, mtx


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    sobelx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])]
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def perspective_warp(img, dst_size=(1280, 720)):
    src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    img_size = np.float32([(src = src * img_size
    dst = dst * np.float32(dst_size)
                           M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, dst_size)


def inv_perspective_warp(img, dst_size=(1280, 720)):
    src = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    dst = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, dst_size)


def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5 / 720
    xm_per_pix = 3.7 / 720
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    return (left_curverad, right_curverad, center)


def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((img, img, img)) * 255
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int32(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if draw_windows:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100, 255, 255), 3)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    return out_img, (left_fitx, right_fitx), (left_fit, right_fit), ploty


def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    cv2.fillPoly(color_img, np.int_([points]), (0, 200, 255))
    cv2.polylines(color_img, np.int32([left]), isClosed=False, color=(255, 0, 255), thickness=20)
    cv2.polylines(color_img, np.int32([right]), isClosed=False, color=(0, 255, 255), thickness=20)
    inv_perspective = inv_perspective_warp(color_img)
    result = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return result


def detect_obstacles(gray, img):
    car_cascade = cv2.CascadeClassifier('Support/cars.xml')
    bike_cascade = cv2.CascadeClassifier('Support/bike.xml')
    bus_cascade = cv2.CascadeClassifier('Support/bus.xml')
    pedestrian_cascade = cv2.CascadeClassifier('Support/pedestrian.xml')

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    bikes = bike_cascade.detectMultiScale(gray, 1.1, 1)
    buses = bus_cascade.detectMultiScale(gray, 1.1, 1)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 1)

    obstacles = []
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'Carro', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        obstacles.append('Carro')
    for (x, y, w, h) in bikes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Moto', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        obstacles.append('Moto')
    for (x, y, w, h) in buses:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, 'Ônibus', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        obstacles.append('Ônibus')
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, 'Pedestre', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        obstacles.append('Pedestre')

    return img, obstacles


def process_frame(img):
    img_ = pipeline(img)
    img_ = perspective_warp(img_)
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=True)
    curverad = get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    img = draw_lanes(img, curves[0], curves[1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Curvatura da Pista: {lane_curve:.0f} m', (50, 50), font, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Deslocamento do Veículo: {curverad[2]:.4f} m', (50, 100), font, 1, (255, 255, 255), 2)

    return img, curves[0], curves[1]


def calc_steering(center_dist):
    steering_angle = center_dist * 0.4  # Ajuste este multiplicador para alterar a sensibilidade da direção
    return int(np.clip(steering_angle, -45, 45))


def main():
    cap = cv2.VideoCapture('TEST/Test_3.mp4')

    # Inicializa deques para armazenar valores passados
    left_curves = deque(maxlen=10)
    right_curves = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        start_time = time.time()

        # Usa multithreading para processar o frame e detectar obstáculos simultaneamente
        with concurrent.futures.ThreadPoolExecutor() as executor:
            frame_future = executor.submit(process_frame, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obstacle_future = executor.submit(detect_obstacles, gray, frame)

        # Aguarda a conclusão de ambas as tarefas
        frame, left_fit, right_fit = frame_future.result()
