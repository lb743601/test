import cv2
import numpy as np
import gxipy as gx
from PIL import Image
from time import sleep
from periphery import Serial
serial = Serial("/dev/ttyS0", 9600)
green_lower = np.array([35, 43, 46], dtype=np.uint8)
green_upper = np.array([77, 255, 255], dtype=np.uint8)

# Define the color range for yellow in HSV
yellow_lower = np.array([26, 43, 46], dtype=np.uint8)
yellow_upper = np.array([34, 255, 255], dtype=np.uint8)

def check_colors(hsv_colors):
    # Check the first color (row) for being green
    green_mask = cv2.inRange(hsv_colors[0], green_lower, green_upper)
    is_first_color_green = green_mask.any()

    # Check the second color (row) for being yellow
    yellow_mask = cv2.inRange(hsv_colors[1], yellow_lower, yellow_upper)
    is_second_color_yellow = yellow_mask.any()

    return is_first_color_green and is_second_color_yellow
def is_yellow_green(colors):
    color1, color2 = colors

    # 比较红色和蓝色值来区分黄色和绿色
    yellow = green = None

    # 黄色的红色值通常比绿色的高，蓝色值较低
    if color1[0] > color2[0] and color1[2] < color2[2]:
        yellow = color1
        green = color2
    elif color2[0] > color1[0] and color2[2] < color1[2]:
        yellow = color2
        green = color1

    # 判断颜色是否已正确分配
    if yellow is not None and green is not None:
        return np.array_equal(colors[0], yellow)

    # 如果颜色无法准确判断，则返回False
    return False
def pG(image):
    sigma = 10  # sigma = a * complex_index + b
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma)
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    classNum = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.2)
    _, labels, centers = cv2.kmeans(pixels, classNum, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    control_matrix = labels.flatten()
    control_matrix = control_matrix.reshape(image.shape[0], image.shape[1])
    control_matrix = np.float32(control_matrix)
    control_matrix_temp = cv2.normalize(control_matrix, None, 0, 255, cv2.NORM_MINMAX)
    control_matrix_uint8 = control_matrix_temp.astype(np.uint8)
    control_matrix = cv2.resize(control_matrix, (8, 8), interpolation=cv2.INTER_NEAREST)
    control_matrix = control_matrix == 1.0
    return control_matrix,centers,control_matrix_uint8
class camera:
    def __init__(self):
        self.device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self.device_manager.update_device_list()
        if dev_num == 0:
            print("Number of enumerated devices is 0")
            return

    def open_camera(self):
        self.cam = self.device_manager.open_device_by_index(1)

        # exit when the camera is a mono camera
        if self.cam.PixelColorFilter.is_implemented() is False:
            print("This sample does not support mono camera.")
            self.cam.close_device()
            return
        # set continuous acquisition
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    def set_ex_gain(self, ex, gain):
        self.cam.ExposureTime.set(ex)

        # set gain
        self.cam.Gain.set(gain)

        # get param of improving image quality
        if self.cam.GammaParam.is_readable():
            gamma_value = self.cam.GammaParam.get()
            self.gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        else:
            self.gamma_lut = None
        if self.cam.ContrastParam.is_readable():
            contrast_value = self.cam.ContrastParam.get()
            self.contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        else:
            self.contrast_lut = None
        if self.cam.ColorCorrectionParam.is_readable():
            self.color_correction_param = self.cam.ColorCorrectionParam.get()
        else:
            self.color_correction_param = 0

    def stream_on(self):
        self.cam.stream_on()

    def stream_off(self):
        self.cam.stream_off()

    def close_crema(self):
        self.cam.close_device()


    def get_current_image(self):
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            return None
        rgb_image = raw_image.convert("RGB")
        #rgb_image.image_improvement(self.color_correction_param, self.contrast_lut, self.gamma_lut)
        numpy_image = rgb_image.get_numpy_array()
        return numpy_image

    def save_image(self):
        np_image = self.get_current_image()
        img = Image.fromarray(np_image, 'RGB')
        img.save("image22.jpg")
if __name__ == "__main__":
    c = camera()
    c.open_camera()
    c.stream_on()
    c.set_ex_gain(40000, 10)
    while True:
        np_image=c.get_current_image()
        np_image=cv2.resize(np_image,(400,400))
        control_matrix, centers,control_matrix_uint8=pG(np_image)
        if(is_yellow_green(centers)):
            control_matrix_uint8=255-control_matrix_uint8
        cv2.imshow("0-1",control_matrix_uint8)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    c.stream_off()
    c.close_crema()
    cv2.destroyAllWindows()

