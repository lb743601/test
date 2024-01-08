import cv2
import numpy as np
# from skimage import feature

# my environment: Python 3.7 + opencv 4.6.0

# calculate channel complexity
#================================================================
def calculate_complexity(image, bin_num = 256):
    # rescale
    image_normalized = cv2.normalize(image,None,0.0,255.0,cv2.NORM_MINMAX)

    # calculate image variance
    image_max_variance = np.sqrt((1-1/bin_num)**2 + (1/bin_num)**2 * (bin_num-1))

    # calculate histogram
    histogram, bins = np.histogram(image_normalized.flatten(),bin_num-1,(0,bin_num-1))
    # rescale histogram
    histogram = histogram.astype(np.float64)
    histogram_rescale = histogram / histogram.sum()

    uniform_prob = 1/bin_num
    sum_squared_diff = np.sum((histogram_rescale - uniform_prob)**2)

    complex = 1 - np.sqrt(sum_squared_diff) / image_max_variance

    return complex

# main code
#================================================================
if __name__ == '__main__':
    # read color image
    image = cv2.imread('./data/image.png', cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image from path")

    # save original color image
    # cv2.imwrite('./results/originalColorImage.bmp',image)

    # step 1: calculate the color complex coefficients
    #   including: color complex, texture complex
    # calculate color complex
    # image_B, image_G, image_R = cv2.split(image)
    # cv2.imwrite('./results/image_B.bmp',image_B)
    # cv2.imwrite('./results/image_G.bmp',image_G)
    # cv2.imwrite('./results/image_R.bmp',image_R)
    # colorComplex_B = calculate_complexity(image_B)
    # colorComplex_G = calculate_complexity(image_G)
    # colorComplex_R = calculate_complexity(image_R)
    # alpha_B = 1/3
    # alpha_G = 1/3
    # alpha_R = 1/3
    # color_complex = alpha_B * colorComplex_B + alpha_G * colorComplex_G + alpha_R * colorComplex_R
    # del image_B, image_G, image_R, colorComplex_B, colorComplex_G, colorComplex_R, alpha_B, alpha_G, alpha_R

    # calculate texture complex
    # hog_features, hog_image = feature.hog(image, visualize = True, channel_axis = -1)
    # hog_image_normalized = cv2.normalize(hog_image,None,0.0,255.0,cv2.NORM_MINMAX)
    # cv2.imwrite('./results/hog_image.bmp',hog_image_normalized)
    # texture_complex = calculate_complexity(hog_image)
    # del hog_features, hog_image_normalized, hog_image

    # step 2: based on color complex coefficients, gaussian blurring
    # omega1 = 0.2
    # omega2 = 1 - omega1
    # complex_index = omega1 * color_complex + omega2 * texture_complex
    # a = 22
    # b = 9.2
    sigma = 10 # sigma = a * complex_index + b
    blurred_image = cv2.GaussianBlur(image,(5,5),sigma)
    # cv2.imwrite('./results/blurred_image.bmp',blurred_image)
    # del omega1, omega2, complex_index, a, b, sigma, texture_complex, color_complex

    # step 3: color pattern clustering using k-means
    pixels = image.reshape((-1,3))
    pixels = np.float32(pixels)

    classNum = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.2)
    _, labels, centers = cv2.kmeans(pixels, classNum, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    # segmented_image = centers[labels.flatten()]
    # segmented_image = segmented_image.reshape(image.shape)
    # cv2.imwrite('./results/segmented_image.bmp', segmented_image)

    control_matrix = labels.flatten()
    control_matrix = control_matrix.reshape(image.shape[0], image.shape[1])
    control_matrix = np.float32(control_matrix)
    # control_matrix_temp = cv2.normalize(control_matrix,None,0,255,cv2.NORM_MINMAX)
    control_matrix = cv2.resize(control_matrix,(8,8),interpolation = cv2.INTER_NEAREST)
    control_matrix = control_matrix == 1.0
    # cv2.imwrite('./results/control_matrix.bmp', control_matrix_temp)