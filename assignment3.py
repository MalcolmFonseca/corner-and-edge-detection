import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 1 EDGE DETECTION ###########################################

# PART A ###############################################
def manual_convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    #pad image to handle borders
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    #init output image
    output = np.zeros_like(image)
    
    #perform convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

#load Image
image = plt.imread('image1.png')
if len(image.shape) == 3:
    image = np.mean(image, axis=2)  #convert to grayscale if RGB

def gaussian_filter(size, sigma):
    filter = np.zeros((size, size))
    m = size // 2
    for x in range(-m, m+1):
        for y in range(-m, m+1):
            filter[x+m, y+m] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    #normalize
    return filter / np.sum(filter)

#create Gaussian filters with σ = 1 and σ = 2
gaussian_filter_1 = gaussian_filter(5, 1)
gaussian_filter_2 = gaussian_filter(5, 2)

#apply convolution
smoothed_image_1 = manual_convolution(image, gaussian_filter_1)
smoothed_image_2 = manual_convolution(image, gaussian_filter_2)

#plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(smoothed_image_1, cmap='gray')
plt.title('Smoothed with σ = 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_image_2, cmap='gray')
plt.title('Smoothed with σ = 2')
plt.axis('off')

plt.show()

# PART B ###############################################

#sobel filters
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def sobel_edge_detection(image):
    grad_x = manual_convolution(image, sobel_x)
    grad_y = manual_convolution(image, sobel_y)
    edge_map = np.sqrt(grad_x**2 + grad_y**2)
    return grad_x, grad_y, edge_map

#apply sobel smoothed images
grad_x_1, grad_y_1, edge_map_1 = sobel_edge_detection(smoothed_image_1)
grad_x_2, grad_y_2, edge_map_2 = sobel_edge_detection(smoothed_image_2)

#plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(grad_x_1, cmap='gray')
plt.title('Gradient X (σ = 1)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(grad_y_1, cmap='gray')
plt.title('Gradient Y (σ = 1)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(edge_map_1, cmap='gray')
plt.title('Edge Map (σ = 1)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(grad_x_2, cmap='gray')
plt.title('Gradient X (σ = 2)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(grad_y_2, cmap='gray')
plt.title('Gradient Y (σ = 2)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(edge_map_2, cmap='gray')
plt.title('Edge Map (σ = 2)')
plt.axis('off')

plt.show()

# PART C ###############################################

def gaussian_derivative_filters(size, sigma):
    filter_x = np.zeros((size, size))
    filter_y = np.zeros((size, size))
    m = size // 2
    for x in range(-m, m+1):
        for y in range(-m, m+1):
            g = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
            filter_x[x+m, y+m] = -x / sigma**2 * g
            filter_y[x+m, y+m] = -y / sigma**2 * g
    return filter_x, filter_y

#gaussian filters with σ = 1 and σ = 2
gaussian_derivative_x_1, gaussian_derivative_y_1 = gaussian_derivative_filters(5, 1)
gaussian_derivative_x_2, gaussian_derivative_y_2 = gaussian_derivative_filters(5, 2)

#apply convolution
grad_x_1_deriv = manual_convolution(image, gaussian_derivative_x_1)
grad_y_1_deriv = manual_convolution(image, gaussian_derivative_y_1)
edge_map_1_deriv = np.sqrt(grad_x_1_deriv**2 + grad_y_1_deriv**2)

grad_x_2_deriv = manual_convolution(image, gaussian_derivative_x_2)
grad_y_2_deriv = manual_convolution(image, gaussian_derivative_y_2)
edge_map_2_deriv = np.sqrt(grad_x_2_deriv**2 + grad_y_2_deriv**2)

#plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(grad_x_1_deriv, cmap='gray')
plt.title('Gradient X (σ = 1, Derivative)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(grad_y_1_deriv, cmap='gray')
plt.title('Gradient Y (σ = 1, Derivative)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(edge_map_1_deriv, cmap='gray')
plt.title('Edge Map (σ = 1, Derivative)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(grad_x_2_deriv, cmap='gray')
plt.title('Gradient X (σ = 2, Derivative)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(grad_y_2_deriv, cmap='gray')
plt.title('Gradient Y (σ = 2, Derivative)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(edge_map_2_deriv, cmap='gray')
plt.title('Edge Map (σ = 2, Derivative)')
plt.axis('off')

plt.show()

# PROBLEM 2 CORNER DETECTION ###########################################

def harris_corner_detection(image, k=0.04, window_size=5, sigma=1):
    #compute gradients
    grad_x = manual_convolution(image, sobel_x)
    grad_y = manual_convolution(image, sobel_y)

    #compute products of gradients
    Ixx = grad_x**2
    Iyy = grad_y**2
    Ixy = grad_x * grad_y

    #apply gaussian filter
    gaussian = gaussian_filter(window_size, sigma)
    Ixx = manual_convolution(Ixx, gaussian)
    Iyy = manual_convolution(Iyy, gaussian)
    Ixy = manual_convolution(Ixy, gaussian)

    #compute corner response
    det_M = Ixx * Iyy - Ixy**2
    trace_M = Ixx + Iyy
    R = det_M - k * trace_M**2

    return R

#apply corner detection
corner_response = harris_corner_detection(image)

def non_maximum_suppression(corner_response, threshold=0.01):
    corners = np.zeros_like(corner_response)
    for i in range(1, corner_response.shape[0] - 1):
        for j in range(1, corner_response.shape[1] - 1):
            if corner_response[i, j] > threshold and corner_response[i, j] == np.max(corner_response[i-1:i+2, j-1:j+2]):
                corners[i, j] = 1
    return corners

#apply non-max suppression
corners = non_maximum_suppression(corner_response)

#plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(corner_response, cmap='gray')
plt.title('Corner Response Map')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.scatter(*np.where(corners == 1)[::-1], c='red', s=10)
plt.title('Detected Corners')
plt.axis('off')

plt.show()