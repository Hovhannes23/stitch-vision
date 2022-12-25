import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def resize_and_RGB(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    return image


def compute_norms(images):
    mean_R, mean_G, mean_B = torch.mean(images, dim=[0, 1, 2])
    std_R, std_G, std_B = torch.std(images, dim=[0, 1, 2])
    norm_R = transforms.Normalize(mean_R, std_R)
    norm_G = transforms.Normalize(mean_G, std_G)
    norm_B = transforms.Normalize(mean_B, std_B)
    return norm_R, norm_G, norm_B


def compute_norms_aug(images):
    mean_R, mean_G, mean_B = torch.mean(images, dim=[0, 1, 2, 3])
    std_R, std_G, std_B = torch.std(images, dim=[0, 1, 2, 3])
    norm_R = transforms.Normalize(mean_R, std_R)
    norm_G = transforms.Normalize(mean_G, std_G)
    norm_B = transforms.Normalize(mean_B, std_B)
    return norm_R, norm_G, norm_B


def normalize_image(image, norm_R, norm_G, norm_B):
    image[:, :, 0] = norm_R(image[:, :, 0].unsqueeze(0))
    image[:, :, 1] = norm_G(image[:, :, 1].unsqueeze(0))
    image[:, :, 2] = norm_B(image[:, :, 2].unsqueeze(0))
    return image


def blur_and_Sobel(image):
    img = np.float32(image)
    img = cv2.GaussianBlur(image, (21, 21), 0)
    # img = np.float32(blured_img)

    # for RGB
    img_R = np.zeros((128, 128))
    img_G = np.zeros((128, 128))
    img_B = np.zeros((128, 128))

    img_R = img[:, :, 0]
    img_G = img[:, :, 1]
    img_B = img[:, :, 2]

    grad_x_R = cv2.Sobel(img_R, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_R = cv2.Sobel(img_R, cv2.CV_32F, 0, 1, ksize=3)
    abs_grad_x_R = cv2.convertScaleAbs(grad_x_R)
    abs_grad_y_R = cv2.convertScaleAbs(grad_y_R)
    grad_R = cv2.addWeighted(abs_grad_x_R, 0.5, abs_grad_y_R, 0.5, 0)

    grad_x_G = cv2.Sobel(img_G, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_G = cv2.Sobel(img_G, cv2.CV_32F, 0, 1, ksize=3)
    abs_grad_x_G = cv2.convertScaleAbs(grad_x_G)
    abs_grad_y_G = cv2.convertScaleAbs(grad_y_G)
    grad_G = cv2.addWeighted(abs_grad_x_G, 0.5, abs_grad_y_G, 0.5, 0)

    grad_x_B = cv2.Sobel(img_B, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_B = cv2.Sobel(img_B, cv2.CV_32F, 0, 1, ksize=3)
    abs_grad_x_B = cv2.convertScaleAbs(grad_x_B)
    abs_grad_y_B = cv2.convertScaleAbs(grad_y_B)
    grad_B = cv2.addWeighted(abs_grad_x_B, 0.5, abs_grad_y_B, 0.5, 0)

    grad = np.zeros((128, 128, 3))
    grad = cv2.merge([grad_R, grad_G, grad_B])

    # # for grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


# def normalize_image(image, norm_type=cv2.NORM_L2):
#     image = np.float32(image)
#     # return cv2.normalize(image, None, alpha=1.0, beta=0.0, norm_type=norm_type)
#     return image


def binarizeImage(image):
    retval, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    return image


def canny_edges(image):
    return cv2.Canny(image, 100, 100)
