import cv2
import numpy as  np

# 水平翻转
def horizon_flip(img):
    return img[:, ::-1]


def vertical_flip(img):
    return img[::-1]


def crop(img, crop_x, crop_y):
    '''
    读取部分图像，进行裁剪
    :param img:
    :param crop_x:裁剪x尺寸
    :param crop_y:裁剪y尺寸
    :return:
    '''
    import random
    rows, cols = img.shape[:2]
    # 偏移像素点
    x_offset = random.randint(0, cols - crop_x)
    y_offset = random.randint(0, rows - crop_y)

    # 读取部分图像
    img_part = img[y_offset:(y_offset + crop_y), x_offset:(x_offset + crop_x)]

    return img_part


def resize(img):
    width, height, _ = img.shape
    return cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)


def move(image, dx=20, dy=10):
    '''
    使用Numpy数组构建矩阵，数据类型是np.float32，然后传给函数cv2.warpAffine();
    函数cv2.warpAffine() 的第三个参数的是输出图像的大小，它的格式应该是图像的（宽，高）。
    应该记住的是图像的宽对应的是列数，高对应的是行数
    '''
    # 平移矩阵[[1,0,-100],[0,1,-12]]
    # x 下移动 100
    # y 下移动 12

    width, height, _ = image.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    img_change = cv2.warpAffine(image, M, (width, height))
    cv2.imshow("res", img_change)
    cv2.waitKey(0)


def rotate(img, rate=30, sclae=1):
    rows, cols = img.shape[:2]
    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子以及窗口大小来防止旋转后超出边界的问题。
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rate, sclae)
    # 第三个参数是输出图像的尺寸中心
    dst = cv2.warpAffine(img, M, (cols, rows))
    print(dst.shape)
    return dst


def GaussianBlur(img):  # 图片高斯模糊
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = np.clip(img, 0, 255)
    return img


file_path = "../training/African/00000/00000.jpg"
image = cv2.imread(file_path, cv2.IMREAD_COLOR)
cv2.imshow('org', image)
cv2.waitKey()
