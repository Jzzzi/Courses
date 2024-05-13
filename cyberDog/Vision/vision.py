import cv2
import numpy as np

# 创造窗口
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
# 创造滑条
cv2.createTrackbar('lh', 'mask', 0, 180, lambda x: x)
cv2.createTrackbar('ls', 'mask', 0, 255, lambda x: x)
cv2.createTrackbar('lv', 'mask', 0, 255, lambda x: x)
cv2.createTrackbar('uh', 'mask', 0, 180, lambda x: x)
cv2.createTrackbar('us', 'mask', 0, 255, lambda x: x)
cv2.createTrackbar('uv', 'mask', 0, 255, lambda x: x)
# 加载图像
image = cv2.imread('rgb_left.jpg', cv2.IMREAD_COLOR)
# 根据滑条值创建掩膜
while True:
    # 转换到HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 获取滑条值
    lh = cv2.getTrackbarPos('lh', 'mask')
    ls = cv2.getTrackbarPos('ls', 'mask')
    lv = cv2.getTrackbarPos('lv', 'mask')
    uh = cv2.getTrackbarPos('uh', 'mask')
    us = cv2.getTrackbarPos('us', 'mask')
    uv = cv2.getTrackbarPos('uv', 'mask')
    # 创建掩膜
    mask = cv2.inRange(hsv, (lh, ls, lv), (uh, us, uv))
    # 显示掩膜
    cv2.imshow('mask', mask)
    # 显示原图
    # cv2.imshow('pic', image)
    # 按下ESC退出
    if cv2.waitKey(1) == 27:
        break