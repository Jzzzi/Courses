import cv2
import numpy as np

# 加载图像
image = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

# 转换到灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)

cv2.imshow('gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用霍夫变换检测圆形
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000,
                           param1=30, param2=30, minRadius=10, maxRadius=100)

# print(circles)

# 将检测结果转换为整数
circles = np.round(circles[0, :]).astype("int")

# 遍历所有检测到的圆形
for (x, y, r) in circles:
    # 绘制圆形
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    # 绘制圆心
    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# 显示结果图像
cv2.imshow('detected circles', image)
cv2.imwrite('result.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
