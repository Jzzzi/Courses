import cv2
import tkinter as tk

if __name__ == '__main__':
    rgb = cv2.imread('rgbImage.png')
    infra1 = cv2.imread('infra1Image.png')
    depth = cv2.imread('depthImage.png')
    fx = 388.5847168
    fy = 388.5847168
    cx = 322.05200195
    cy = 236.50979614

    gx = 470.0
    gy = 464.68
    dx = 300.0
    dy = 250.50979614

    center_rgb = (500.0, 200.0)
    center_infra1 = ((center_rgb[0]-dx)/gx*fx+cx, (center_rgb[1]-dy)/gy*fx+cy)
    cv2.circle(rgb, (int(center_rgb[0]), int(center_rgb[1])), 5, (0, 0, 255), -1)
    cv2.circle(infra1, (int(center_infra1[0]), int(center_infra1[1])), 5, (0, 0, 255), -1)
    # cv2.circle(infra1, (int(cx), int(cy)), 5, (0, 255, 0), -1)
    cv2.circle(depth, (int(center_infra1[0]), int(center_infra1[1])), 5, (0, 0, 255), -1)

    cv2.imshow('rgb', rgb)
    cv2.imshow('infra1', infra1)
    cv2.imshow('depth', depth)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()