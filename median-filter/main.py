import sys
import time

import cv2
import numpy as np
import math

from print_progress import printProgress


def set_range(amount):
    range_abs = int(amount / 2)
    range_ary = []

    for i in range(-1 * range_abs, range_abs + 1):
        range_ary.append(i)

    return range_ary


def median_filter(src, filter_amount):
    func_start = time.time()

    filter_range = set_range(filter_amount)

    if src is None:
        print('Image load failed!')
        sys.exit()

    dst_ary = []

    # print(src.shape[0])  # y 671
    # print(src.shape[1])  # x 635

    y_shape = src.shape[0]
    x_shape = src.shape[1]

    for y in range(0, y_shape):
        line = []
        for x in range(0, x_shape):
            tmp = []
            for _y in range(0, len(filter_range)):
                for _x in range(0, len(filter_range)):
                    y_bound = y + filter_range[_y]
                    x_bound = x + filter_range[_x]

                    printProgress(y, y_shape, "Progress", "Complete")
                    # print("Processing: " + str(math.floor((y / y_shape) * 100)) + "%")

                    if y_bound < 0 or y_bound >= y_shape or x_bound < 0 or x_bound >= x_shape:
                        tmp.append(src[y][x])
                    else:
                        tmp.append(src[y_bound][x_bound])
            tmp.sort()
            line.append(np.median(tmp))
        dst_ary.append(line)

    func_end = time.time()

    print("\n")
    print("Filter applied: " + str(filter_amount))
    print("Time Spend: " + str(math.ceil(func_end - func_start)) + "sec")

    # making border
    dst = cv2.copyMakeBorder(np.asarray(dst_ary, dtype=np.uint8), 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)

    cv2.waitKey()


if __name__ == "__main__":
    src = cv2.imread('./images/lenna.png', cv2.IMREAD_GRAYSCALE)
    median_filter(src, 3)
