import cv2
import numpy as np
import math


def homomorphic_filter(img, gH, gL, d0):
    img_log = np.log1p(np.array(img, dtype="float") / 255)

    m = img.shape[0]
    n = img.shape[1]

    hom_filter = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dx = i - m // 2
            dy = j - n // 2
            d = math.sqrt(dx * dx + dy * dy)
            hom_filter[i, j] = (gH - gL) * (1 - math.exp(-1 * d * d / (d0 * d0))) + gL

    # 1: fft
    img_fft = np.fft.fftshift(np.fft.fft2(img_log))
    # 2: H(u,v)
    img_filtered = img_fft * hom_filter
    # 3: fft**-1
    img_ifft = np.fft.ifft2(np.fft.ifftshift(img_filtered))
    # 4: exp()
    img_exp = np.exp(np.real(img_ifft)) - 1

    img_norm = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.uint8(img_norm * 255)

    cv2.imshow("src", img)
    cv2.imshow("out", img_out)
    cv2.waitKey()


if __name__ == "__main__":
    src = cv2.imread("./homomorphicInput.jpeg", cv2.IMREAD_GRAYSCALE)
    homomorphic_filter(src, 0.2, 0.1, 10)
