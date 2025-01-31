import cv2
import numpy as np
from matplotlib import pyplot
from PIL import Image


def calc_initial_pos(h, w):
    initial_pos = np.random.randint(0, 2 * (h + w))
    if initial_pos >= h:
        initial_pos -= h
    else:
        return initial_pos, 0

    if initial_pos >= w:
        initial_pos -= w
    else:
        return h - 1, initial_pos

    if initial_pos >= h:
        initial_pos -= h
    else:
        return initial_pos, w - 1

    return 0, initial_pos


def black_canvas(size=150):
    return np.zeros((size, size), dtype=np.uint8)


class ApplySpaceDefects:
    def __init__(self, line_prob=1, lava_prob=0, glare_prob=1, line_val=127, lava_val=255, glare_val=(15, 15)):
        self.line_p = line_prob
        self.lava_p = lava_prob
        self.glare_prob = glare_prob

        self.line_val = line_val
        self.lava_val = lava_val
        self.glare_val = glare_val

    def create_gauss(self, image):
        noise = np.zeros_like(image)
        cv2.randn(noise, 20, 20)
        return cv2.add(image, noise)

    def create_lines(self, image):
        blank_image = np.zeros((int(np.ceil(image.shape[0] * np.sqrt(2))), int(np.ceil(image.shape[1] * np.sqrt(2)))),
                               dtype=image.dtype)

        h, w = blank_image.shape[:2]

        for i in range(0, h, 2):
            if np.random.rand() > 0.5:
                continue
            blank_image[i] = self.line_val * (np.random.rand(w) > 0.5)

        angle = np.random.rand() * 180
        mat = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1)
        blank_image = cv2.warpAffine(blank_image, mat, dsize=(w, h))

        offset_x = int(np.ceil((np.sqrt(2) - 1) * image.shape[1] / 2))
        offset_y = int(np.ceil((np.sqrt(2) - 1) * image.shape[0] / 2))
        image = cv2.add(image, blank_image[offset_y: offset_y + image.shape[0], offset_x: offset_x + image.shape[1]])
        return image

    def create_glare(self, image):
        h, w = image.shape[:2]

        part = 0.7
        y, x = calc_initial_pos(int(part * h), int(part * w))
        y += int((1 - part) * h)
        x += int((1 - part) * w)

        y_end = h // 2
        x_end = w // 2

        angle = np.arctan2(y_end - y, x_end - x)
        reference_distance = 2 * h * w / (h + w)
        number_of_steps = np.random.randint(6, 10)

        for i in range(number_of_steps):
            radius = reference_distance * (0.1 + np.random.rand() * 0.1)
            step = reference_distance * (0.05 + np.random.rand() * 0.05)
            color = self.glare_val[0] + np.random.randint(0, self.glare_val[1] + 1)

            tmp = np.zeros_like(image)
            cv2.circle(tmp, (int(x), int(y)), int(radius), (color,), -1)
            image = cv2.add(image, tmp)

            x += np.cos(angle) * step
            y += np.sin(angle) * step

        return image

    def create_lava(self, image):
        h, w = image.shape[:2]
        image_copy = np.zeros_like(image)

        y, x = calc_initial_pos(h, w)

        inside_portion = 0.7
        y_end = np.random.randint(int((1 - inside_portion) / 2 * h), int((0.5 + inside_portion / 2) * h))
        x_end = np.random.randint(int((1 - inside_portion) / 2 * w), int((0.5 + inside_portion / 2) * w))

        angle = np.arctan2(y_end - y, x_end - x)
        reference_distance = 2 * h * w / (h + w)
        step = reference_distance * (0.1 + np.random.rand() * 0.2)
        shrinking = 0.6 + np.random.rand() * 0.2
        radius = reference_distance * (0.1 + np.random.rand() * 0.1)
        number_of_steps = np.random.randint(3, 7)

        for i in range(number_of_steps):
            cv2.circle(image_copy, (int(x), int(y)), int(radius), (self.lava_val,), -1)

            angle_deviation = np.deg2rad(np.random.rand() * 40 - 20)
            angle += angle_deviation
            x += np.cos(angle) * step
            y += np.sin(angle) * step

            step *= shrinking
            radius *= shrinking

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (int(0.35 * reference_distance), int(0.35 * reference_distance)))
        image_copy = cv2.morphologyEx(image_copy, cv2.MORPH_CLOSE, kernel)
        image = np.where(image_copy == self.lava_val, image_copy, image)

        return image

    def apply_defects(self, image):
        image = self.create_gauss(image)
        if np.random.rand() < self.line_p:
            image = self.create_lines(image)
        if np.random.rand() < self.lava_p:
            image = self.create_lava(image)
        if np.random.rand() < self.glare_prob:
            image = self.create_glare(image)
        return image

