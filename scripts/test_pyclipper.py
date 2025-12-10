import cv2
import numpy as np
import matplotlib.pyplot as plt

import pyclipper
from shapely.geometry import Polygon

UNCLIP_RATIO = 1.5


def unclip(box, unclip_ratio=UNCLIP_RATIO):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])

if __name__ == "__main__":
    rect = [(120, 180), (160, 140), (200, 200), (180, 220)]

    pts = np.array(rect, dtype=np.float32)
    min_rect = cv2.minAreaRect(pts)

    box_points = cv2.boxPoints(min_rect)

    plt.figure()

    tri = np.vstack([pts, pts[0]])
    plt.plot(tri[:, 0], tri[:, 1], '-o')

    box_closed = np.vstack([box_points, box_points[0]])
    plt.plot(box_closed[:, 0], box_closed[:, 1], '-o',)

    expanded_box = unclip(box_points)
    box, sside = get_mini_boxes(expanded_box)
    expanded_box_closed = np.vstack([box, box[0]])
    plt.plot(expanded_box_closed[:, 0], expanded_box_closed[:, 1], '-o',)

    plt.axis('equal')
    plt.grid(True)
    plt.show()
    