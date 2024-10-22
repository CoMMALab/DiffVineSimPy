import os
from matplotlib import pyplot as plt

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

ipm = 39.3701 / 1000   # inches per mm


def load_rect(path):

    plt.clf()

    rects = []
    with open(path, 'r') as file:
        for line in file:
            x, y, w, h = map(float, line.strip().split())
            rects.append((x / ipm / 1000, y / ipm / 1000, w / ipm / 1000, h / ipm / 1000))

    for rect in rects:
        x, y, w, h = rect
        rectangle = plt.Rectangle((x, y), w, h, fill = True, edgecolor = 'red', linewidth = 2)
        print(rectangle)
        plt.gca().add_patch(rectangle)

    plt.xlim([-0.1, 1])
    plt.ylim([-1, 0.1])

    plt.pause(0.5)


if __name__ == "__main__":
    for filename in sorted(os.listdir('rects')):
        load_rect('rects/' + filename)
