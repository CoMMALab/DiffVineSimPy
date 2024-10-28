import cv2
import numpy as np
from sklearn.cluster import KMeans

# for better visualization
def display_large(img, k=3):
    if img is None:
        print("Error: Image not found.")
        return
    #img = cv2.bilateralFilter(img, 9, 75, 75)
    #img = (img * 255/k).astype(np.uint8)
    # Resize image to double its size for better visibility
    scale_factor = 6  # Adjust this factor as needed
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# classifies pixels into floor, wall, vine (doesnt work, finds shadows instead of vine)
def basic3(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Reshape the image to be a list of pixels
    pixels = img.reshape((-1, 3))

    # Use k-means to reduce the image to 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_

    # Determine which cluster is the most frequent
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    floor_label = labels[np.argmax(counts)]
    floor_color = kmeans.cluster_centers_[floor_label]
    print(kmeans.cluster_centers_)
    # Find the darkest color, assuming it's the wall
    brightness = np.sum(dominant_colors, axis=1)
    wall_label = labels[np.argmin(brightness)]

    # The remaining label is for the worm
    worm_label = list(set(labels) - {floor_label, wall_label})[0]

    # Create a labeled image based on the identified clusters
    label_img = np.zeros_like(kmeans.labels_)
    label_img[kmeans.labels_ == floor_label] = 1  # Floor
    label_img[kmeans.labels_ == wall_label] = 2   # Wall
    label_img[kmeans.labels_ == worm_label] = 3   # Worm

    # Reshape the labeled image back to the original image shape
    labeled_img = label_img.reshape(img.shape[0], img.shape[1])
    scaled_img = (labeled_img * 85).astype(np.uint8)
    #display_large(scaled_img, 3)
    return scaled_img

# classifies into k to see which one gives clearest
# at k= 5, we can see the vine
def basicK(img, k):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV better than RGB

    # Reshape the image to be a list of pixels
    pixels = img.reshape((-1, 3))

    # Use k-means to reduce the image to 3 clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)

    # Reshape the labeled image back to the original image shape
    labeled_img = kmeans.labels_.reshape(img.shape[0], img.shape[1])
    scaled_img = (labeled_img * 255/k).astype(np.uint8)
    #img = treatment(scaled_img)
    #display_large(img)
    return scaled_img

# scale up saturation to hopefully detect walls better (didnt work at all)
def satK(img, k):
    if img is None:
        print("Error: Image not found.")
        return

    # Convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Optionally increase the saturation by a scale factor
    saturation_scale = 1.5
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_scale, 0, 255)

    # Reshape the image for clustering
    pixel_values = hsv_img.reshape((-1, 3))
    # Optionally use only H and S channels
    pixel_values = pixel_values[:, :2]  # Using only H and S for clustering

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixel_values)
    labeled_img = kmeans.labels_.reshape(hsv_img.shape[0], hsv_img.shape[1])

    scaled_img = (labeled_img * int(255/k)).astype(np.uint8)
    display_large(scaled_img)
    return scaled_img

# post classification, unite shades of walls
def treatment(img):
    img = cv2.medianBlur(img, 3)
    #img = (img/85).astype(np.uint8)
    flattened_img = img.flatten()
    for i, pixel in enumerate(flattened_img):
        if pixel == 0:
            flattened_img[i] = 85

    # to be implemented:
    # segment walls, use rectangle bound to find vertical vs horizontal
    # and crop top of horizontal segments for better top-down view
    
    img = flattened_img.reshape(img.shape[0], img.shape[1])
    return img
img = cv2.imread('./data/frames/vid0/frame_0027.jpg')
img2 = cv2.imread('./data/frames/vid0/frame_1000.jpg')
#basic3(img)
#pic = basicK(img, 3)
#pic = treatment(pic)
#display_large(pic, 3)
#basicK(img, 3)
#satK(img, 6)