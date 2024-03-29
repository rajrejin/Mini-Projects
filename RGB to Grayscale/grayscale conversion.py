import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:/FAU/4. WS 23/DSS/Exercises/My Projects/RGB to Grayscale/leaves.jpg')
if img is None:
    print("Could not open or find the image")
else:
    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to floating point
    img_float = img_rgb.astype(float)

    # Split the image into R, G, and B components
    r, g, b = cv2.split(img_float)

    # Lightness method
    lightness = (np.minimum(r, np.minimum(g, b)) + np.maximum(r, np.maximum(g, b))) / 2

    # Average method
    average = (r+g+b)/3

    # Luminosity method
    luminosity = 0.289 * r + 0.5870 * g + 0.1140 * b

    # Display the images
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(img_rgb)
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(lightness, cmap='gray')
    axs[1].set_title('Lightness Method')
    axs[1].axis('off')

    axs[2].imshow(average, cmap='gray')
    axs[2].set_title('Average Method')
    axs[2].axis('off')

    axs[3].imshow(luminosity, cmap='gray')
    axs[3].set_title('Luminosity Method')
    axs[3].axis('off')

    plt.show()