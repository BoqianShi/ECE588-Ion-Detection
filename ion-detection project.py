# %%
# ECE 588 Final Project
# Ion Chain detection

# Group member: Ashrit Verma, Boqian Shi, Shaorong Ma

# Part 1: Image loading

# Import necessary libraries
# Method 1 DoG
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters, transform, color, exposure

def _difference_of_gaussians(image, low_sigma, preserve_range=True):
    """
    Apply a bandpass filter for blob detection by computing two gaussian filters (with sigma
    values low_sigma and low_sigma*1.6) and subtracting the more-blurred image with the
    less-blurred image.
    """
    im1 = filters.gaussian(image, sigma=low_sigma, preserve_range=preserve_range)
    im2 = filters.gaussian(image, sigma=low_sigma*1.6, preserve_range=preserve_range)
    filt_image = (im1 - im2).clip(min=0)  # apply bandpass filter (difference of gaussians)

    return filt_image

# Load the image
image_path = 'eth_seven_pts.png' 


# Grayscale Transform
image = io.imread(image_path, as_gray=True)  


# Apply the DoG algorithm
low_sigma = 9  # The best value after experiments
denoised_image = _difference_of_gaussians(image, low_sigma)

# Scale the denoised image into the same size
target_height, target_width = 960, 1280
current_height, current_width = denoised_image.shape

# Calculate new width and height, maintaining aspect ratio
scaled_width = int(current_width * (target_height / current_height))
scaled_image = transform.resize(denoised_image, (target_height, scaled_width))

# Crop to the middle part if the image is too wide
if scaled_width > target_width:
    start_x = scaled_width // 2 - target_width // 2
    scaled_image = scaled_image[:, start_x:start_x + target_width]

# Convert to 3 channels (RGB)
scaled_image_rgb = color.gray2rgb(scaled_image)
scaled_image_rgb = exposure.equalize_adapthist(scaled_image_rgb)

# Display the results for comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(scaled_image_rgb)
ax[1].set_title('Denoised and Rescaled Image')
ax[1].axis('off')

plt.show()


# %%
# Multiple images processing

import os

# Directory containing the all the images
dataset_directory = './dataset'

# Iterate over all files in the dataset directory
for filename in os.listdir(dataset_directory):
    file_path = os.path.join(dataset_directory, filename)

    # Similar work with previous code
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = io.imread(file_path, as_gray=True)

        # Apply the DoG algorithm
        denoised_image = _difference_of_gaussians(image, low_sigma)

        current_height, current_width = denoised_image.shape
        scaled_width = int(current_width * (target_height / current_height))
        scaled_image = transform.resize(denoised_image, (target_height, scaled_width))

        if scaled_width > target_width:
            start_x = scaled_width // 2 - target_width // 2
            scaled_image = scaled_image[:, start_x:start_x + target_width]

        scaled_image_rgb = color.gray2rgb(scaled_image)
        scaled_image_rgb = exposure.equalize_adapthist(scaled_image_rgb)


# %%
# Comparison methods:

# 1. Median Filtering
from skimage.filters import median

def median_filter_denoise(image):
    """
    Apply Median Filtering for denoising an image.
    """
    return median(image)

# Apply Median Filtering
median_denoised_image = median_filter_denoise(image)


# %%
# 2. Bilateral Filtering

from skimage.restoration import denoise_bilateral

def bilateral_filter_denoise(image, sigma_color, sigma_spatial):
    """
    Apply Bilateral Filtering for denoising an image.
    """
    return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

# Apply Bilateral Filtering
sigma_color, sigma_spatial = 0.05, 15  # Adjust these parameters as needed
bilateral_denoised_image = bilateral_filter_denoise(image, sigma_color, sigma_spatial)


# %%
# Adjust the subplot layout to 2x2
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Display original image
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

# Display DoG denoised image
ax[0, 1].imshow(denoised_image, cmap='gray')
ax[0, 1].set_title('DoG Denoised Image (sigma = 9)')
ax[0, 1].axis('off')

# Display Median filtered image
ax[1, 0].imshow(median_denoised_image, cmap='gray')
ax[1, 0].set_title('Median Filtered Image')
ax[1, 0].axis('off')

# Display Bilateral filtered image
ax[1, 1].imshow(bilateral_denoised_image, cmap='gray')
ax[1, 1].set_title('Bilateral Filtered Image(sigma = 0.05, spatial = 15))')
ax[1, 1].axis('off')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('comparison_denoising_methods.png', dpi=300)  # Save with high resolution
plt.show()


# %%
# Normalize and output the image

# Normalize the image to the range [0, 255]
normalized_image = (scaled_image_rgb - np.min(scaled_image_rgb)) / (np.max(scaled_image_rgb) - np.min(scaled_image_rgb))
normalized_image = (normalized_image * 255).astype(np.uint8)

# Save the normalized image
output_path = './scaled_image_rgb.png'  # PNG format
io.imsave(output_path, normalized_image)

print(f"Denoised image saved as {output_path}")


# %%
# Segmentation methods
# 1. Detect local maxima
from skimage.feature import peak_local_max

def _max_coord(image, width=False):
    """
    Detect local maxima in the provided image.
    If width is True, return also the max width of all the ions.
    """
    
    # Detect local maxima
    coordinates = peak_local_max(image, threshold_abs=1, min_distance=15)

    if width:
        try:
            max_width = _ion_max_spot_size(image[coordinates[0, 0]])
        except IndexError:
            max_width = 0

        return coordinates, max_width

    return coordinates

# Example usage:
# Assuming 'denoised_image' from the previous code block
# And assuming 'settings.BLOB_SIZE' and 'settings.ION_DIST' are defined
# Example usage:
coords = _max_coord(scaled_image_rgb)
# Convert 'coords' to the expected format [(x, y, type), ...]
# Assuming type is 1 for all points for demonstration
local_maxima_points = [(coord[1], coord[0], 1) for coord in coords]

# If no coordinates found, try adjusting parameters
if len(coords) == 0:
    print("No local maxima detected. Adjusting parameters to experiment.")
    for thresh in np.linspace(0.5, 2, 4):  # Example range, adjust as needed
        for dist in range(1, 5):  # Example range, adjust as needed
            coords = peak_local_max(scaled_image_rgb, threshold_abs=thresh, min_distance=dist)
            if len(coords) > 0:
                print(f"Local maxima detected with threshold {thresh} and distance {dist}")
                break
        if len(coords) > 0:
            break

print("Coordinates of local maxima:", coords)

# %%
# Plot drawing functions
import cv2
import numpy as np

def plot_detection(image, points, ring_radius=10, dot_radius=3, ring_thickness=2, alpha=0.5):
    # Convert the image to a 3-channel image for color drawing
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for point in points:
        # Check the type of ion and select the color accordingly
        if point[2] == 1:
            # Yellow for Bright Ion (BGR format)
            center_color = (0, 255, 255)
            ring_color = (255, 255, 255)
        else:
            # Green for Bright Ion (BGR format)
            center_color = (0, 255, 0)
            ring_color = (0, 0, 0)
        ring_radius = 20
        dot_radius = 5
        alpha  =  0.15
        # Create an overlay for the ring
        overlay = image.copy()
        cv2.circle(overlay, tuple(np.flip(point[:2])), ring_radius, color=ring_color, thickness=ring_thickness)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw the smaller filled circle at the center
        image = cv2.circle(image, tuple(np.flip(point[:2])), dot_radius, color=center_color, thickness=-1)

    return image


if scaled_image_rgb.dtype != np.uint8:
    scaled_image_rgb_cv2 = (scaled_image_rgb * 255).astype(np.uint8)
else:
    scaled_image_rgb_cv2 = scaled_image_rgb

highlighted_image = plot_detection(scaled_image_rgb, coords)

# Display the highlighted image
plt.imshow(highlighted_image)
plt.axis('off')
plt.show()


# %% [markdown]
# #### Coutour centering

# %%
# Segmentation method 2: Contour Detection
# Centroids of the contours can be used as the coordinates of the ions
import cv2
import numpy as np

def contour_detection(image):
    """
    Apply Contour Detection to segment an image.
    """
    # Convert image to float and scale if it's not in float format
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float64) / 255

    # Convert to grayscale if it's a 3-channel image
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Ensure the image is in uint8 format
    image_gray_uint8 = (image_gray * 255).astype(np.uint8) if image_gray.dtype != np.uint8 else image_gray

    # Apply a binary threshold to get a binary image
    _, binary_image = cv2.threshold(image_gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a blank canvas
    contour_image = np.zeros_like(image_gray_uint8)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1) 

    return contour_image, contours

# Ensure scaled_image_rgb is properly converted before applying contour detection
scaled_image_rgb_uint8 = (scaled_image_rgb * 255).astype(np.uint8) if scaled_image_rgb.dtype != np.uint8 else scaled_image_rgb
contour_segmented_image, contours = contour_detection(scaled_image_rgb_uint8)


# %%
# Find the centroids of the largest contours
def find_centroids_of_largest_contours(contours, num_contours):
    """
    Find the centroids of the largest 'num_contours' contours.
    """
    # Sort contours by area, descending
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    centroids = []
    for contour in sorted_contours[:num_contours]:
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY, 1))  # Assuming type is 1 for all points
        else:
            # Handle the case where contour area is zero
            x, y, _, _ = cv2.boundingRect(contour)
            centroids.append((x, y, 1))

    return centroids
ion_num = 7

contour_centroids = find_centroids_of_largest_contours(contours, ion_num)
contour_centroids = [(contour_centroid[1], contour_centroid[0], 1) for contour_centroid in contour_centroids]

contour_highlighted = plot_detection(scaled_image_rgb, contour_centroids)
plt.imshow(contour_highlighted)
plt.axis('off')
plt.show()


# %%
# Segmentation part comparisons
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a 1 x 2 subplot layout

# Local Maxima Segmentation
axs[0].imshow(highlighted_image)  # Use axs[0] for the first subplot
axs[0].set_title('Local Maxima Segmentation')
axs[0].axis('off')

# Contour Detection Segmentation
axs[1].imshow(contour_highlighted)  # Use axs[1] for the second subplot
axs[1].set_title('Contour Detection Segmentation')
axs[1].axis('off')

plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Measured by hand, differnet for each image
hand_measured = [
    (540, 769),
    (543, 828),
    (545, 881),
    (546, 931),
    (548, 981),
    (547, 1033),
    (549, 1094)
]
ion_num = 7  # Replace with your actual ion number

# Sort lists by y-coordinate
coords_sorted = sorted(coords[:ion_num], key=lambda x: x[1])
contour_centroids_sorted = sorted(contour_centroids[:ion_num], key=lambda x: x[1])
hand_measured_sorted = sorted(hand_measured, key=lambda x: x[1])

# Calculate differences and variance
coords_differences = [euclidean(coords_sorted[i][:2], hand_measured_sorted[i]) for i in range(ion_num)]
contour_centroids_differences = [euclidean(contour_centroids_sorted[i][:2], hand_measured_sorted[i]) for i in range(ion_num)]

# Calculate the total variance for each method
total_variance_local_maxima = sum(d**2 for d in coords_differences)
total_variance_contour_centroids = sum(d**2 for d in contour_centroids_differences)

# Create a DataFrame for comparison with the specified column order
comparison_df = pd.DataFrame({
    'Hand Measured Coords (x, y)': [f"({x}, {y})" for x, y in hand_measured_sorted],
    'Local Maxima Coords (x, y)': [f"({x}, {y})" for x, y, _ in coords_sorted],
    'Contour Centroids Coords (x, y)': [f"({x}, {y})" for x, y, _ in contour_centroids_sorted]
})

# Add a final row for the total variance
final_row_df = pd.DataFrame([{
    'Hand Measured Coords (x, y)': "Total Variance",
    'Local Maxima Coords (x, y)': total_variance_local_maxima,
    'Contour Centroids Coords (x, y)': total_variance_contour_centroids
}], index=[len(comparison_df)])

# Concatenate the final row to the DataFrame
comparison_df = pd.concat([comparison_df, final_row_df])

# Save the DataFrame to an image
fig, ax = plt.subplots(figsize=(8, 3))  # Set figure size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=comparison_df.values, colLabels=comparison_df.columns, loc='center', cellLoc='center', colColours=["lightblue"]*comparison_df.shape[1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)


plt.gcf().set_facecolor("white")  # Set background to white
plt.savefig('./comparison_table.png', bbox_inches='tight', pad_inches=0.05, dpi=300)  # Save as image
plt.show()
plt.close()
7


