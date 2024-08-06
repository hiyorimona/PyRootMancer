
# PyRootMancer - Root Segmentation Project

Welcome to PyRootMancer, the magical tool for all your root segmentation needs! 🌱✨ Whether you're a plant physiologist, agronomist, or ecologist, PyRootMancer is here to help you unravel the mysteries of root systems with the power of deep learning.

Harnessing the enchanting capabilities of the U-Net model architecture, PyRootMancer ensures that your root images are segmented with pinpoint accuracy. This wizardry not only facilitates precise detection and segmentation of root structures but also aids in comprehensive analyses and research.

Root system analysis has never been this easy and fun! By automating the segmentation process, PyRootMancer allows you to efficiently process large datasets, extract meaningful data, and conduct thorough analyses with minimal manual intervention. Get ready to streamline your workflow, enhance segmentation accuracy, and accelerate your discoveries in plant science.


## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Authors](#authors)

## Installation

```sh
pip install pyrootmancer
```

## Dependencies

The project dependencies are managed using Poetry. Below is a list of key dependencies:

- tensorflow-io-gcs-filesystem: `0.30.0`
- tensorflow: `2.16.1`
- tensorflow-intel: `2.16.1`
- opencv-python: `^4.9.0.80`
- pandas: `^2.2.2`
- patchify: `^0.2.3`
- matplotlib: `^3.8.4`
- scikit-image: `^0.23.2`
- skan: `^0.11.1`
- numpy: `1.23.5`
- pytest: `^8.2.0`
- pytest-mock: `^3.14.0`
- sphinx: `^7.3.7`
- coverage: `^7.5.3`
- sphinx_rtd_theme: `2.0.0`
- networkx: `^3.3`

For a complete list of dependencies, refer to the `pyproject.toml` or `requirements.txt` file.

## Usage Guide

This guide provides instructions on how to use the functions from the provided modules, including details on where zip files should be placed for unzipping and how to execute various functionalities.

### Prerequisites

Ensure you have the following installed: - Python 3.x - Required libraries (numpy, pandas, cv2, skimage, etc.)

Before running the scripts, ensure that the necessary zip files are placed in the correct directories.

### Directory Structure

The directory structure should be as follows:

```
project_root/
│
├── data/
│   ├── predictions_clean/
│   ├── test/
│   │   └── [test images]
│   ├── raw/
|   |   └── [zip files]
│   └── output/
│
├── scripts/
│   ├── features/
│   │   └── feature_scaling.py
│   ├── data/
│   │   └── data_preprocessing.py
│   ├── models/
│   │   └── model_training.py
│   └── utils/
│       └── configuration.py
│
└── main.py
```

Place your zip files containing the data predictions in `data/predictions_clean/` and your test images in `data/test/`.

### Unzipping Files

To unzip the files in the specified directories, ensure you have the `DataPipelineSetup` class configured to handle unzipping.

## Feature Scaling Functions
The feature scaling functions are located in `scripts/features/feature_scaling.py`. Below are examples of how to use these functions.

\### Creating and Sorting DataFrame

`create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)`

**Description:** Creates a DataFrame from image coordinates and sorts it based on specified criteria.

**Example usage:**

```python
from pyrootmancer.scripts.features.feature_scaling import create_and_sort_dataframe

image_coord_dst_0 = [100, 200, 300, 400, 500]
image_coord_dst_1 = [50, 150, 250, 350, 450]

sorted_df = create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)
print(sorted_df)
```

\### Getting Image Coordinates

`get_image_coordinates(df, num)
`
**Description:** Retrieves and scales image coordinates from a DataFrame.

**Example usage:**

```python
import pandas as pd
import numpy as np
from pyrootmancer.scripts.features.feature_scaling import get_image_coordinates

df = pd.DataFrame({'X': [50, 150, 250], 'Y': [100, 200, 300]})
num = 1

robot_position = get_image_coordinates(df, num)
print(f"Robot position: {robot_position}")
```

### Landmark Extraction Functions

The landmark extraction functions are located in `scripts/landmark_extraction.py`.

\### Opening and Closing

`opening_closing(img)`

**Description:** Applies morphological operations to remove noise from the input image.

**Example usage:**

```python
import cv2
from pyrootmancer.scripts.landmark_extraction import LandmarkExtraction

landmark_extractor = LandmarkExtraction()
img = cv2.imread('path_to_image', 0)
processed_img = landmark_extractor.opening_closing(img)
cv2.imshow('Processed Image', processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

\### Removing Small Components

`remove_small_components(mask)`

**Description:** Removes small connected components from the mask based on the number of labels.

**Example usage:**

```python
import numpy as np
from pyrootmancer.scripts.landmark_extraction import LandmarkExtraction

landmark_extractor = LandmarkExtraction()
mask = np.zeros((100, 100), dtype=np.uint8)  # Example mask
processed_mask = landmark_extractor.remove_small_components(mask)
```

\### Getting Bottom Coordinates

`get_bottom_coordinates(input_folder, num_img, threshold=50)`

**Description:** Processes and filters coordinates from the skeleton image.

**Example usage:**

```python
from pyrootmancer.scripts.landmark_extraction import LandmarkExtraction

landmark_extractor = LandmarkExtraction()
input_folder = 'data/predictions_clean'
num_img = 1
bottom_coordinates = landmark_extractor.get_bottom_coordinates(input_folder, num_img)
print(f"Bottom coordinates: {bottom_coordinates}")
```

\### Detecting Roots

`detect(chosen_images, input_folder, test_folder, num_img)`

**Description:** Detects roots in the images and overlays results.

**Example usage:**

```python
from pyrootmancer.scripts.landmark_extraction import LandmarkExtraction

landmark_extractor = LandmarkExtraction()
chosen_images = 'example_image.jpg'
input_folder = 'data/predictions_clean'
test_folder = 'data/test'
num_img = 1
result_img = landmark_extractor.detect(chosen_images, input_folder, test_folder, num_img)
cv2.imshow('Result Image', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Instance Segmentation Functions

The instance segmentation functions are located in `scripts/instance_segmentation.py`.

\### Testing Overlaying

`test_overlaying(image_path, output_folder, model_folder, model_name)`

**Description:** Overlays a predicted mask on a randomly selected original image and displays the result.

**Example usage:**

```python
from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

segmenter = InstanceSegmentation()
image_path = 'path_to_image.jpg'
output_folder = 'data/output'
model_folder = 'models'
model_name = 'model.h5'
blended_img = segmenter.test_overlaying(image_path, output_folder, model_folder, model_name)
```

\### Returning Original Size Image

`return_original_size_image(image_path, output_folder)`

**Description:** Resizes predicted masks to match the size of the original images and overlays them.

**Example usage:**

```python
from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

segmenter = InstanceSegmentation()
image_path = 'path_to_predicted_image.jpg'
output_folder = 'data/output'
resized_img = segmenter.return_original_size_image(image_path, output_folder)
```

\### Overlaying for Folder

`overlay(test_folder, predicted_folder, output_folder)`

**Description:** Overlays predicted masks onto test images for a folder.

**Example usage:**

```python
from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

segmenter = InstanceSegmentation()
test_folder = 'data/test'
predicted_folder = 'data/predictions_clean'
output_folder = 'data/output'
segmenter.overlay(test_folder, predicted_folder, output_folder)
```

\### Returning Original Size for Folder

`return_original_size_folder(test_folder, output_folder)`

**Description:** Resizes predicted masks to match the size of the original images and overlays them for all images in a folder.

**Example usage:**

```python
from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

segmenter = InstanceSegmentation()
test_folder = 'data/test'
output_folder = 'data/output'
segmenter.return_original_size_folder(test_folder, output_folder)
```

## Conclusion

Follow the provided examples to utilize the functions effectively in your projects. Ensure that the necessary files are placed in the correct directories and the required dependencies are installed. For any issues, refer to the troubleshooting section in the respective function’s documentation.

## Note

For the full documentation, please see the GitHub Pages.

## Authors

- Simona Dimitrova (222667@buas.nl) - `data_preprocessing.py`
- Jakub Cyba (223860@buas.nl) - `model_training.py`
- Cédric Verhaegh (221350@buas.nl) - `instance_segmentation.py`
- Thomas Pichardo (223834@buas.nl) - `root_coord_extraction.py`
- Samuel Vieira Vasconcelos (211941@buas.nl) - `root_length.py`

