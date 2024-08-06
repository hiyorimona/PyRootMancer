
.. _usage:

Usage Guide
===========

This guide provides instructions on how to use the functions from the provided modules, including details on where zip files should be placed for unzipping and how to execute various functionalities.

Prerequisites
-------------
Ensure you have the following installed:
- Python 3.x
- Required libraries (numpy, pandas, cv2, skimage, etc.)

Before running the scripts, ensure that the necessary zip files are placed in the correct directories.

Directory Structure
-------------------
The directory structure should be as follows:

.. code-block:: text

    project_root/
    │
    ├── data/
    │   ├── predictions_clean/
    │   │   └── [zip files]
    │   ├── test/
    │   │   └── [test images]
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

Place your zip files containing the data predictions in ``data/predictions_clean/`` and your test images in ``data/test/``.

Unzipping Files
---------------
To unzip the files in the specified directories, ensure you have the ``DataPipelineSetup`` class configured to handle unzipping.

Feature Scaling Functions
-------------------------
The feature scaling functions are located in ``scripts/features/feature_scaling.py``. Below are examples of how to use these functions.

### Creating and Sorting DataFrame

``create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)``

**Description:** Creates a DataFrame from image coordinates and sorts it based on specified criteria.

**Example usage:**

.. code-block:: python

    from pyrootmancer.scripts.features.feature_scaling import create_and_sort_dataframe

    image_coord_dst_0 = [100, 200, 300, 400, 500]
    image_coord_dst_1 = [50, 150, 250, 350, 450]

    sorted_df = create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)
    print(sorted_df)

### Getting Image Coordinates

``get_image_coordinates(df, num)``

**Description:** Retrieves and scales image coordinates from a DataFrame.

**Example usage:**

.. code-block:: python

    import pandas as pd
    import numpy as np
    from pyrootmancer.scripts.features.feature_scaling import get_image_coordinates

    df = pd.DataFrame({'X': [50, 150, 250], 'Y': [100, 200, 300]})
    num = 1

    robot_position = get_image_coordinates(df, num)
    print(f"Robot position: {robot_position}")

Landmark Extraction Functions
-----------------------------
The landmark extraction functions are located in ``scripts/landmark_extraction.py``.

### Opening and Closing

``opening_closing(img)``

**Description:** Applies morphological operations to remove noise from the input image.

**Example usage:**

.. code-block:: python

    import cv2
    from pyrootmancer.scripts.landmark_extraction import LandmarkExtraction

    landmark_extractor = LandmarkExtraction()
    img = cv2.imread('path_to_image', 0)
    processed_img = landmark_extractor.opening_closing(img)
    cv2.imshow('Processed Image', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

### Removing Small Components

``remove_small_components(mask)``

**Description:** Removes small connected components from the mask based on the number of labels.

**Example usage:**

.. code-block:: python

    import numpy as np
    from pyrootmancer.scripts.landmark_extraction import LandmarkExtraction

    landmark_extractor = LandmarkExtraction()
    mask = np.zeros((100, 100), dtype=np.uint8)  # Example mask
    processed_mask = landmark_extractor.remove_small_components(mask)

### Getting Bottom Coordinates

``get_bottom_coordinates(input_folder, num_img, threshold=50)``

**Description:** Processes and filters coordinates from the skeleton image.

**Example usage:**

.. code-block:: python

    from pyrootmancer.scripts.landmark_extraction import LandmarkExtraction

    landmark_extractor = LandmarkExtraction()
    input_folder = 'data/predictions_clean'
    num_img = 1
    bottom_coordinates = landmark_extractor.get_bottom_coordinates(input_folder, num_img)
    print(f"Bottom coordinates: {bottom_coordinates}")

### Detecting Roots

``detect(chosen_images, input_folder, test_folder, num_img)``

**Description:** Detects roots in the images and overlays results.

**Example usage:**

.. code-block:: python

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

Instance Segmentation Functions
-------------------------------
The instance segmentation functions are located in ``scripts/instance_segmentation.py``.

### Testing Overlaying

``test_overlaying(image_path, output_folder, model_folder, model_name)``

**Description:** Overlays a predicted mask on a randomly selected original image and displays the result.

**Example usage:**

.. code-block:: python

    from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

    segmenter = InstanceSegmentation()
    image_path = 'path_to_image.jpg'
    output_folder = 'data/output'
    model_folder = 'models'
    model_name = 'model.h5'
    blended_img = segmenter.test_overlaying(image_path, output_folder, model_folder, model_name)

### Returning Original Size Image

``return_original_size_image(image_path, output_folder)``

**Description:** Resizes predicted masks to match the size of the original images and overlays them.

**Example usage:**

.. code-block:: python

    from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

    segmenter = InstanceSegmentation()
    image_path = 'path_to_predicted_image.jpg'
    output_folder = 'data/output'
    resized_img = segmenter.return_original_size_image(image_path, output_folder)

### Overlaying for Folder

``overlay(test_folder, predicted_folder, output_folder)``

**Description:** Overlays predicted masks onto test images for a folder.

**Example usage:**

.. code-block:: python

    from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

    segmenter = InstanceSegmentation()
    test_folder = 'data/test'
    predicted_folder = 'data/predictions_clean'
    output_folder = 'data/output'
    segmenter.overlay(test_folder, predicted_folder, output_folder)

### Returning Original Size for Folder

``return_original_size_folder(test_folder, output_folder)``

**Description:** Resizes predicted masks to match the size of the original images and overlays them for all images in a folder.

**Example usage:**

.. code-block:: python

    from pyrootmancer.scripts.instance_segmentation import InstanceSegmentation

    segmenter = InstanceSegmentation()
    test_folder = 'data/test'
    output_folder = 'data/output'
    segmenter.return_original_size_folder(test_folder, output_folder)

Conclusion
----------
Follow the provided examples to utilize the functions effectively in your projects. Ensure that the necessary files are placed in the correct directories and the required dependencies are installed. For any issues, refer to the troubleshooting section in the respective function's documentation.
