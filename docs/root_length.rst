
Root Length Calculator Class
============================

This module defines a class `RootLengthCalculator` for predicting and measuring the length of roots in images using a trained deep learning model.

Initialization
--------------

**Description:** Initializes the RootLengthCalculator class.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import RootLengthCalculator

   # Example usage
   img_dir = '/path/to/images'
   model_path = '/path/to/model.h5'
   custom_objects = {'f1': f1, 'iou': iou}
   root_calculator = RootLengthCalculator(img_dir, model_path, custom_objects)

Prediction Method
-----------------

**Description:** Predicts the mask for the given image using the loaded model.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import predict_image

   # Example usage
   img = cv2.imread('/path/to/image.jpg')
   predicted_mask = predict_image(root_calculator, img, 256)

Image Processing
----------------

**Description:** Processes all images in the specified directory.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import process_images

   # Example usage
   process_images(root_calculator, img_dir)

Analysis and Annotation
-----------------------

**Description:** Analyzes the image to find and annotate parts based on the predicted mask.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import analyze_image

   # Example usage
   img = cv2.imread('/path/to/image.jpg')
   predicted_mask = predict_image(root_calculator, img, 256)
   analyze_image(root_calculator, 'example_image.jpg', img, predicted_mask)

Length Calculation
------------------

**Description:** Calculates the length of the main root.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import calculate_length

   # Example usage
   skeleton = skeletonize(predicted_mask > 0.5)
   summary = summarize(Skeleton(skeleton))
   length = calculate_length(root_calculator, skeleton, summary)

Results Saving
--------------

**Description:** Saves the results to a CSV file.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import save_results

   # Example usage
   csv_filename = '/path/to/save/results.csv'
   save_results(root_calculator, csv_filename)

Troubleshooting
---------------

**Issue:** Model loading fails with a TensorFlow error.

**Solution:** Ensure `model_path` points to a valid trained Keras model file. Check TensorFlow version compatibility and verify that required custom objects (`custom_objects`) are correctly specified.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import RootLengthCalculator

   img_dir = '/path/to/images'
   model_path = '/incorrect/path/to/model.h5'  # Example of incorrect path
   custom_objects = {'f1': f1, 'iou': iou}
   
   try:
       root_calculator = RootLengthCalculator(img_dir, model_path, custom_objects)
   except Exception as e:
       print(f"Error: {e}")

---

**Issue:** No parts detected in images.

**Solution:** Adjust parameters `min_area`, `min_top`, `max_top`, and `max_left` in the `analyze_image` method to better fit the characteristics of your images.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import analyze_image

   img = cv2.imread('/path/to/image.jpg')
   predicted_mask = predict_image(root_calculator, img, 256)
   analyze_image(root_calculator, 'example_image.jpg', img, predicted_mask)

---

**Issue:** Incorrect results saved to CSV.

**Solution:** Verify that the `save_results` method is called after `process_images` and check the format of the `csv_filename`. Ensure permissions are set correctly for file writing.

.. code-block:: python

   from pyrootmancer.scripts.features.root_length import save_results

   csv_filename = '/incorrect/path/to/save/results.csv'  # Example of incorrect path
   save_results(root_calculator, csv_filename)

Execution
---------

.. code-block:: python

   if __name__ == "__main__":
       img_dir = "/path/to/images"
       model_path = "/path/to/model.h5"
       custom_objects = {"f1": f1, "iou": iou}
       csv_filename = "/path/to/save/results.csv"
       
       root_calculator = RootLengthCalculator(img_dir, model_path, custom_objects)
       process_images(root_calculator, img_dir)
       save_results(root_calculator, csv_filename)

Function Examples
-----------------

.. rst:directive:: py:function:: pyrootmancer.scripts.features.root_length.RootLengthCalculator.predict_image(img, target_size)

   Example usage:
   
   .. code-block:: python
   
      from pyrootmancer.scripts.features.root_length import RootLengthCalculator

      img_dir = '/path/to/images'
      model_path = '/path/to/model.h5'
      custom_objects = {'f1': f1, 'iou': iou}
      root_calculator = RootLengthCalculator(img_dir, model_path, custom_objects)

      img = cv2.imread('/path/to/image.jpg')
      predicted_mask = root_calculator.predict_image(img, 256)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.root_length.RootLengthCalculator.analyze_image(file_name, img, predicted_mask)

   Example usage:
   
   .. code-block:: python
   
      from pyrootmancer.scripts.features.root_length import RootLengthCalculator

      img_dir = '/path/to/images'
      model_path = '/path/to/model.h5'
      custom_objects = {'f1': f1, 'iou': iou}
      root_calculator = RootLengthCalculator(img_dir, model_path, custom_objects)

      img = cv2.imread('/path/to/image.jpg')
      predicted_mask = root_calculator.predict_image(img, 256)
      root_calculator.analyze_image('example_image.jpg', img, predicted_mask)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.root_length.RootLengthCalculator.calculate_length(skeleton, summary)

   Example usage:
   
   .. code-block:: python
   
      from pyrootmancer.scripts.features.root_length import RootLengthCalculator

      img_dir = '/path/to/images'
      model_path = '/path/to/model.h5'
      custom_objects = {'f1': f1, 'iou': iou}
      root_calculator = RootLengthCalculator(img_dir, model_path, custom_objects)

      skeleton = skeletonize(predicted_mask > 0.5)
      summary = summarize(Skeleton(skeleton))
      length = root_calculator.calculate_length(skeleton, summary)

.. rst:directive:: py:function:: pyrootmancer.scripts.features.root_length.RootLengthCalculator.save_results(csv_filename)

   Example usage:
   
   .. code-block:: python
   
      from pyrootmancer.scripts.features.root_length import RootLengthCalculator

      img_dir = '/path/to/images'
      model_path = '/path/to/model.h5'
      custom_objects = {'f1': f1, 'iou': iou}
      root_calculator = RootLengthCalculator(img_dir, model_path, custom_objects)

      csv_filename = '/path/to/save/results.csv'
      root_calculator.save_results(csv_filename)

