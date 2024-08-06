
Functions for DataFrame Creation and Image Coordinates Retrieval
================================================================

This module defines functions for creating and sorting a DataFrame based on image coordinates, as well as retrieving scaled and converted image coordinates.

create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)
---------------------------------------------------------------

**Description:** Creates a DataFrame from image coordinates and sorts it based on specified criteria.

Parameters:
-----------
image_coord_dst_0 : list
    List of Y-axis coordinates of image points.
image_coord_dst_1 : list
    List of X-axis coordinates of image points.

Returns:
--------
pandas.DataFrame
    A sorted DataFrame containing the top 5 image coordinates.

Example usage:

.. code-block:: python

   from pyrootmancer.scripts.features.feature_scaling import create_and_sort_dataframe

   image_coord_dst_0 = [100, 200, 300, 400, 500]
   image_coord_dst_1 = [50, 150, 250, 350, 450]

   sorted_df = create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)
   print(sorted_df)

get_image_coordinates(df, num)
-------------------------------

**Description:** Retrieves and scales image coordinates from a DataFrame.

Parameters:
-----------
df : pandas.DataFrame
    DataFrame containing image coordinates.
num : int
    Index of the row in the DataFrame to retrieve coordinates from.

Returns:
--------
numpy.ndarray
    Robot position coordinates based on scaled image coordinates.

Example usage:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from pyrootmancer.scripts.features.feature_scaling import get_image_coordinates

   df = pd.DataFrame({'X': [50, 150, 250], 'Y': [100, 200, 300]})
   num = 1

   robot_position = get_image_coordinates(df, num)
   print(f"Robot position: {robot_position}")

Troubleshooting
---------------

**Issue:** DataFrame not sorted correctly.

**Solution:** Verify that `image_coord_dst_0` and `image_coord_dst_1` contain valid numerical values. Ensure that the DataFrame is sorted correctly using `create_and_sort_dataframe`.

.. code-block:: python

   from pyrootmancer.scripts.features.feature_scaling import create_and_sort_dataframe

   image_coord_dst_0 = [100, 200, 300, 400, 500]
   image_coord_dst_1 = [50, 150, 250, 350, 450]

   try:
       sorted_df = create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)
       if sorted_df is None:
           print("Error: DataFrame not sorted correctly.")
   except Exception as e:
       print(f"Error: {e}")

---

**Issue:** Error in calculating robot position.

**Solution:** Ensure that `df` is a valid DataFrame containing columns 'X' and 'Y' with numerical values. Verify the correctness of scaling factors and conversion factors used in `get_image_coordinates`.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from pyrootmancer.scripts.features.feature_scaling import get_image_coordinates

   df = pd.DataFrame({'X': [50, 150, 250], 'Y': [100, 200, 300]})
   num = 1

   try:
       robot_position = get_image_coordinates(df, num)
       if robot_position is None:
           print("Error: Calculation of robot position failed.")
   except Exception as e:
       print(f"Error: {e}")

Execution
---------

.. code-block:: python

   if __name__ == "__main__":
       image_coord_dst_0 = [100, 200, 300, 400, 500]
       image_coord_dst_1 = [50, 150, 250, 350, 450]

       sorted_df = create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1)
       print(sorted_df)

       df = pd.DataFrame({'X': [50, 150, 250], 'Y': [100, 200, 300]})
       num = 1

       robot_position = get_image_coordinates(df, num)
       print(f"Robot position: {robot_position}")
