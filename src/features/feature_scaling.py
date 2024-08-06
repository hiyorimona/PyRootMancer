import numpy as np
import pandas as pd


def create_and_sort_dataframe(image_coord_dst_0: list, image_coord_dst_1: list) -> pd.DataFrame:
    """
    Creates a DataFrame from image coordinates and sorts it based on specific criteria.

    This function creates a DataFrame with columns 'X', 'Y', and 'Z' from the provided image coordinates.
    It then selects the top 5 rows with the largest 'Y' values and sorts them by 'Y' (descending) and 'X' (ascending).

    Parameters:
    -----------
    image_coord_dst_0 : list
        A list of Y-coordinates.
    image_coord_dst_1 : list
        A list of X-coordinates.

    Returns:
    --------
    pd.DataFrame
        A sorted DataFrame containing the root tip coordinates.
    """
    # Create a DataFrame with columns 'X', 'Y', and 'Z'
    df = pd.DataFrame({'X': image_coord_dst_1, 'Y': image_coord_dst_0, 'Z': [0] * len(image_coord_dst_0)})
    # Select the top 5 rows with the largest 'Y' values and sort by 'Y' (descending) and 'X' (ascending)
    sort_x = df.nlargest(5, 'Y').sort_values(by=['Y', 'X'], ascending=[False, True])
    print(f'{len(sort_x)} root tips coordinates found')
    print(sort_x)
    return sort_x


def get_image_coordinates(df: pd.DataFrame, num: int) -> np.ndarray:
    """
    Converts image coordinates to robot coordinates.

    This function takes a DataFrame and a row number, scales the coordinates,
    and converts them to robot coordinates.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing the image coordinates.
    num : int
        The row number to convert.

    Returns:
    --------
    np.ndarray
        The converted robot coordinates.
    """
    # Define the plate position
    plate = np.array([0.10775, 0.088, 0.057])
    # Define scaling factors
    scaling_factors = np.array([1099, 1099]) / np.array([2752, 2731])

    # Scale the coordinates from the DataFrame
    landmark_scaled = df.loc[num, ['X', 'Y']].values * scaling_factors
    print(f'Scaled coordinates of root tip {landmark_scaled}')

    # Define conversion factors
    conversion_factors = np.array([150 / 1099, 151 / 1099])
    # Convert to mm and then to robot position
    root_tip_mm = (landmark_scaled * conversion_factors) / np.array([1100, 1091])
    root_tip_position = np.append(root_tip_mm[::-1], 0)

    # Calculate the final robot position
    root_tip_robot_position = plate + root_tip_position

    return root_tip_robot_position
