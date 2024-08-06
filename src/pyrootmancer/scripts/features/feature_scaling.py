import numpy as np
import pandas as pd


def create_and_sort_dataframe(image_coord_dst_0, image_coord_dst_1):
    df = pd.DataFrame({'X': image_coord_dst_1, 'Y': image_coord_dst_0, 'Z': [0] * len(image_coord_dst_0)})
    sort_x = df.nlargest(5, 'Y').sort_values(by=['Y', 'X'], ascending=[False, True])
    print(f'{len(sort_x)} root tips coordinates found')
    print(sort_x)
    return sort_x


def get_image_coordinates(df, num):
    plate = np.array([0.10775, 0.088, 0.057])
    scaling_factors = np.array([1099, 1099]) / np.array([2752, 2731])

    landmark_scaled = df.loc[num, ['X', 'Y']].values * scaling_factors
    # print(f'Scaling factors {scaling_factors}')
    print(f'Scaled coordinates of root tip {landmark_scaled}')

    conversion_factors = np.array([150 / 1099, 151 / 1099])
    root_tip_mm = (landmark_scaled * conversion_factors) / np.array([1100, 1091])
    root_tip_position = np.append(root_tip_mm[::-1], 0)

    root_tip_robot_position = plate + root_tip_position
    # print(f'Goal position: {root_tip_robot_position}')

    return root_tip_robot_position
