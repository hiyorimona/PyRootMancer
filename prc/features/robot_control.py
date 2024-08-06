import cv2
import networkx as nx
import numpy as np
from ot2_env_wrapper import OT2Env
from simple_pid import PID
from skan import Skeleton, summarize
from skimage.morphology import skeletonize
from stable_baselines3 import PPO


class RobotControl:
    def setup_environment(self):
        self.env = OT2Env(render=False)
        self.obs, info = self.env.reset()
        rl_model = PPO.load("./RL_model")
        pid_x = PID(10, 0, 0, setpoint=0)
        pid_y = PID(10, 0, 0, setpoint=0)
        pid_z = PID(10, 0, 0, setpoint=0)
        return self.env, rl_model, self.obs, pid_x, pid_y, pid_z

    @staticmethod
    def process_image(image, model_work, model):
        plate_size_mm = 0.150

        # cropping image
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = image[:-15, 0:-100]
        edges = cv2.Canny(image, 0, 255)
        points = np.argwhere(edges > 0)

        y, x = points.min(axis=0)
        w, h = points.max(axis=0)

        range_y = w - y
        mid_x = (x + h) / 2

        x = int(mid_x - range_y / 2)
        h = int(mid_x + range_y / 2)

        roi = image[y:w, x:h]

        predicted_mask = model_work.predict_image(roi, model)
        kernel = np.ones((6, 6), np.uint8)
        predicted_mask = cv2.dilate(predicted_mask, kernel, iterations=1)
        predicted_mask = cv2.erode(predicted_mask, kernel, iterations=1)
        predicted_mask_uint8 = (predicted_mask > 0.3).astype(np.uint8)
        preds = np.array(predicted_mask_uint8, dtype=np.uint8)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            preds
        )
        sorted_components = sorted(
            range(1, retval),
            key=lambda x: stats[x, cv2.CC_STAT_AREA],
            reverse=True,
        )

        min_area = 200
        min_top = 300
        max_top = 1000
        max_left = 2600

        selected_components = [
            i
            for i in sorted_components
            if min_area <= stats[i, cv2.CC_STAT_AREA]
            and min_top <= stats[i, cv2.CC_STAT_TOP] <= max_top
            and max_left >= stats[i, cv2.CC_STAT_LEFT] <= max_left
        ]

        filtered_image = np.zeros_like(labels, dtype=np.uint8)

        part_boxes = {}
        part_width = range_y / 5

        for idx, component_idx in enumerate(selected_components, start=1):
            x, y, w, h = (
                stats[component_idx, cv2.CC_STAT_LEFT],
                stats[component_idx, cv2.CC_STAT_TOP],
                stats[component_idx, cv2.CC_STAT_WIDTH],
                stats[component_idx, cv2.CC_STAT_HEIGHT],
            )

            part_number = min(x // part_width + 1, 5)

            if part_number not in part_boxes or (
                w * h > part_boxes[part_number][2] * part_boxes[part_number][3]
                and x >= (part_number - 1) * part_width
                and x <= part_number * part_width
            ):
                part_boxes[part_number] = (x, y, w, h)

            filtered_image[labels == component_idx] = part_number

        filtered_image_bin = (filtered_image > 0).astype(np.uint8)
        skeleton = skeletonize(filtered_image_bin)
        skel_data = summarize(Skeleton(skeleton))

        skel_data_sorted = skel_data.sort_values(by='coord-dst-1')
        skel_data_sorted = skel_data_sorted[
            skel_data_sorted['branch-type'] == 1
        ]
        skel_data_unique = skel_data_sorted['skeleton-id'].unique()

        conversion_factor = plate_size_mm / range_y
        return (
            skel_data_sorted,
            skel_data_unique,
            conversion_factor,
            filtered_image,
        )

    @staticmethod
    def control_robot(
        self,
        env,
        rl_model,
        obs,
        skel_data_sorted,
        skel_data_unique,
        conversion_factor,
    ):
        for skeleton in skel_data_unique:
            skel_data_current = skel_data_sorted[
                skel_data_sorted['skeleton-id'] == skeleton
            ]
            G = nx.from_pandas_edgelist(
                skel_data_current,
                source='node-id-src',
                target='node-id-dst',
                edge_attr='branch-distance',
            )

            if nx.number_of_nodes(G) > 0:
                longest_array = max(list(nx.connected_components(G)), key=len)
                root_data = skel_data_current[
                    (
                        skel_data_current['node-id-src'].isin(longest_array)
                        | skel_data_current['node-id-dst'].isin(longest_array)
                    )
                    & (skel_data_current['branch-type'] == 1)
                ]

                maxi = skel_data_current['node-id-dst'].idxmax()
                goal_x = skel_data_current.loc[maxi, 'image-coord-dst-0']
                goal_y = skel_data_current.loc[maxi, 'image-coord-dst-1']
                goal_z = 0.1695

                goal_x = (goal_x * conversion_factor) + 0.10775
                goal_y = (goal_y * conversion_factor) + 0.088
                goal_position_new = [goal_x, goal_y, goal_z]

                env.goal_position = goal_position_new

                max_iterations = 100000
                iteration_count = 0
                while iteration_count < max_iterations:
                    action, _states = rl_model.predict(obs)
                    obs, rewards, terminated, truncated, info = env.step(
                        action
                    )

                    pipette_position_np = np.array(obs[:3])
                    distance = (
                        np.array(goal_position_new) - pipette_position_np
                    )
                    error = np.linalg.norm(distance)

                    if error < 0.001:
                        action = np.array([0, 0, 0, 1])
                        obs, rewards, terminated, truncated, info = env.step(
                            action
                        )

                        for i in range(100):
                            obs, reward, done, truncated, info = env.step(
                                [0, 0, 0]
                            )

                        break
                    iteration_count += 1
