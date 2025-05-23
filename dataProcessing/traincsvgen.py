import mediapipe as mp
import cv2
import os
import csv
from tqdm import tqdm  # progress bar module

# mediapipe setup
hand_detector = mp.solutions.hands
hands_instance = hand_detector.Hands(static_image_mode=True, max_num_hands=1)
hand_drawing = mp.solutions.drawing_utils

# folder being converted
dataset_path = "asl_alphabet_test"
output_csv = "test.csv"

# create csv
with open(output_csv, mode='w', newline='') as f:
    # writer that will write to csv file
    writer = csv.writer(f)
    # setup headers
    headers = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
    writer.writerow(headers)

    # go through each asl label folder
    for label in sorted(os.listdir(dataset_path)):
        # get the path to the current label letter
        label_path = os.path.join(dataset_path, label)
        # ignore any file that isn't a directory/folder
        if not os.path.isdir(label_path):
            continue

        # get the list of all image files in the current label folder
        image_files = os.listdir(label_path)
        print(f"[INFO] Processing label '{label}' ({len(image_files)} images)...")

        # now loop through each image in said label, with a progress bar
        for img_file in tqdm(image_files, desc=f"  Processing {label}", unit="img"):
            # get the path to the image
            img_path = os.path.join(label_path, img_file)

            # now we actually process the image
            image = cv2.imread(img_path)
            # skip empty images
            if image is None:
                continue
            # convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # get results
            results = hands_instance.process(image_rgb)

            # now to get the landmark data and store it into the csv file
            if results.multi_hand_landmarks:
                # only 1 hand per image in the dataset
                hand_landmark = results.multi_hand_landmarks[0].landmark
                x = [pt.x for pt in hand_landmark]  # x coordinates
                y = [pt.y for pt in hand_landmark]  # y coordinates
                z = [pt.z for pt in hand_landmark]  # z coordinates

                # add the info into the csv row
                writer.writerow(x + y + z + [label])
