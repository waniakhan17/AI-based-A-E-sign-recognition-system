import cv2
import mediapipe as mp
import csv
import os

# Path to the dataset folder containing subfolders A, B, C, D, E
dataset_path = "dataset"

# Output CSV file where all extracted landmarks will be stored
output_csv = "landmarks_dataset.csv"

# Each letter has its own folder inside the dataset
letters = ["A", "B", "C", "D", "E"]

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,        # Use static mode for images (not video)
    max_num_hands=1,               # Detect only one hand per image
    min_detection_confidence=0.5   # Minimum confidence for detection
)

print("Starting landmark extraction...")

# Open the CSV file only once for writing
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)

    # Create header row: label, x0, y0, z0, x1, y1, z1 ... x20, y20, z20
    header = ["label"]
    for i in range(21):  
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

    # Loop over each letter folder
    for letter in letters:
        folder_path = os.path.join(dataset_path, letter)

        # If folder doesn't exist, skip it
        if not os.path.isdir(folder_path):
            print("Folder not found:", folder_path)
            continue

        print(f"Processing letter {letter} ...")

        # Loop through all images inside the letter folder
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)

            # Read image using OpenCV
            img = cv2.imread(img_path)
            if img is None:  # Skip unreadable/corrupted images
                print("Skipping unreadable image:", img_path)
                continue

            # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process image to detect hand and extract landmarks
            results = hands.process(img_rgb)

            # If at least one hand is detected
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Store (x, y, z) of all 21 landmarks in a list
                landmarks_list = []
                for lm in hand_landmarks.landmark:
                    landmarks_list.extend([lm.x, lm.y, lm.z])

                # Write one row per image: [letter, 63 landmark values]
                writer.writerow([letter] + landmarks_list)

print("Dataset created successfully!")
