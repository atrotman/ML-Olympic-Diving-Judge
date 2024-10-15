import os
import json
import csv

def prepare_labels(dataset_folder, json_file, output_csv_path):
    """
    Prepares a CSV file containing labels for the dataset.

    Parameters:
    dataset_folder: Path to folder with train, validation, and test splits
    json_file: Path to the JSON file with dive labels
    output_csv_path: Path to the output CSV file

    Returns:
    None
    """
    # Load JSON file containing dive labels
    with open(json_file, 'r') as f:
        label_data = json.load(f)

    # Create output CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['frame_path', 'dive_type', 'position', 'difficulty', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop through train, validation, and test folders
        for split in ['train', 'validation', 'test']:
            split_folder = os.path.join(dataset_folder, split)
            video_folders = [f for f in os.listdir(split_folder) if os.path.isdir(os.path.join(split_folder, f))]

            for video_folder in video_folders:
                video_folder_name = os.path.basename(video_folder)

                # Find the corresponding dive label information
                video_metadata = next(
                    (item for item in label_data if item["video"].startswith(video_folder_name)),
                    None
                )

                if not video_metadata:
                    print(f"No label found for video folder: {video_folder_name}")
                    continue

                # Extract frame paths and write labels
                video_path = os.path.join(split_folder, video_folder)
                frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]

                for frame in frames:
                    frame_path = os.path.join(split, video_folder, frame)
                    writer.writerow({
                        'frame_path': frame_path,
                        'dive_type': video_metadata.get('dive_type', 'Unknown'),
                        'position': video_metadata.get('position', 'Unknown'),
                        'difficulty': video_metadata.get('difficulty', 0.0),
                        'score': video_metadata.get('score', 0.0)
                    })

if __name__ == "__main__":
    dataset_folder = "Model building/CNN/Dataset splits"
    json_file = "Shortened videos/Frames and labeling/labeled_results.json"
    output_csv_path = "Model building/CNN/CNN labeling/cnn_dive_labels.csv" 

    prepare_labels(dataset_folder, json_file, output_csv_path)
