import os
import shutil
import random

def split_dataset(input_base_folder, output_base_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataset of videos randomly into training, validation, and test sets.

    Parameters:
    input_base_folder: Path to the folder containing extracted video frames
    output_base_folder: Path to the base folder where the training, validation, and test sets will be stored
    train_ratio: Proportion of the dataset to be used for training
    val_ratio: Proportion of the dataset to be used for validation
    test_ratio: Proportion of the dataset to be used for testing
    seed: Random seed for reproducibility

    Returns:
    None
    """
    random.seed(seed)
    
    # Create output folders for train, validation, and test sets
    train_folder = os.path.join(output_base_folder, 'train')
    val_folder = os.path.join(output_base_folder, 'validation')
    test_folder = os.path.join(output_base_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all video folders (each folder contains frames for one video)
    video_folders = [f for f in os.listdir(input_base_folder) if os.path.isdir(os.path.join(input_base_folder, f))]
    random.shuffle(video_folders)

    # Calculate the number of videos for each set
    total_videos = len(video_folders)
    train_count = int(total_videos * train_ratio)
    val_count = int(total_videos * val_ratio)

    # Assign videos to train, validation, and test sets
    train_videos = video_folders[:train_count]
    val_videos = video_folders[train_count:train_count + val_count]
    test_videos = video_folders[train_count + val_count:]

    # Helper function to move/copy video folders to their respective set
    def move_videos(video_list, destination_folder):
        for video_folder in video_list:
            src = os.path.join(input_base_folder, video_folder)
            dst = os.path.join(destination_folder, video_folder)
            shutil.copytree(src, dst)
            print(f"Copied {video_folder} to {destination_folder}")

    # Move video folders to train, validation, and test folders
    move_videos(train_videos, train_folder)
    move_videos(val_videos, val_folder)
    move_videos(test_videos, test_folder)

if __name__ == "__main__":
    input_base_folder = "Shortened videos\Frames and labeling\Extracted frames"
    output_base_folder = "Model building/CNN/Dataset splits" 

    # Split dataset into train, validation, and test sets
    split_dataset(input_base_folder, output_base_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)