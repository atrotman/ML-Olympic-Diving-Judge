import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Paths to datasets
dataset_folder = "Model building/CNN/Dataset splits"
csv_file = 'Model building/CNN/CNN labeling/cnn_dive_labels.csv'

# CSV check
if not os.path.exists(csv_file):
    print(f"Error: The CSV file at {csv_file} was not found. Please check the path.")
    exit(1)

class DivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
            csv_file: Path to the CSV file with labels
            root_dir: Directory with all image frames from videos
            transform: Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Construct path to the frame image
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        
        # Apply provided transformations
        if self.transform:
            image = self.transform(image)
        
        # Geting dive difficulty and scores
        difficulty = self.annotations.iloc[idx, 3]
        final_score = self.annotations.iloc[idx, 4]

        # Convert difficulty and score to tensors
        difficulty = torch.tensor(difficulty, dtype=torch.float)
        final_score = torch.tensor(final_score, dtype=torch.float)

        return image, (difficulty, final_score)

# Define transformations for data augmentation (training set)
data_transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define transformations for validation/test set (no augmentation)
data_transforms_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization for conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization for conv2
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization
        
        # Output layers for three judge scores
        self.judge_scores = nn.Linear(128, 3)  # Predict scores from three judges (1-10 each)

        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.judge_scores.weight)

    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.dropout(torch.max_pool2d(x, 2))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten tensor
        x = torch.relu(self.fc1(x))
        
        judge_scores = torch.sigmoid(self.judge_scores(x)) * 9 + 1  # Predict scores in range [1, 10]
        judge_scores = torch.clamp(judge_scores, min=1.0, max=10.0)  # Explicitly clip in range [1, 10]
        
        return judge_scores

if __name__ == "__main__":
    # Load training, validation, and test datasets
    train_dataset = DivingDataset(csv_file=csv_file, root_dir=dataset_folder, transform=data_transforms_train)
    val_dataset = DivingDataset(csv_file=csv_file, root_dir=dataset_folder, transform=data_transforms_val_test)
    test_dataset = DivingDataset(csv_file=csv_file, root_dir=dataset_folder, transform=data_transforms_val_test)

    # Split train, validation, and test datasets
    def filter_split(dataset, split_type):
        filtered_indices = dataset.annotations[dataset.annotations['frame_path'].str.startswith(split_type)].index
        return torch.utils.data.Subset(dataset, filtered_indices)

    # Filter subsets for train, validation, and test
    train_subset = filter_split(train_dataset, 'train')
    val_subset = filter_split(val_dataset, 'validation')
    test_subset = filter_split(test_dataset, 'test')

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate model, loss function, and optimizer
    model = CNN()
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10 
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            difficulty, final_labels = labels

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            judge_scores = model(images)

            # Calculate the predicted final score
            judge_sum = torch.sum(judge_scores, dim=1)  # Sum the three judge scores
            predicted_final_score = judge_sum * difficulty

            # Calculate loss for final score
            loss = criterion(predicted_final_score, final_labels)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
            optimizer.step()

            # Loss for the batch
            running_loss += loss.item()

            # Print loss every 10 batches for monitoring
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        # Validation step
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                difficulty, final_labels = labels
                judge_scores = model(images)
                judge_sum = torch.sum(judge_scores, dim=1)  # Sum the three judge scores
                predicted_final_score = judge_sum * difficulty
                loss = criterion(predicted_final_score, final_labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss after Epoch [{epoch+1}/{num_epochs}]: {val_loss:.4f}")

    print("Training complete.")