import pandas as pd
import torch
from torch.utils.data import Dataset

class SensorDataset(Dataset):
    """
    PyTorch Dataset for sensor data stored in CSV files.
   
    Each CSV is expected to have a 'labels' column along with sensor feature columns.
    The features are loaded and converted into a tensor of shape (1, num_features),
    where 1 acts as the channel dimension.
    """
    def __init__(self, csv_path, feature_cols=None, label_col='labels', transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file.
            feature_cols (list, optional): List of column names to use as features.
                                           If None, all columns except label_col are used.
            label_col (str): Name of the label column. Default is 'labels'.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.df = pd.read_csv(csv_path)
        self.label_col = label_col
        if feature_cols is None:
            # Automatically use all columns except the label column.
            self.feature_cols = [col for col in self.df.columns if col != label_col]
        else:
            self.feature_cols = feature_cols
        self.num_classes = len(self.df[self.label_col].unique())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row corresponding to the idx.
        row = self.df.iloc[idx]
        # Extract features as a numpy array (float32)
        features = row[self.feature_cols].values.astype('float32')
        # Convert to tensor and reshape to (1, num_features).
        # Here 1 acts as the channel dimension.
        features = torch.tensor(features).unsqueeze(0)
        label = int(row[self.label_col])
        if self.transform:
            features = self.transform(features)
        return features, label

if __name__ == "__main__":
    # Example usage with your CSV paths:
    source_train_path = "/content/drive/MyDrive/FAS2AR/data/Source_train.csv"
    source_test_path  = "/content/drive/MyDrive/FAS2AR/data/Source_test.csv"
    target_train_path = "/content/drive/MyDrive/FAS2AR/data/Target_train.csv"
    target_test_path  = "/content/drive/MyDrive/FAS2AR/data/Target_test.csv"

    # Load the source training data
    source_train = pd.read_csv(source_train_path)
    FEATURES_dset = [col for col in source_train.columns if col != 'labels']
    print('Number of labels:', len(source_train['labels'].unique()))
    print('Feature columns:', FEATURES_dset)

    # Create an instance of the dataset
    dataset = SensorDataset(source_train_path, feature_cols=FEATURES_dset, label_col='labels')
    print("Number of samples in Source_train:", len(dataset))
   
    # Check the shape of a sample's features and its label
    sample_features, sample_label = dataset[0]
    print("Sample features shape:", sample_features.shape)  # Expected: (1, num_features)
    print("Sample label:", sample_label)