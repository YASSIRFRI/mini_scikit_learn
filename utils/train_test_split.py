import random
from collections import defaultdict

def train_test_split(data, test_size=0.25, random_state=None, stratify=None):
    """
    Splits the data into training and testing sets.

    Parameters:
    - data: list or array-like, the dataset to split.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, seed used by the random number generator.
    - stratify: list or array-like, if not None, data is split in a stratified fashion using this as the class labels.

    Returns:
    - train_data: list or array-like, the training set.
    - test_data: list or array-like, the testing set.
    """
    
    if random_state is not None:
        random.seed(random_state)
    
    if stratify is None:
        data_copy = data[:]
        random.shuffle(data_copy)
        
        n_total = len(data)
        n_test = int(n_total * test_size)
        
        test_data = data_copy[:n_test]
        train_data = data_copy[n_test:]
    else:
        stratified_data = defaultdict(list)
        for item, label in zip(data, stratify):
            stratified_data[label].append(item)
        
        train_data = []
        test_data = []
        
        for label, items in stratified_data.items():
            n_total_label = len(items)
            n_test_label = int(n_total_label * test_size)
            
            random.shuffle(items)
            
            test_data.extend(items[:n_test_label])
            train_data.extend(items[n_test_label:])
        
        # Shuffle final train and test data to mix items from different classes
        random.shuffle(train_data)
        random.shuffle(test_data)
    
    return train_data, test_data
