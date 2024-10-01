import os
import logging
from typing import List, Tuple, Dict

def get_paths_to_files(dir_path: str) -> Tuple[List[str], List[str]]:
    """
    Recursively get all file paths and file names in a directory, excluding hidden files.

    Args:
        dir_path (str): The path to the directory to search.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - The first list contains the full file paths.
            - The second list contains the file names.

    Example:
        filepaths, filenames = get_paths_to_files("/path/to/dataset")
    """
    filepaths = []
    fnames = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        filepaths.extend(os.path.join(dirpath, f) for f in filenames if not f.startswith('.'))
        fnames.extend([f for f in filenames if not f.startswith('.')])
    return filepaths, fnames


def get_dataset_paths(dataset_dir: str) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Get file paths and names for train, test, and validation sets.

    Args:
        dataset_dir (str): The path to the main dataset directory containing 'train', 'test', and 'val' subdirectories.

    Returns:
        Dict[str, Tuple[List[str], List[str]]]: A dictionary with keys 'train', 'test', and 'val'.
        Each value is a tuple containing two lists:
            - The first list contains the full file paths.
            - The second list contains the file names.

    Example:
        dataset_paths = get_dataset_paths("/path/to/dataset")
        train_paths, train_names = dataset_paths['train']
        test_paths, test_names = dataset_paths['test']
        val_paths, val_names = dataset_paths['val']
    """
    dataset_paths = {}
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Directory not found: {split_dir}")
        
        filepaths = []
        fnames = []
        for dirpath, dirnames, filenames in os.walk(split_dir):
            filepaths.extend(os.path.join(dirpath, f) for f in filenames if not f.startswith('.'))
            fnames.extend([f for f in filenames if not f.startswith('.')])
        
        dataset_paths[split] = (filepaths, fnames)
    
    return dataset_paths


def get_logger(ch_log_level=logging.INFO, fh_log_level=logging.INFO):
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(ch_log_level)
    
    # File Handler
    fh = logging.FileHandler('training.log')
    fh.setLevel(fh_log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger