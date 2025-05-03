import pytest
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from src.data.dataset import F1Dataset

@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset structure."""
    dataset = F1Dataset(str(tmp_path))
    return dataset

@pytest.fixture
def sample_dataset_files(tmp_path):
    """Create sample dataset files for testing."""
    images_dir = tmp_path / "raw_images"
    labels_dir = tmp_path / "raw_labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    # Create sample race files
    for race_id in range(3):
        for frame in range(2):
            # Create image file
            img_name = f"race_{race_id}_min{frame*10}.jpg"
            img_path = images_dir / img_name
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            # Use PIL to properly save the image
            Image.fromarray(img).save(str(img_path), format='JPEG')
            
            # Create corresponding label file
            label_name = img_name.replace('.jpg', '.txt')
            label_path = labels_dir / label_name
            label_path.write_text("0 0.5 0.5 0.1 0.1")
    
    return str(images_dir), str(labels_dir)

def test_dataset_initialization(temp_dataset, tmp_path):
    """Test dataset directory structure creation."""
    for split in ['train', 'val', 'test']:
        split_dir = tmp_path / split
        assert (split_dir / 'images').exists()
        assert (split_dir / 'labels').exists()

def test_dataset_splitting(temp_dataset, sample_dataset_files):
    """Test dataset splitting functionality."""
    images_dir, labels_dir = sample_dataset_files
    temp_dataset.split_dataset(images_dir, labels_dir)
    
    # Check if files were distributed across splits
    total_files = 0
    for split in ['train', 'val', 'test']:
        split_images = list((Path(temp_dataset.base_dir) / split / 'images').glob('*.jpg'))
        split_labels = list((Path(temp_dataset.base_dir) / split / 'labels').glob('*.txt'))
        assert len(split_images) == len(split_labels)
        total_files += len(split_images)
    
    # Verify all files were distributed
    original_files = len(list(Path(images_dir).glob('*.jpg')))
    assert total_files == original_files

def test_augmentation(temp_dataset, sample_dataset_files):
    """Test image augmentation functionality."""
    images_dir, labels_dir = sample_dataset_files
    
    # Split dataset with fixed seed for reproducibility
    temp_dataset.split_dataset(images_dir, labels_dir, train_ratio=0.8, seed=42)
    
    # Verify we have training images before augmentation
    train_img_dir = Path(temp_dataset.train_dir) / 'images'
    original_train_images = list(train_img_dir.glob('*.jpg'))
    assert len(original_train_images) > 0, "No training images found before augmentation"
    
    # Apply augmentations
    augmentation_config = {
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'blur_limit': 7,
        'rain_prob': 0.5,
        'fog_prob': 0.5
    }
    temp_dataset.augment_training_set(augmentation_config)
    
    # Check if augmented files were created
    train_images = list(train_img_dir.glob('aug_*.jpg'))
    train_labels = list((Path(temp_dataset.train_dir) / 'labels').glob('aug_*.txt'))
    assert len(train_images) > 0, "No augmented images were created"
    assert len(train_images) == len(train_labels), "Mismatch between augmented images and labels"

def test_auto_orient_image(temp_dataset, tmp_path):
    """Test image auto-orientation."""
    # Create a test image
    img_path = tmp_path / "test.jpg"
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Properly save image using PIL
    Image.fromarray(img).save(str(img_path))
    
    # Test orientation
    oriented_img = temp_dataset.auto_orient_image(str(img_path))
    assert isinstance(oriented_img, np.ndarray)
    assert oriented_img.shape == (100, 100, 3)