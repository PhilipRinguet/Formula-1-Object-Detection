import os
import shutil
import random
import numpy as np
import cv2
import albumentations as A
from PIL import Image, ExifTags
from typing import Dict, List, Tuple

class F1Dataset:
    def __init__(self, base_dir: str):
        """
        Initialize F1 dataset manager.
        
        Args:
            base_dir (str): Base directory for dataset
        """
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'train')
        self.val_dir = os.path.join(base_dir, 'val')
        self.test_dir = os.path.join(base_dir, 'test')
        
        # Create directories
        for d in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(os.path.join(d, 'images'), exist_ok=True)
            os.makedirs(os.path.join(d, 'labels'), exist_ok=True)

    def auto_orient_image(self, image_path: str) -> np.ndarray:
        """
        Auto-orient image using EXIF data.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Oriented image
        """
        try:
            img = Image.open(image_path)
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
                    
            exif = dict(img._getexif().items())
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
                
            return np.array(img)
        except (AttributeError, KeyError, IndexError):
            # Return original image if EXIF data not available
            return cv2.imread(image_path)

    def split_dataset(self, 
                     images_dir: str, 
                     labels_dir: str, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15,
                     seed: int = 42) -> None:
        """
        Split dataset into train/val/test sets preserving race segments.
        
        Args:
            images_dir (str): Directory containing images
            labels_dir (str): Directory containing labels
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
            seed (int): Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

        # Group images by race segments
        def get_race_id(filename: str) -> str:
            parts = filename.split('_')
            if len(parts) > 1:
                timestamp = parts[1]
                try:
                    time_value = int(timestamp.replace('min', ''))
                    return f'race_{time_value // 10}'
                except ValueError:
                    return 'unknown'
            return 'unknown'

        # Group files by race
        races = {}
        for img_file in os.listdir(images_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                race_id = get_race_id(img_file)
                if race_id not in races:
                    races[race_id] = []
                races[race_id].append(img_file)

        # Split races
        race_ids = list(races.keys())
        random.shuffle(race_ids)
        
        n_races = len(race_ids)
        train_idx = int(n_races * train_ratio)
        val_idx = int(n_races * (train_ratio + val_ratio))
        
        splits = {
            'train': race_ids[:train_idx],
            'val': race_ids[train_idx:val_idx],
            'test': race_ids[val_idx:]
        }

        # Copy files to respective directories
        for split_name, race_list in splits.items():
            split_img_dir = os.path.join(self.base_dir, split_name, 'images')
            split_label_dir = os.path.join(self.base_dir, split_name, 'labels')
            
            for race_id in race_list:
                for img_file in races[race_id]:
                    # Copy and auto-orient image
                    img_path = os.path.join(images_dir, img_file)
                    oriented_img = self.auto_orient_image(img_path)
                    cv2.imwrite(os.path.join(split_img_dir, img_file), oriented_img)
                    
                    # Copy label if exists
                    label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                    label_path = os.path.join(labels_dir, label_file)
                    if os.path.exists(label_path):
                        shutil.copy2(label_path, os.path.join(split_label_dir, label_file))

    def apply_augmentations(self, 
                          image: np.ndarray,
                          augmentation_config: Dict) -> np.ndarray:
        """
        Apply augmentations to an image.
        
        Args:
            image (np.ndarray): Input image
            augmentation_config (Dict): Augmentation parameters
            
        Returns:
            np.ndarray: Augmented image
        """
        transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=augmentation_config.get('brightness_limit', 0.2),
                contrast_limit=augmentation_config.get('contrast_limit', 0.2),
                p=augmentation_config.get('brightness_contrast_prob', 0.5)
            ),
            A.MotionBlur(
                blur_limit=augmentation_config.get('blur_limit', 7),
                p=augmentation_config.get('motion_blur_prob', 0.3)
            ),
            A.CLAHE(
                clip_limit=augmentation_config.get('clahe_limit', 4.0),
                p=augmentation_config.get('clahe_prob', 0.3)
            ),
            A.RandomRain(
                p=augmentation_config.get('rain_prob', 0.25)
            ),
            A.RandomFog(
                p=augmentation_config.get('fog_prob', 0.25)
            ),
            A.RandomSunFlare(
                flare_roi=augmentation_config.get('flare_roi', (0, 0, 1, 0.5)),
                p=augmentation_config.get('sunflare_prob', 0.3)
            ),
        ])
        
        augmented = transform(image=image)
        return augmented['image']

    def augment_training_set(self, augmentation_config: Dict = None) -> None:
        """
        Apply augmentations to training set images.
        
        Args:
            augmentation_config (Dict): Configuration for augmentations
        """
        if augmentation_config is None:
            augmentation_config = {}
            
        train_img_dir = os.path.join(self.train_dir, 'images')
        train_label_dir = os.path.join(self.train_dir, 'labels')
        
        for img_file in os.listdir(train_img_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(train_img_dir, img_file)
                img = cv2.imread(img_path)
                
                # Apply augmentations
                augmented = self.apply_augmentations(img, augmentation_config)
                
                # Save augmented image
                aug_filename = f"aug_{img_file}"
                cv2.imwrite(os.path.join(train_img_dir, aug_filename), augmented)
                
                # Copy corresponding label file
                label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(train_label_dir, label_file)
                if os.path.exists(label_path):
                    shutil.copy2(
                        label_path,
                        os.path.join(train_label_dir, f"aug_{label_file}")
                    )

    def download_from_roboflow(self, api_key: str, workspace: str, project: str, version: int) -> None:
        """
        Download dataset from Roboflow.
        
        Args:
            api_key (str): Roboflow API key
            workspace (str): Roboflow workspace name
            project (str): Roboflow project name
            version (int): Dataset version
        """
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        dataset = project.version(version).download("yolov11")
        
        # Move downloaded data to our directory structure
        download_dir = os.path.join(os.getcwd(), project)
        self._reorganize_downloaded_data(download_dir)

    def _reorganize_downloaded_data(self, download_dir: str) -> None:
        """
        Reorganize downloaded data into our directory structure.
        
        Args:
            download_dir (str): Directory containing downloaded data
        """
        for split in ['train', 'valid', 'test']:
            src_img_dir = os.path.join(download_dir, split, 'images')
            src_label_dir = os.path.join(download_dir, split, 'labels')
            
            dst_split = 'val' if split == 'valid' else split
            dst_img_dir = os.path.join(self.base_dir, dst_split, 'images')
            dst_label_dir = os.path.join(self.base_dir, dst_split, 'labels')
            
            if os.path.exists(src_img_dir):
                for file in os.listdir(src_img_dir):
                    shutil.copy2(
                        os.path.join(src_img_dir, file),
                        os.path.join(dst_img_dir, file)
                    )
            
            if os.path.exists(src_label_dir):
                for file in os.listdir(src_label_dir):
                    shutil.copy2(
                        os.path.join(src_label_dir, file),
                        os.path.join(dst_label_dir, file)
                    )
        
        # Clean up download directory
        shutil.rmtree(download_dir, ignore_errors=True)

    def create_car_detection_labels(self) -> None:
        """
        Create labels for car detection task (single class - F1 car).
        All driver labels are converted to a single F1 car class (0).
        """
        for split in ['train', 'val', 'test']:
            label_dir = os.path.join(self.base_dir, split, 'labels')
            car_label_dir = os.path.join(self.base_dir, f'{split}_car', 'labels')
            os.makedirs(car_label_dir, exist_ok=True)
            
            # Copy corresponding images
            img_dir = os.path.join(self.base_dir, split, 'images')
            car_img_dir = os.path.join(self.base_dir, f'{split}_car', 'images')
            os.makedirs(car_img_dir, exist_ok=True)
            
            for label_file in os.listdir(label_dir):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    lines = f.readlines()
                
                # Convert all class ids to 0 (F1 car)
                car_labels = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Ensure valid YOLO format
                        car_labels.append(f"0 {' '.join(parts[1:])}\n")
                
                # Save car detection labels
                with open(os.path.join(car_label_dir, label_file), 'w') as f:
                    f.writelines(car_labels)
                
                # Copy corresponding image
                img_file = label_file.replace('.txt', '.jpg')
                if os.path.exists(os.path.join(img_dir, img_file)):
                    shutil.copy2(
                        os.path.join(img_dir, img_file),
                        os.path.join(car_img_dir, img_file)
                    )

    def create_team_detection_labels(self, driver_team_mapping: Dict[str, str]) -> None:
        """
        Create labels for team detection task.
        Maps driver classes to their respective teams.
        
        Args:
            driver_team_mapping (Dict[str, str]): Mapping from driver names to team names
        """
        # Create reverse mapping from class indices to team names
        with open(os.path.join(self.base_dir, 'data.yaml'), 'r') as f:
            import yaml
            data = yaml.safe_load(f)
            class_names = data['names']
        
        # Create team to index mapping
        teams = sorted(set(driver_team_mapping.values()))
        team_to_idx = {team: idx for idx, team in enumerate(teams)}
        
        for split in ['train', 'val', 'test']:
            label_dir = os.path.join(self.base_dir, split, 'labels')
            team_label_dir = os.path.join(self.base_dir, f'{split}_team', 'labels')
            os.makedirs(team_label_dir, exist_ok=True)
            
            # Copy corresponding images
            img_dir = os.path.join(self.base_dir, split, 'images')
            team_img_dir = os.path.join(self.base_dir, f'{split}_team', 'images')
            os.makedirs(team_img_dir, exist_ok=True)
            
            for label_file in os.listdir(label_dir):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    lines = f.readlines()
                
                # Convert driver classes to team classes
                team_labels = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        driver_idx = int(parts[0])
                        driver_name = class_names[driver_idx]
                        team_name = driver_team_mapping[driver_name]
                        team_idx = team_to_idx[team_name]
                        team_labels.append(f"{team_idx} {' '.join(parts[1:])}\n")
                
                # Save team detection labels
                with open(os.path.join(team_label_dir, label_file), 'w') as f:
                    f.writelines(team_labels)
                
                # Copy corresponding image
                img_file = label_file.replace('.txt', '.jpg')
                if os.path.exists(os.path.join(img_dir, img_file)):
                    shutil.copy2(
                        os.path.join(img_dir, img_file),
                        os.path.join(team_img_dir, img_file)
                    )

    def generate_yaml_configs(self, driver_team_mapping: Dict[str, str]) -> None:
        """
        Generate YAML configuration files for all three detection tasks.
        
        Args:
            driver_team_mapping (Dict[str, str]): Mapping from driver names to team names
        """
        import yaml
        
        # Car detection config
        car_config = {
            'train': os.path.join(self.base_dir, 'train_car'),
            'val': os.path.join(self.base_dir, 'val_car'),
            'test': os.path.join(self.base_dir, 'test_car'),
            'nc': 1,
            'names': ['F1_Car']
        }
        
        with open(os.path.join(self.base_dir, 'car_detection.yaml'), 'w') as f:
            yaml.dump(car_config, f)
        
        # Team detection config
        teams = sorted(set(driver_team_mapping.values()))
        team_config = {
            'train': os.path.join(self.base_dir, 'train_team'),
            'val': os.path.join(self.base_dir, 'val_team'),
            'test': os.path.join(self.base_dir, 'test_team'),
            'nc': len(teams),
            'names': teams
        }
        
        with open(os.path.join(self.base_dir, 'team_detection.yaml'), 'w') as f:
            yaml.dump(team_config, f)
        
        # Driver detection config (using original data.yaml as basis)
        with open(os.path.join(self.base_dir, 'data.yaml'), 'r') as f:
            driver_config = yaml.safe_load(f)
        
        with open(os.path.join(self.base_dir, 'driver_detection.yaml'), 'w') as f:
            yaml.dump(driver_config, f)