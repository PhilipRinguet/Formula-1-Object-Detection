from dataset import F1Dataset
import os

# Driver to team mapping
DRIVER_TEAM_MAPPING = {
    'RedBull_Ver': 'RedBull',
    'RedBull_Per': 'RedBull',
    'Ferrari_Lec': 'Ferrari',
    'Ferrari_Sai': 'Ferrari',
    'Mercedes_Ham': 'Mercedes',
    'Mercedes_Rus': 'Mercedes',
    'McLaren_Nor': 'McLaren',
    'McLaren_Pia': 'McLaren',
    'AstonMartin_Alo': 'AstonMartin',
    'AstonMartin_Str': 'AstonMartin',
    'Alpine_Gas': 'Alpine',
    'Alpine_Oco': 'Alpine',
    'Williams_Alb': 'Williams',
    'Williams_Sar': 'Williams',
    'AlphaTauri_Tsu': 'AlphaTauri',
    'AlphaTauri_Ric': 'AlphaTauri',
    'Alfa_Bot': 'AlfaRomeo',
    'Alfa_Zho': 'AlfaRomeo',
    'Haas_Hul': 'Haas',
    'Haas_Mag': 'Haas'
}

def setup_f1_dataset(base_dir: str):
    """
    Set up the F1 dataset by downloading from Roboflow and creating detection task datasets.
    
    Args:
        base_dir (str): Base directory for dataset
    """
    # Initialize dataset manager
    dataset = F1Dataset(base_dir)
    
    # Get API key from environment variable
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError("Please set the ROBOFLOW_API_KEY environment variable")
    
    # Download dataset from Roboflow
    dataset.download_from_roboflow(
        api_key=api_key,
        workspace="projectwork-ziouu",
        project="f2-fjp23",
        version=9
    )
    
    # Create car detection dataset
    dataset.create_car_detection_labels()
    
    # Create team detection dataset
    dataset.create_team_detection_labels(DRIVER_TEAM_MAPPING)
    
    # Generate YAML configs for all detection tasks
    dataset.generate_yaml_configs(DRIVER_TEAM_MAPPING)
    
    print("Dataset setup complete! Created datasets for:")
    print("1. F1 Car Detection")
    print("2. F1 Team Detection")
    print("3. F1 Driver Detection")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "data", "processed")
    setup_f1_dataset(base_dir)