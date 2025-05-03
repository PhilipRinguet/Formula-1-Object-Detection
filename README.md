# F1 Real-Time Video Object Detection

## Abstract
Formula 1 (F1) has experienced a surge in global popularity in recent years. However, following a race in real-time can be difficult due to its technical complexity, high-speed nature, and the simultaneous presence of up to twenty drivers on track.

This project develops a real-time object detection system capable of identifying both F1 teams and individual drivers in live race footage. The model leverages deep learning techniques to enhance viewer engagement by providing automated, real-time driver recognition, particularly in challenging conditions such as low visibility, high-speed action, and varying camera perspectives.

## Project Structure
```
├── configs/              # Configuration files for models and training
│   ├── model_configs/    # Model-specific configurations
│   └── training_configs/ # Training pipeline configurations
├── data/                 # Data directory
│   ├── external/        
│   ├── interim/         
│   ├── processed/       
│   └── raw/            
├── models/              # Trained model files
├── notebooks/          
├── src/                 # Source code
│   ├── data/           # Data processing scripts
│   ├── models/         # Model definitions
│   ├── training/       # Training scripts
│   └── visualization/  # Visualization utilities
└── tests/              # Test files
```

## Key Features
- Three-stage transfer learning approach:
  1. F1 Car Detection (base model)
  2. F1 Team Classification
  3. F1 Driver Classification
- Real-time processing capabilities (60+ FPS)
- Support for various race conditions (day/night, weather variations)
- Multiple camera angle support

## Dataset
- Custom dataset built from 3 hours and 10 minutes of 2024 F1 season footage
- 839 annotated images capturing various race conditions
- Annotations include F1 cars, teams, and drivers
- Data augmentation techniques applied to handle class imbalance

## Model Architecture
- Based on YOLOv11s architecture
- Optimized for real-time performance
- Three-stage transfer learning approach
- Custom loss functions and attention mechanisms

## Performance Metrics
### F1 Car Detection
- Precision: 98%
- Recall: 95%
- mAP@50: 98%
- mAP@50-95: 80%

### F1 Team Detection
- Precision: 81%
- Recall: 81%
- mAP@50: 84%
- mAP@50-95: 69%

### F1 Driver Detection
- Variable performance across drivers
- Best performing drivers achieve >70% precision and recall
- Real-time inference at 85.3 FPS
- Per-frame latency: 11.7ms

## Installation & Setup

1. Create a virtual environment:
```bash
python -m venv f1env
source f1env/bin/activate  # Linux/Mac
f1env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your credentials:
# - ROBOFLOW_API_KEY: Get from https://app.roboflow.com
# - DAGSHUB_USERNAME and DAGSHUB_TOKEN (optional): Get from https://dagshub.com/settings/tokens
```

## Usage
1. Dataset Preparation:
```python
python src/data/setup_dataset.py
```

2. Model Training:
```python
python main.py
```

3. Real-time Inference:
```python
# Example code for video inference will be added
```

## Future Work
- Increase dataset size, especially for underrepresented drivers
- Implement custom loss functions with class-weighted penalties
- Optimize video processing pipeline for reduced latency
- Explore model quantization techniques

## Contact
Please contact philipringuet@gmail.com for any questions or issues.

## Acknowledgments
- Dataset: F1 TV Pro
- Base model: YOLOv11 by Ultralytics
- Annotation platform: Roboflow

## License
MIT License