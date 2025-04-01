# TennisFlow

Tennis shot classification and analysis using machine learning and computer vision.

## Overview

TennisFlow is an application that uses computer vision and machine learning techniques to recognize and classify tennis shots from video. The system detects body keypoints using pose estimation and classifies tennis shots (forehand, backhand, serve, volley) using various machine learning models.

## Features

- Pose estimation for tennis players using MediaPipe
- Tennis shot classification using various models:
  - Basic LSTM classifier
  - Weighted LSTM classifier
  - Deep LSTM classifier
  - Static Pose Enhanced LSTM (best performing)
- Video comparison tool to visualize different model predictions
- Real-time processing capabilities

## Project Structure

- `tennisflow/`: Main package
  - `src/`: Core modules
    - `classification/`: Shot classification models
    - `preprocessing/`: Data preprocessing utilities
    - `visualization/`: Visualization tools
  - `scripts/`: Command-line scripts for various tasks

## Recent Improvements

- Enhanced video comparison visualization with better readability
- Support for portrait-oriented videos with automatic rotation
- Improved text overlay with larger font and background for better visibility

## Models

Several models have been evaluated, with the Static Pose Enhanced LSTM (Model 4) showing the best performance, achieving ~97% accuracy on validation data.

## Installation

### Prerequisites
- Python 3.8+
- Pip
- OpenCV
- TensorFlow 2.6+

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/username/TennisFlow.git
   cd TennisFlow
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Analysis Mode

To analyze a tennis video:

```
python tennisflow/main.py analyze path/to/video.mp4 --output-dir results
```

Options:
- `--config`: Path to custom configuration file (default: config/default_config.json)
- `--output-dir`: Directory to save results (default: timestamped directory in output/)
- `--no-visualization`: Skip creating visualization video

### Data Preparation

To prepare training data from annotated videos:

```
python tennisflow/main.py prepare-data --videos-dir videos/ --annotations-dir annotations/ --output-dir prepared_data/
```

Options:
- `--config`: Path to custom data preparation config (default: config/data_preparation_config.json)

### Training Mode

To train the shot classifier on prepared data:

```
python tennisflow/main.py train --data-dir prepared_data/ --output-model models/my_classifier.h5
```

Options:
- `--config`: Path to custom configuration file (default: config/default_config.json)

## Tennis Shot Recognition Integration

TennisFlow includes integration with the [tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition) repository, which provides a valuable dataset and techniques for tennis shot classification.

### Importing External Data

To import data from the tennis_shot_recognition repository:

```
python tennisflow/scripts/import_keller_data.py --input-dir path/to/tennis_shot_recognition/shots --output-dir prepared_data
```

### Annotation Tool

TennisFlow includes an enhanced annotation tool based on the tennis_shot_recognition approach:

```
python tennisflow/scripts/annotate_video.py path/to/video.mp4 --output annotation.csv
```

Controls:
- F: Annotate forehand
- B: Annotate backhand
- S: Annotate serve
- V: Annotate volley
- N: Annotate neutral
- Space: Pause/resume
- Left/Right arrows: Navigate frames
- [/]: Adjust playback speed

### Extracting Shot Sequences

To extract shot sequences from an annotated video:

```
python tennisflow/scripts/extract_shot_sequences.py path/to/video.mp4 annotation.csv output_dir --show
```

### Preparing Combined Training Data

To prepare training data from multiple sources:

```
python tennisflow/scripts/prepare_training_data.py --input-dirs dir1 dir2 --output-dir combined_data
```

### Training the RNN Classifier

Once you have prepared your training data, you can train the RNN classifier:

```
python tennisflow/scripts/train_rnn_classifier.py --data-dir prepared_data/ --output-model models/rnn_classifier.h5
```

Options:
- `--config`: Path to custom model configuration (optional)
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate for optimizer
- `--log-level`: Logging level (debug, info, warning, error)

### Evaluating the RNN Classifier

To evaluate the trained model on test data:

```
python tennisflow/scripts/evaluate_rnn_classifier.py --model models/rnn_classifier.h5 --test-data test_data.npz --output-dir evaluation_results
```

This will generate:
- Confusion matrices
- Precision, recall, and F1 score graphs
- Classification reports
- Detailed metrics in JSON format

## TenniSet Integration for Improved Serve Classification

TennisFlow now integrates with the [TenniSet](https://github.com/HaydenFaulkner/Tennis) dataset to improve serve classification. This dataset provides professionally annotated tennis videos with detailed serve events.

### Processing TenniSet Data

To extract serve keypoints from TenniSet videos:

```
python tennisflow/scripts/process_tenniset_data.py --input-dir path/to/tenniset/data --output-dir prepared_data/serves --filter-type savgol
```

Options:
- `--window-size`: Number of frames to extract per sequence (default: 30)
- `--max-serves`: Maximum number of serves to process per video
- `--filter-type`: Temporal filter to apply (`savgol`, `kalman`, or `none`)

### Combining Datasets with TenniSet Serves

To combine existing data with TenniSet serve data for a more balanced dataset:

```
python tennisflow/scripts/prepare_combined_training_data.py --input-dirs prepared_data/train prepared_data/train2 --serve-data-dir prepared_data/serves --output-dir prepared_data/combined --balance-classes
```

Options:
- `--validation-split`: Fraction of data to use for validation (default: 0.2)
- `--balance-classes`: Balance classes in the training set
- `--max-samples-per-class`: Maximum number of samples per class

### Training with Advanced Class Handling

Train an improved classifier with class weighting and data augmentation:

```
python tennisflow/scripts/train_combined_rnn_classifier.py --data-dir prepared_data/combined --output-model models/improved_classifier.h5 --use-class-weights --use-augmentation --bidirectional
```

Options:
- `--use-class-weights`: Use class weights to handle class imbalance
- `--use-augmentation`: Use data augmentation for underrepresented classes
- `--bidirectional`: Use bidirectional LSTM layers for improved sequence learning
- `--lstm-units`: Number of units in LSTM layers
- `--dropout-rate`: Dropout rate for regularization

## Model 4 - Enhanced Tennis Shot Classification with Static Poses

TennisFlow introduces Model 4, an advanced classification approach that leverages static pose references to improve shot classification accuracy. This model extends the previous architecture by incorporating similarity features derived from comparing each frame to canonical static poses for each shot type.

### Key Features of Model 4

- **Static Pose Integration**: Uses high-quality, normalized reference poses for each shot type
- **Similarity Feature Extraction**: Computes frame-by-frame similarity to canonical poses
- **Dual-Stream Architecture**: Processes both temporal sequence data and pose similarity features
- **Enhanced Ready Position Recognition**: Significantly improves neutral/ready position detection
- **Parameter Efficiency**: Maintains a reasonable model size while improving accuracy

### Training Model 4

To train Model 4 with static pose integration:

```
python tennisflow/scripts/train_model4_with_static_poses.py --data-dir prepared_data/combined --static-poses-dir prepared_data/static/poses --output-model models/model4_classifier.h5 --use-class-weights
```

Required parameters:
- `--data-dir`: Directory containing the prepared training data
- `--static-poses-dir`: Directory containing static pose JSON files
- `--output-model`: Path to save the trained model

Optional parameters:
- `--use-class-weights`: Use class weights to handle class imbalance
- `--use-augmentation`: Apply data augmentation techniques
- `--lstm-units`: Number of units in LSTM layers (default: 96)
- `--dropout-rate`: Dropout rate for regularization (default: 0.3)
- `--bidirectional`: Use bidirectional LSTM (default: unidirectional)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Maximum number of training epochs (default: 100)
- `--patience`: Early stopping patience (default: 15)
- `--comparison-report`: Path to the model comparison report to update

### Evaluating Model 4

The training script automatically evaluates the model on the validation set and updates the model comparison report with new metrics. The evaluation generates:

- Confusion matrix visualization
- Detailed classification report (precision, recall, F1-score)
- Per-class metrics and overall accuracy
- Model training history plots

### Model Comparison Results

Model 4 with static pose enhancement builds upon the success of Model 3, with particular improvements in the following areas:
- Better discrimination between neutral/ready position and the beginning of shot sequences
- More accurate classification of complex forehand patterns
- Improved robustness to variations in player style and camera angle

## Configuration

TennisFlow uses JSON configuration files to control pipeline behavior:

- `config/default_config.json`: Main pipeline configuration
- `config/data_preparation_config.json`: Data preparation settings

## Project Structure

```
tennisflow/
├── config/                    # Configuration files
├── src/                       # Source code
│   ├── classification/        # Shot classification 
│   ├── kinematics/            # Biomechanical analysis
│   ├── pipeline/              # Pipeline coordination
│   ├── pose_estimation/       # Pose detection and processing
│   ├── reporting/             # Report generation
│   ├── scripts/               # Utility scripts
│   ├── smoothing/             # Temporal filtering
│   └── visualization/         # Video visualization
├── models/                    # Trained models
├── main.py                    # Application entry point
└── requirements.txt           # Dependencies
```

## Annotation Format

TennisFlow supports two annotation formats:

### CSV Format
```
timestamp,shot_type
1.25,forehand
2.75,backhand
5.5,serve
```

### JSON Format
```json
{
  "shots": [
    {"timestamp": 1.25, "type": "forehand"},
    {"timestamp": 2.75, "type": "backhand"},
    {"timestamp": 5.5, "type": "serve"}
  ]
}
```

## Dataset Sources

TennisFlow can integrate data from multiple sources to improve classification accuracy:

1. **tennis_shot_recognition**: Provides basic shot classification data
2. **TenniSet**: Professionally annotated tennis videos with detailed serve events
3. **Custom annotations**: Using the built-in annotation tool

## Advanced Classification Features

The enhanced RNN classifier includes:

- **Class weighting**: Addresses class imbalance issues
- **Data augmentation**: Generates synthetic examples for underrepresented classes like serves
- **Bidirectional LSTM**: Captures dependencies in both temporal directions
- **Advanced metrics**: Detailed per-class performance analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 