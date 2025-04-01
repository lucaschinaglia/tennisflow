# TennisFlow Scripts

This directory contains utility scripts for running various tasks in the TennisFlow project.

## Available Scripts

### `compare_models_on_video.py`

This script runs multiple classification models on the same video and creates a visualization that displays their predictions side by side for comparison.

#### Usage

```bash
python tennisflow/scripts/compare_models_on_video.py VIDEO_PATH \
  --model1 PATH_TO_MODEL1 \
  --model2 PATH_TO_MODEL2 \
  --model3 PATH_TO_MODEL3 \
  --model4 PATH_TO_MODEL4 \
  --static-poses-dir PATH_TO_STATIC_POSES
```

#### Features:
- Supports portrait and landscape video orientations
- Automatically rotates portrait videos for better visualization
- Enhanced text overlay with improved readability
- Color-coded borders based on shot type
- Creates grid visualization with original video and model predictions

### `evaluate_model.py`

Script to evaluate a trained model against test data.

### `preprocess_data.py`

Script for preprocessing raw data before training models.

### `train_model.py`

Script for training machine learning models for shot classification.

# SSL Fix for MoveNet Model

This directory contains scripts to fix SSL certificate verification issues when downloading and using the MoveNet model from TensorFlow Hub, particularly on macOS.

## The Problem

On macOS (and sometimes other systems), Python may fail to verify SSL certificates when trying to download models from TensorFlow Hub, resulting in errors like:

```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>
```

## The Solution

We provide several scripts to fix this issue:

1. **core_patch_ssl.py** - The most comprehensive solution that works reliably
2. **monkey_patch_tensorflow_hub.py** - A runtime patch for TensorFlow Hub
3. **bypass_ssl.py** - A general SSL verification bypass utility
4. **fix_ssl_issue.py** - A more permanent solution that creates patches 

## Recommended Usage

### Option 1: Run your script with the SSL patcher

The most reliable way to fix SSL issues is to run your script with our SSL patcher:

```bash
# Runs your script with SSL verification disabled
./scripts/core_patch_ssl.py your_script.py
```

### Option 2: Import the bypass module in your script

```python
# Add this at the very beginning of your script
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.bypass_ssl import bypass_ssl
bypass_ssl()

# Rest of your imports and code
import tensorflow_hub as hub
# ...
```

### Option 3: Apply the permanent fix (requires restart)

```bash
# Apply a more permanent fix by patching the pose_estimation module
./scripts/fix_ssl_issue.py
```

## Testing

You can test if your SSL fix is working with:

```bash
./scripts/core_patch_ssl.py ./scripts/test_ssl_fix.py
```

If successful, you should see:

```
✅ SSL fix is working correctly! You can now use MoveNet without SSL issues.
```

## Security Considerations

These fixes disable SSL certificate verification, which reduces security. This is acceptable for local development and research use, but should not be used in production environments.

Only use these fixes on your local machine where security is not a primary concern. 