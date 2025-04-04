#!/usr/bin/env python3

import os
import sys
import argparse

def update_pose_estimator(path_to_model):
    """
    Update the tennisflow/pose_estimation/movenet.py file to use a locally downloaded model.
    
    Args:
        path_to_model: Path to the downloaded MoveNet model
    """
    # Get the absolute path to the model
    path_to_model = os.path.abspath(path_to_model)
    
    # Path to the movenet.py file
    movenet_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "pose_estimation", "movenet.py")
    
    if not os.path.exists(movenet_file):
        print(f"Error: MoveNet estimator file not found at {movenet_file}")
        return False
    
    # Read the file
    with open(movenet_file, 'r') as f:
        content = f.read()
    
    # Find the _load_model method
    if "_load_model" not in content:
        print("Error: Could not find _load_model method in MoveNetPoseEstimator")
        return False
    
    # Check if it's already using a local model
    if "tf.saved_model.load" in content and "signatures['serving_default']" in content:
        print("MoveNetPoseEstimator is already configured to use a local model")
        
        # Update the path
        import re
        pattern = r"self\.model = tf\.saved_model\.load\(['\"](.+)['\"]\)"
        replacement = f"self.model = tf.saved_model.load('{path_to_model}')"
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content == content:
            print("Could not update model path. Try manual modification.")
            return False
        
        content = new_content
    else:
        # Replace the hub.load call with loading from local path
        old_code = """
    def _load_model(self):
        """

        if old_code not in content:
            print("Error: Could not find the exact _load_model method signature")
            return False
        
        # Find where the method ends
        method_start = content.find(old_code)
        
        # Look for the next method or end of class
        next_def = content.find("    def ", method_start + len(old_code))
        if next_def == -1:
            # Look for end of class
            next_def = content.find("class ", method_start + len(old_code))
            if next_def == -1:
                # Use the end of file
                next_def = len(content)
        
        method_body = content[method_start:next_def]
        
        # Create new method body with local model loading
        new_method = f"""
    def _load_model(self):
        \"\"\"Load the MoveNet model from a local path.\"\"\"
        import tensorflow as tf
        
        # Load the model from local path
        self.model = tf.saved_model.load('{path_to_model}')
        self.movenet = self.model.signatures['serving_default']
        """
        
        # Replace the method
        content = content.replace(method_body, new_method)
    
    # Write the updated file
    with open(movenet_file, 'w') as f:
        f.write(content)
    
    print(f"Successfully updated {movenet_file} to use the local model at {path_to_model}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Update MoveNetPoseEstimator to use a locally downloaded model')
    parser.add_argument('model_path', help='Path to the downloaded MoveNet model directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    success = update_pose_estimator(args.model_path)
    
    if success:
        print("\nYou can now use the MoveNetPoseEstimator with your locally downloaded model.")
    else:
        print("\nFailed to update the MoveNetPoseEstimator. Please update it manually:")
        print("1. Open tennisflow/pose_estimation/movenet.py")
        print("2. Find the _load_model method")
        print("3. Replace the hub.load call with:")
        print(f"   self.model = tf.saved_model.load('{args.model_path}')")
        print("   self.movenet = self.model.signatures['serving_default']")

if __name__ == "__main__":
    main() 