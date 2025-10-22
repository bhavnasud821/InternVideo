import ast
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score

def read_video_data(file_path, num_classes, multilabel=False):
    """
    Reads a file with video data and creates a map from video_id to (probabilities, true_label).

    The file format is expected to be one of the following:
    video_id segment_idx probabilities [multilabel_true_label]
    video_id segment_idx probabilities single_true_label

    Args:
        file_path (str): The path to the input file.
        num_classes (int): The total number of classes.
        multilabel (bool): Flag indicating if the labels are multilabel (list format).

    Returns:
        dict: A dictionary where keys are video_ids (str) and values are
              lists of tuples. Each tuple contains (probabilities_list, true_label_np_array).
              NOTE: The original function body was structured to return a dict 
              where the value is (probabilities, target) for a single segment, 
              implying the map is from (video_id + segment_idx) to data, or that 
              the file only contains one segment per video. 
              The *original docstring* suggests returning a list of tuples 
              per video_id, but the *original code* overwrites the key, 
              so this implementation will follow the original code's final 
              behavior: **map from video_id to the last segment's data.**
    """
    video_data_map = {}

    with open(file_path, "r") as f:
        # NOTE: The original code skipped the header line. Keeping this behavior.
        try:
            f.readline()
        except:
            # Handle case of empty file or no header
            pass 

        for line in f:
            line = line.strip()
            if not line:
                continue # Skip empty lines

            # 1. Find the start and end of the probabilities list
            prob_start_index = line.find('[')
            prob_end_index = line.find(']')
            
            # The line must contain at least one bracket for probabilities
            if prob_start_index == -1 or prob_end_index == -1:
                print(f"Skipping line due to invalid probability list format: {line}")
                continue 

            # The probabilities list is between the first '[' and first ']'
            probabilities_str = line[prob_start_index : prob_end_index + 1].strip()
            
            # 2. Extract parts before and after the probabilities list
            prefix = line[:prob_start_index].strip()
            suffix = line[prob_end_index + 1:].strip()

            # 3. Parse prefix (File Path and Segment Index)
            # Prefix is expected to be: /path/to/video.mp4 0
            # Split the prefix by the last space to separate segment index
            prefix_parts = prefix.rsplit(' ', 1)
            
            if len(prefix_parts) < 2:
                # Assuming no segment index, just the path (less likely given example)
                file_path = prefix_parts[0]
            else:
                # The file path is the part before the last space in the prefix
                file_path = prefix_parts[0] 
                # segment_idx_str = prefix_parts[1] # Segment index is available but unused in the original function

            # 4. Parse suffix (True Label)
            class_id_str = suffix

            # 5. Parse the probabilities list
            # Remove brackets and split by comma
            probabilities_list_str = probabilities_str.strip('[]')
            try:
                probabilities = [float(p.strip()) for p in probabilities_list_str.split(',') if p.strip()]
            except ValueError:
                print(f"Skipping line due to invalid probability values: {line}")
                continue

            # 6. Parse the class label and convert to target array
            if multilabel:
                # Multilabel format: e.g., '[0 0 0]' or '0 0 0'
                # Remove brackets if present, then use np.fromstring
                clean_class_id_str = class_id_str.strip('[]')
                try:
                    target = np.fromstring(clean_class_id_str, dtype=int, sep=' ')
                except ValueError:
                    print(f"Skipping line due to invalid multilabel format: {line}")
                    continue
            else:
                # Single label format: e.g., '3'
                try:
                    class_id = int(class_id_str)
                except ValueError:
                    print(f"Skipping line due to invalid single-label ID: {line}")
                    continue
                    
                target = np.zeros(num_classes, dtype=np.float32)
                if class_id != -1 and 0 <= class_id < num_classes:
                    target[class_id] = 1.0

            # 7. Update the map (overwrites previous segment data for the same file_path)
            # The original code's final output was a map from video_id to (probabilities, target)
            video_data_map[file_path] = (probabilities, target)

    return video_data_map

def calculate_combined_precision_recall(thresholds_model_A, thresholds_model_B, model_A_data, model_B_data, category_names):
    C = len(category_names) + 1
    video_ids = model_A_data.keys()
    all_targets = []
    model_A_probs = []
    model_B_probs = []
    for video_id in video_ids:
        if video_id not in model_B_data:
            print("video id ", video_id, " missing from model B")
            continue
        target = model_A_data[video_id][1]
        all_targets.append(target)
        model_A_probs.append(model_A_data[video_id][0])
        model_B_probs.append(model_B_data[video_id][0])
    
    all_targets = np.array(all_targets)
    model_A_probs = np.array(model_A_probs)
    model_B_probs = np.array(model_B_probs)

    for i in range(C - 1):
        # Convert probabilities to binary predictions based on the threshold
        binary_predictions = ((model_A_probs[:, i] >= thresholds_model_A[i]) & (model_B_probs[:, i] >= thresholds_model_B[i])).astype(int)

        # Calculate precision and recall for the current class
        # Handle cases where there are no true positives or predicted positives to avoid warnings/errors
        try:
            class_precision = precision_score(all_targets[:, i], binary_predictions, zero_division=0)
        except ValueError: # Catches cases where there are no positive samples in true or pred
            class_precision = 0.0 # Or np.nan, depending on desired behavior

        try:
            class_recall = recall_score(all_targets[:, i], binary_predictions, zero_division=0)
        except ValueError:
            class_recall = 0.0

        print(f"Class '{category_names[i]}': Precision = {class_precision:.4f}, Recall = {class_recall:.4f}")


# Define sets of thresholds for each model
thresholds_model_A = [0.5, 0.5, 0.2]
thresholds_model_B = [0.5, 0.5, 0.2]
model_A_file_path = "/home/bhavna/InternVideo/InternVideo2/single_modality/scripts/finetuning/linear_probing/full_tuning_S_model_3_frames_37_yolo_crops_multilabel_v2/internal_0_preds.txt"
model_B_file_path = "/home/bhavna/InternVideo/InternVideo2/single_modality/scripts/finetuning/linear_probing/full_tuning_S_model_3_frames_37_yolo_crops_multilabel_v2/internal_0_preds.txt"
multilabel = True
category_names = [
    'climbing',
    'falling',
    'assault'
]
num_classes = len(category_names)
# Read data for both models
print("Reading data for Model A...")
model_A_data = read_video_data(model_A_file_path, num_classes, multilabel)
print("Reading data for Model B...")
model_B_data = read_video_data(model_B_file_path, num_classes, multilabel)
calculate_combined_precision_recall(thresholds_model_A, thresholds_model_B, model_A_data, model_B_data, category_names)