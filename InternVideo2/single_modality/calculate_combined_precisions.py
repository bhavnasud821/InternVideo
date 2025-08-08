import ast
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score

def read_video_data(file_path):
    """
    Reads a file with video data and creates a map from video_id to (probabilities, true_label).

    The file format is expected to be:
    video_id segment_idx probabilities true_label

    Args:
        file_path (str): The path to the input file.

    Returns:
        dict: A dictionary where keys are video_ids (str) and values are
              lists of tuples. Each tuple contains (probabilities_list, true_label_str)
              for a segment of that video.
    """
    video_data_map = {}

    with open(file_path, "r") as f:
        # Skip the header line
        f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue # Skip empty lines

            # Split the line by the last space to separate the class ID
            parts = line.rsplit(' ', 1)
            class_id_str = parts[1].strip()
            remaining_line = parts[0]

            # Find the index of the first '[' to separate file path from probabilities
            prob_start_index = remaining_line.find('[')
            if prob_start_index == -1:
                return None, None, None # Invalid format

            file_path = remaining_line[:prob_start_index].strip()
            probabilities_str = remaining_line[prob_start_index:].strip()

            # Remove the leading '0 ' if it exists after the file path
            # This assumes the '0' is always there before the probabilities array
            if file_path.endswith(' 0'):
                file_path = file_path[:-2]

            # Parse the probabilities list
            # Remove brackets and split by comma
            probabilities_list_str = probabilities_str.strip('[]')
            probabilities = [float(p.strip()) for p in probabilities_list_str.split(',')]

            class_id = int(class_id_str)

            video_data_map[file_path] = (probabilities, class_id)

    return video_data_map

def calculate_combined_precision_recall(thresholds_model_A, thresholds_model_B, model_A_data, model_B_data):
    C = 4 # num classes
    category_names = [
        'climbing',
        'actively taking objects',
        'falling'
    ]
    video_ids = model_A_data.keys()
    all_targets = []
    model_A_probs = []
    model_B_probs = []
    for video_id in video_ids:
        if video_id not in model_B_data:
            print("video id ", video_id, " missing from model B")
            continue
        true_class_id = model_A_data[video_id][1]
        targets = np.zeros(C, dtype=np.float32)
        if true_class_id != -1:
            targets[true_class_id] = 1.0
        all_targets.append(targets)
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
thresholds_model_A = [0.7, 0.7, 0.7]
thresholds_model_B = [0.0, 0.0, 0.0]

# Read data for both models
print("Reading data for Model A...")
model_A_data = read_video_data("/home/bhavna/InternVideo/InternVideo2/single_modality/scripts/finetuning/linear_probing/full_tuning_S_model_4_frames_18_yolo_crops_only_translation/0_preds_all_data.txt")
print("Reading data for Model B...")
model_B_data = read_video_data("/home/bhavna/InternVideo/InternVideo2/single_modality/scripts/finetuning/linear_probing/full_tuning_B_model_4_frames_18_yolo_crops_only_translation/0_preds_all_data.txt")
calculate_combined_precision_recall(thresholds_model_A, thresholds_model_B, model_A_data, model_B_data)