import numpy as np
import pandas as pd

#chatgpt
# =============================================================================================
# LOAD MASK FILES
# =============================================================================================
# Load ground truth masks (binary masks containing 0 and 1)
# Ground truth represents the actual/correct segmentation masks
liver_gt = np.load('liver_gt.npy')
lungs_gt = np.load('lungs_gt.npy')
heart_gt = np.load('heart_gt.npy')

# Load predicted masks (binary masks containing 0 and 1)
# Predicted masks are the output from your segmentation model
liver_pred = np.load('liver_pred.npy')
lungs_pred = np.load('lungs_pred.npy')
heart_pred = np.load('heart_pred.npy')

# =============================================================================================
# EVALUATION METRIC FUNCTIONS
# =============================================================================================

def dice_coefficient(y_true, y_pred):
    """
    Calculate Dice Coefficient (F1 Score for segmentation)
    
    Dice = 2 * |Intersection| / (|A| + |B|)
    
    Measures the overlap between predicted and ground truth masks.
    Range: 0 (no overlap) to 1 (perfect overlap)
    
    Parameters:
    -----------
    y_true : numpy array
        Ground truth binary mask
    y_pred : numpy array
        Predicted binary mask
    
    Returns:
    --------
    float : Dice coefficient score
    """
    # Flatten 2D/3D arrays into 1D for easier computation
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate intersection (pixels that are 1 in both masks)
    intersection = np.sum(y_true * y_pred)
    
    # Calculate Dice coefficient
    # Adding small epsilon (1e-8) to avoid division by zero
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)
    
    return dice


def iou_score(y_true, y_pred):
    """
    Calculate IoU (Intersection over Union) / Jaccard Index
    
    IoU = |Intersection| / |Union|
    
    Measures the ratio of overlapping area to the total area covered by both masks.
    Range: 0 (no overlap) to 1 (perfect overlap)
    
    Parameters:
    -----------
    y_true : numpy array
        Ground truth binary mask
    y_pred : numpy array
        Predicted binary mask
    
    Returns:
    --------
    float : IoU score
    """
    # Flatten arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate intersection (pixels that are 1 in both masks)
    intersection = np.sum(y_true * y_pred)
    
    # Calculate union (all pixels that are 1 in either mask)
    # Union = Total pixels in both masks - intersection (to avoid double counting)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    
    # Calculate IoU
    # Adding small epsilon (1e-8) to avoid division by zero
    iou = intersection / (union + 1e-8)
    
    return iou


def accuracy(y_true, y_pred):
    """
    Calculate Pixel-wise Accuracy
    
    Accuracy = (Correctly Predicted Pixels) / (Total Pixels)
    
    Measures the percentage of correctly classified pixels (both 0s and 1s).
    Range: 0 (all wrong) to 1 (all correct)
    
    Note: Can be misleading if classes are imbalanced (e.g., background >> foreground)
    
    Parameters:
    -----------
    y_true : numpy array
        Ground truth binary mask
    y_pred : numpy array
        Predicted binary mask
    
    Returns:
    --------
    float : Accuracy score
    """
    # Flatten arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Count pixels where prediction matches ground truth
    correct = np.sum(y_true == y_pred)
    
    # Get total number of pixels
    total = len(y_true)
    
    # Calculate accuracy
    acc = correct / total
    
    return acc


# =============================================================================================
# PREPARE DATA FOR EVALUATION
# =============================================================================================

# List of organ names for display
organs = ['Liver', 'Lungs', 'Heart']

# List of ground truth masks corresponding to each organ
ground_truths = [liver_gt, lungs_gt, heart_gt]

# List of predicted masks corresponding to each organ
predictions = [liver_pred, lungs_pred, heart_pred]

# Dictionary to store evaluation results for all organs
results = {
    'Organ': [],              # Organ names
    'Dice Coefficient': [],   # Dice scores
    'IoU Score': [],          # IoU scores
    'Accuracy': []            # Accuracy scores
}

# =============================================================================================
# CALCULATE METRICS FOR EACH ORGAN
# =============================================================================================

# Print header
print("=" * 70)
print("          Medical Image Segmentation - Evaluation Metrics")
print("=" * 70)
print()

# Loop through each organ and calculate metrics
for i in range(3):
    # Get current organ name
    organ = organs[i]
    
    # Calculate Dice Coefficient for current organ
    dice = dice_coefficient(ground_truths[i], predictions[i])
    
    # Calculate IoU Score for current organ
    iou = iou_score(ground_truths[i], predictions[i])
    
    # Calculate Accuracy for current organ
    acc = accuracy(ground_truths[i], predictions[i])
    
    # Store results in dictionary
    results['Organ'].append(organ)
    results['Dice Coefficient'].append(dice)
    results['IoU Score'].append(iou)
    results['Accuracy'].append(acc)
    
    # Print individual organ results
    print(f"📊 {organ} Evaluation:")
    print(f"   Dice Coefficient: {dice:.4f}")
    print(f"   IoU Score:        {iou:.4f}")
    print(f"   Accuracy:         {acc:.4f}")
    print("-" * 70)

# =============================================================================================
# CREATE COMPARISON TABLES
# =============================================================================================

# Convert results dictionary to pandas DataFrame for better visualization
df = pd.DataFrame(results)

# Print main comparison table
print()
print("=" * 70)
print("                    📋 Comparison Table")
print("=" * 70)
print()
print(df.to_string(index=False))
print()
print("=" * 70)

# =============================================================================================
# CALCULATE AND DISPLAY AVERAGE METRICS
# =============================================================================================

print()
print("📈 Average Metrics Across All Organs:")
print(f"   Average Dice:     {df['Dice Coefficient'].mean():.4f}")
print(f"   Average IoU:      {df['IoU Score'].mean():.4f}")
print(f"   Average Accuracy: {df['Accuracy'].mean():.4f}")
print()
print("=" * 70)

# =============================================================================================
# SAVE RESULTS TO CSV FILE
# =============================================================================================

# Save the results DataFrame to a CSV file for future reference
df.to_csv('evaluation_results.csv', index=False)
print()
print("✅ Results saved to 'evaluation_results.csv'")
print()

# =============================================================================================
# CREATE DETAILED STATISTICS TABLE (TRANSPOSED VIEW)
# =============================================================================================

print("=" * 70)
print("              📊 Detailed Statistics Table")
print("=" * 70)
print()

# Create a transposed table showing each metric as a row
# and each organ as a column for easier comparison
stats_df = pd.DataFrame({
    'Metric': ['Dice Coefficient', 'IoU Score', 'Accuracy'],
    'Liver': [
        f"{df.loc[df['Organ'] == 'Liver', 'Dice Coefficient'].values[0]:.4f}",
        f"{df.loc[df['Organ'] == 'Liver', 'IoU Score'].values[0]:.4f}",
        f"{df.loc[df['Organ'] == 'Liver', 'Accuracy'].values[0]:.4f}"
    ],
    'Lungs': [
        f"{df.loc[df['Organ'] == 'Lungs', 'Dice Coefficient'].values[0]:.4f}",
        f"{df.loc[df['Organ'] == 'Lungs', 'IoU Score'].values[0]:.4f}",
        f"{df.loc[df['Organ'] == 'Lungs', 'Accuracy'].values[0]:.4f}"
    ],
    'Heart': [
        f"{df.loc[df['Organ'] == 'Heart', 'Dice Coefficient'].values[0]:.4f}",
        f"{df.loc[df['Organ'] == 'Heart', 'IoU Score'].values[0]:.4f}",
        f"{df.loc[df['Organ'] == 'Heart', 'Accuracy'].values[0]:.4f}"
    ],
    'Average': [
        f"{df['Dice Coefficient'].mean():.4f}",
        f"{df['IoU Score'].mean():.4f}",
        f"{df['Accuracy'].mean():.4f}"
    ]
})

# Print the detailed statistics table
print(stats_df.to_string(index=False))
print()
print("=" * 70)

# =============================================================================================
# END OF SCRIPT
# =============================================================================================