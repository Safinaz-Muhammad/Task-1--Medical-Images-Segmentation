import numpy as np
from scipy.spatial.distance import directed_hausdorff
import warnings
warnings.filterwarnings('ignore')

#claude
# -----------------------------
# Replace these with your actual .npy mask file paths
# Ground truth masks (binary masks: 0 and 1)
liver_gt = np.load('liver_gt.npy')
lungs_gt = np.load('lungs_gt.npy')
heart_gt = np.load('heart_gt.npy')

# Predicted masks (binary masks: 0 and 1)
liver_pred = np.load('liver_pred.npy')
lungs_pred = np.load('lungs_pred.npy')
heart_pred = np.load('heart_pred.npy')
# -----------------------------

# Dice coefficient calculation
def dice_coefficient(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)
    return dice

# IoU (Jaccard Index)
def iou_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-8)

# Hausdorff distance
def hausdorff_distance(y_true, y_pred):
    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)
    if true_points.size == 0 or pred_points.size == 0:
        return np.inf
    forward_hd = directed_hausdorff(true_points, pred_points)[0]
    backward_hd = directed_hausdorff(pred_points, true_points)[0]
    return max(forward_hd, backward_hd)

# Sensitivity (Recall / True Positive Rate)
def sensitivity(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return tp / (tp + fn + 1e-8)

# Specificity (True Negative Rate)
def specificity(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return tn / (tn + fp + 1e-8)

# Precision (Positive Predictive Value)
def precision(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return tp / (tp + fp + 1e-8)

# Volume Similarity
def volume_similarity(y_true, y_pred):
    vol_true = np.sum(y_true)
    vol_pred = np.sum(y_pred)
    return 1 - abs(vol_true - vol_pred) / (vol_true + vol_pred + 1e-8)

# 95th percentile Hausdorff Distance (more robust to outliers)
def hausdorff_95(y_true, y_pred):
    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)
    if true_points.size == 0 or pred_points.size == 0:
        return np.inf
    
    # Compute distances from each true point to nearest pred point
    from scipy.spatial.distance import cdist
    distances = cdist(true_points, pred_points)
    forward_distances = np.min(distances, axis=1)
    backward_distances = np.min(distances, axis=0)
    
    all_distances = np.concatenate([forward_distances, backward_distances])
    return np.percentile(all_distances, 95)

# Evaluation loop
organs = ['Liver', 'Lungs', 'Heart']
ground_truths = [liver_gt, lungs_gt, heart_gt]
predictions = [liver_pred, lungs_pred, heart_pred]

print("=" * 80)
print("MEDICAL IMAGE SEGMENTATION EVALUATION")
print("=" * 80)

results = {}
for i in range(3):
    organ = organs[i]
    print(f"\n{'─' * 80}")
    print(f"📊 {organ.upper()} SEGMENTATION METRICS")
    print(f"{'─' * 80}")
    
    # Calculate all metrics
    dice = dice_coefficient(ground_truths[i], predictions[i])
    iou = iou_score(ground_truths[i], predictions[i])
    sens = sensitivity(ground_truths[i], predictions[i])
    spec = specificity(ground_truths[i], predictions[i])
    prec = precision(ground_truths[i], predictions[i])
    vol_sim = volume_similarity(ground_truths[i], predictions[i])
    hausdorff = hausdorff_distance(ground_truths[i], predictions[i])
    hausdorff_95_val = hausdorff_95(ground_truths[i], predictions[i])
    
    # Store results
    results[organ] = {
        'Dice': dice,
        'IoU': iou,
        'Sensitivity': sens,
        'Specificity': spec,
        'Precision': prec,
        'Volume Similarity': vol_sim,
        'Hausdorff': hausdorff,
        'Hausdorff 95': hausdorff_95_val
    }
    
    # Display results
    print(f"  Dice Coefficient:      {dice:.4f}")
    print(f"  IoU Score:             {iou:.4f}")
    print(f"  Sensitivity (Recall):  {sens:.4f}")
    print(f"  Specificity:           {spec:.4f}")
    print(f"  Precision:             {prec:.4f}")
    print(f"  Volume Similarity:     {vol_sim:.4f}")
    print(f"  Hausdorff Distance:    {hausdorff:.2f} pixels")
    print(f"  Hausdorff 95%:         {hausdorff_95_val:.2f} pixels")

# Summary statistics
print(f"\n{'=' * 80}")
print("📈 SUMMARY STATISTICS")
print(f"{'=' * 80}")
print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print(f"{'-' * 80}")

metrics_to_summarize = ['Dice', 'IoU', 'Sensitivity', 'Specificity', 'Precision', 'Volume Similarity']
for metric in metrics_to_summarize:
    values = [results[organ][metric] for organ in organs]
    print(f"{metric:<25} {np.mean(values):<12.4f} {np.std(values):<12.4f} {np.min(values):<12.4f} {np.max(values):<12.4f}")

print(f"{'=' * 80}\n")

# Optional: Export results to CSV
try:
    import pandas as pd
    df = pd.DataFrame(results).T
    df.to_csv('segmentation_results.csv')
    print("✅ Results saved to 'segmentation_results.csv'")
except ImportError:
    print("ℹ  Install pandas to export results to CSV: pip install pandas")