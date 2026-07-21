from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

'''
def dice(vol1, vol2, num_classes=None):
    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = num_classes[num_classes != 0]  # Remove background class (0)

    f1_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))
    total_voxels = vol1.size
    
    for idx, label in enumerate(num_classes):
        vol1l = vol1 == label
        vol2l = vol2 == label

        tp = np.sum(vol1l & vol2l)  # True Positives
        fp = np.sum(vol2l & ~vol1l)  # False Positives
        fn = np.sum(vol1l & ~vol2l)  # False Negatives

        if tp + fp + fn > 0:
            precision = tp / (tp + fp + np.finfo(float).eps)
            recall = tp / (tp + fn + np.finfo(float).eps)
            f1_scores[idx] = 2 * (precision * recall) / (precision + recall + np.finfo(float).eps)
            class_proportions[idx] = 1.0 * np.sum(vol1l) / total_voxels
        else:
            f1_scores[idx] = 0.0  # Handle cases with no positive predictions or ground truth
            class_proportions[idx] = 1.0 * np.sum(vol1l) / total_voxels
            
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(class_proportions)), f1_scores, color="skyblue", edgecolor="black")
        plt.xticks(range(len(class_proportions)), [f"Class {int(c)}" for c in class_proportions], rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("F1 Scores for Each Class", fontsize=14)
        plt.xlabel("Classes", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig('/home/mega/Desktop/Data3/Han/registration_code/plot/f1_score.png', dpi=300, bbox_inches='tight')

    return f1_scores, class_proportions
'''
    


# DICE SCORE

def dice(vol1, vol2, num_classes=None, nargout=1):

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    dicem = np.zeros(len(num_classes))
    dicem2 = np.zeros(len(num_classes))
    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2. * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = 1.0 * top / bottom
        dicem2[idx] = 1.0 * int(np.sum(vol1l)) / (vol1l.shape[2] * vol1l.shape[3] * vol1l.shape[4])

    if nargout == 1:
        return dicem, dicem2
    else:
        return (dicem, dicem2, num_classes)



#ACCURACY
'''
def dice(vol1, vol2, num_classes=None, nargout=1):

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    accuracy_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))
    total_voxels = vol1.size  # 전체 voxel 개수

    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab

        # True Positives (TP): 클래스에 대해 일치하는 부분
        true_positive = np.sum(np.logical_and(vol1l, vol2l))

        # True Negatives (TN): 클래스 외의 부분에서 일치하는 부분
        true_negative = np.sum(np.logical_and(~vol1l, ~vol2l))

        # False Positives (FP): vol2가 해당 클래스라고 예측했으나 vol1은 아닌 부분
        false_positive = np.sum(np.logical_and(~vol1l, vol2l))

        # False Negatives (FN): vol1이 해당 클래스라고 했으나 vol2는 아닌 부분
        false_negative = np.sum(np.logical_and(vol1l, ~vol2l))

        # Accuracy: (TP + TN) / (TP + FP + FN + TN)
        accuracy_scores[idx] = (true_positive + true_negative) / total_voxels
        class_proportions[idx] = 1.0 * np.sum(vol1l) / total_voxels

    if nargout == 1:
        return accuracy_scores, class_proportions
    else:
        return (accuracy_scores, class_proportions, num_classes)
'''

#PRECISION
'''
def dice(vol1, vol2, num_classes=None, nargout=1):

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    precision_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))
    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab

        # True Positives (TP): vol1과 vol2 모두에서 해당 클래스인 경우
        true_positive = np.sum(np.logical_and(vol1l, vol2l))

        # False Positives (FP): vol2가 해당 클래스라고 예측했으나 vol1은 아닌 경우
        false_positive = np.sum(np.logical_and(~vol1l, vol2l))

        # Precision: TP / (TP + FP)
        denominator = true_positive + false_positive
        denominator = np.maximum(denominator, np.finfo(float).eps)  # Avoid division by zero
        precision_scores[idx] = true_positive / denominator

        # 클래스 비율 계산
        class_proportions[idx] = 1.0 * np.sum(vol1l) / vol1.size

    if nargout == 1:
        return precision_scores, class_proportions
    else:
        return (precision_scores, class_proportions, num_classes)
'''


#SPECIFICITY
'''
def dice(vol1, vol2, num_classes=None, nargout=1):

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    specificity_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))
    total_voxels = vol1.size  # 전체 voxel 개수

    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab

        # True Negatives (TN): 클래스 외의 부분에서 일치하는 부분
        true_negative = np.sum(np.logical_and(~vol1l, ~vol2l))

        # False Positives (FP): vol2가 해당 클래스라고 예측했으나 vol1은 아닌 경우
        false_positive = np.sum(np.logical_and(~vol1l, vol2l))

        # Specificity: TN / (TN + FP)
        denominator = true_negative + false_positive
        denominator = np.maximum(denominator, np.finfo(float).eps)  # Avoid division by zero
        specificity_scores[idx] = true_negative / denominator

        # 클래스 비율 계산
        class_proportions[idx] = 1.0 * np.sum(vol1l) / total_voxels

    if nargout == 1:
        return specificity_scores, class_proportions
    else:
        return (specificity_scores, class_proportions, num_classes)
'''


#RECALL
'''
def dice(vol1, vol2, num_classes=None, nargout=1):

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    recall_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))
    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab

        # True Positives (TP): vol1과 vol2 모두에서 해당 클래스인 경우
        true_positive = np.sum(np.logical_and(vol1l, vol2l))

        # False Negatives (FN): vol1이 해당 클래스라고 했으나 vol2는 아닌 경우
        false_negative = np.sum(np.logical_and(vol1l, ~vol2l))

        # Recall: TP / (TP + FN)
        denominator = true_positive + false_negative
        denominator = np.maximum(denominator, np.finfo(float).eps)  # Avoid division by zero
        recall_scores[idx] = true_positive / denominator

        # 클래스 비율 계산
        class_proportions[idx] = 1.0 * np.sum(vol1l) / vol1.size

    if nargout == 1:
        return recall_scores, class_proportions
    else:
        return (recall_scores, class_proportions, num_classes)    
'''
    
'''   
    # HAUSDORFF DISTANCE

def dice(vol1, vol2, num_classes=None, nargout=1):

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    hausdorff_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))

    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab

        # Extract boundary points for the two volumes
        vol1_points = np.argwhere(vol1l)
        vol2_points = np.argwhere(vol2l)

        if len(vol1_points) == 0 or len(vol2_points) == 0:
            # If either volume has no points for the class, set Hausdorff distance to infinity
            hausdorff_scores[idx] = np.inf
            continue

        # Compute pairwise distances between points in vol1 and vol2
        distances_1_to_2 = cdist(vol1_points, vol2_points)
        distances_2_to_1 = cdist(vol2_points, vol1_points)

        # Hausdorff distance: max of the minimum distances
        h1 = np.max(np.min(distances_1_to_2, axis=1))
        h2 = np.max(np.min(distances_2_to_1, axis=1))
        hausdorff_scores[idx] = max(h1, h2)

        # 클래스 비율 계산
        class_proportions[idx] = 1.0 * np.sum(vol1l) / vol1.size

    if nargout == 1:
        return hausdorff_scores, class_proportions
    else:
        return (hausdorff_scores, class_proportions, num_classes)
'''  
    
'''
#hausdorff distance 95 percentile
def dice(vol1, vol2, num_classes=None, nargout=1):
    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    hausdorff_95_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))

    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab

        # Extract boundary points for the two volumes
        vol1_points = np.argwhere(vol1l)
        vol2_points = np.argwhere(vol2l)

        if len(vol1_points) == 0 or len(vol2_points) == 0:
            # If either volume has no points for the class, set Hausdorff 95 distance to infinity
            hausdorff_95_scores[idx] = np.inf
            continue

        # Compute pairwise distances between points in vol1 and vol2
        distances_1_to_2 = cdist(vol1_points, vol2_points)
        distances_2_to_1 = cdist(vol2_points, vol1_points)

        # Get the minimum distance for each point
        min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
        min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)

        # Compute the 95th percentile of distances
        hd95_1 = np.percentile(min_distances_1_to_2, 95)
        hd95_2 = np.percentile(min_distances_2_to_1, 95)
        hausdorff_95_scores[idx] = max(hd95_1, hd95_2)

        # 클래스 비율 계산
        class_proportions[idx] = 1.0 * np.sum(vol1l) / vol1.size

    if nargout == 1:
        return hausdorff_95_scores, class_proportions
    else:
        return (hausdorff_95_scores, class_proportions, num_classes)
'''



#HD95

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion
'''
def dice(vol1, vol2, num_classes=None, nargout=1):
    vol1 = vol1 > 0
    vol2 = vol2 > 0

    # Distance transform (distance to nearest zero)
    dt1 = distance_transform_edt(~vol1)
    dt2 = distance_transform_edt(~vol2)

    # Extract boundary points
    b1 = vol1 ^ binary_erosion(vol1)
    b2 = vol2 ^ binary_erosion(vol2)

    # distances from vol1 boundary to vol2
    d1 = dt2[b1]
    d2 = dt1[b2]

    hd95_1 = np.percentile(d1, 95)
    hd95_2 = np.percentile(d2, 95)
    
    class_pro = 1
    
    return max(hd95_1, hd95_2), max(hd95_1, hd95_2)
'''


#HD

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion
'''
def dice(vol1, vol2, num_classes=None, nargout=1):
    vol1 = vol1 > 0
    vol2 = vol2 > 0

    # Distance transform (distance to nearest zero)
    dt1 = distance_transform_edt(~vol1)
    dt2 = distance_transform_edt(~vol2)

    # Extract boundary points
    b1 = vol1 ^ binary_erosion(vol1)
    b2 = vol2 ^ binary_erosion(vol2)

    # distances from vol1 boundary to vol2
    d1 = dt2[b1]
    d2 = dt1[b2]

    hd95_1 = np.percentile(d1, 100)
    hd95_2 = np.percentile(d2, 100)
    
    class_pro = 1
    
    return max(hd95_1, hd95_2), max(hd95_1, hd95_2)
'''
'''
def dice(vol1, vol2, num_classes=None, nargout=1, spacing=None):

    v1 = vol1 > 0
    v2 = vol2 > 0

    dt1 = distance_transform_edt(~v1, sampling=spacing)
    dt2 = distance_transform_edt(~v2, sampling=spacing)

    struct = np.ones((3,) * v1.ndim)  
    b1 = v1 ^ binary_erosion(v1, structure=struct)
    b2 = v2 ^ binary_erosion(v2, structure=struct)

    d1 = dt2[b1]
    d2 = dt1[b2]

    n1 = np.count_nonzero(b1)
    n2 = np.count_nonzero(b2)
    if n1 + n2 == 0:
        return 0.0

    asd = (d1.sum() + d2.sum()) / (n1 + n2)
    
    return asd, asd
'''
'''
#asd
def dice(vol1, vol2, num_classes=None, nargout=1):

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    asd_scores = np.zeros(len(num_classes))
    class_proportions = np.zeros(len(num_classes))

    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab

        # Extract boundary points for the two volumes
        vol1_points = np.argwhere(vol1l)
        vol2_points = np.argwhere(vol2l)

        if len(vol1_points) == 0 or len(vol2_points) == 0:
            # If either volume has no points for the class, set ASD to infinity
            asd_scores[idx] = np.inf
            continue

        # Compute pairwise distances between points in vol1 and vol2
        distances_1_to_2 = cdist(vol1_points, vol2_points)
        distances_2_to_1 = cdist(vol2_points, vol1_points)

        # Average Surface Distance: mean of the minimum distances
        d1 = np.mean(np.min(distances_1_to_2, axis=1))
        d2 = np.mean(np.min(distances_2_to_1, axis=1))
        asd_scores[idx] = (d1 + d2) / 2.0

        # Compute class proportions
        class_proportions[idx] = 1.0 * np.sum(vol1l) / vol1.size

    if nargout == 1:
        return asd_scores, class_proportions
    else:
        return (asd_scores, class_proportions, num_classes)
'''


def abs_vol_difference(predictions, labels, num_classes):
    """Calculates the absolute volume difference for each class between
        labels and predictions.

    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate avd for

    Returns:
        np.ndarray: avd per class
    """

    avd = np.zeros((num_classes))
    eps = 1e-6
    for i in range(num_classes):
        avd[i] = np.abs(np.sum(predictions == i) - np.sum(labels == i)
                        ) / (np.float(np.sum(labels == i)) + eps)

    return avd.astype(np.float32)


def crossentropy(predictions, labels, logits=True):
    """Calculates the crossentropy loss between predictions and labels

    Args:
        prediction (np.ndarray): predictions
        labels (np.ndarray): labels
        logits (bool): flag whether predictions are logits or probabilities

    Returns:
        float: crossentropy error
    """

    if logits:
        maxes = np.amax(predictions, axis=-1, keepdims=True)
        softexp = np.exp(predictions - maxes)
        softm = softexp / np.sum(softexp, axis=-1, keepdims=True)
    else:
        softm = predictions
    loss = np.mean(-1. * np.sum(labels * np.log(softm + 1e-8), axis=-1))
    return loss.astype(np.float32)