import tqdm
from config import *
import numpy as np
import sys
sys.path.append("../")

from utils.metrics import keypoint_pck_accuracy, keypoint_epe
from utils.utils import heatmaps_to_coordinates

def get_bb_w_and_h(gt_keypoints, bb_factor = 1):
    '''
    inputs:
        gt_keypoints shape (batch_size, 2)

    returns:
        normalize (batch_size, (bb_width, bb_height))
    '''
    
    normalize = np.zeros((gt_keypoints.shape[0], 2))
   
    # normalize = get_bb_batch(true_keypoints)
    for idx, img in enumerate(gt_keypoints):
       
        xmax, ymax = img.max(axis=0)
        xmin, ymin = img.min(axis=0)
       
        width = xmax - xmin
        height = ymax - ymin
        normalize[idx][0] = width * bb_factor
        normalize[idx][1] = height * bb_factor

    return normalize

def keypoint_auc(pred, gt, mask, normalize, num_step=20):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (float): Normalization factor.

    Returns:
        float: Area under curve.
    """
    # print('Normalize: ', normalize.shape, ' - ',normalize)

    
    # nor = np.tile(np.array([[normalize, normalize]]), (pred.shape[0], 1))
    nor = normalize
    # print('Nor: ', nor.shape, ' - ', nor)
    x = [1.0 * i / num_step for i in range(num_step)]
    y = []
    for thr in x:
        _, avg_acc, _ = keypoint_pck_accuracy(pred, gt, mask, thr, nor)
        y.append(avg_acc)

    auc = 0
    for i in range(num_step):
        auc += 1.0 / num_step * y[i]
    return auc
def batch_epe_calculation(pred_keypoints, true_keypoints, treshold = 0.2, mask = None, normalize = None):
    
    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21),dtype=int)

    epe = keypoint_epe(pred_keypoints * MODEL_IMG_SIZE, true_keypoints * MODEL_IMG_SIZE, mask)

    return epe

def batch_auc_calculation(pred_keypoints, true_keypoints, num_step=20, mask = None, normalize = None):

    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21),dtype=int)
    
    if normalize == None:
        normalize = get_bb_w_and_h(true_keypoints)
    
    auc = keypoint_auc(pred = pred_keypoints, gt = true_keypoints, mask = mask, normalize = normalize, num_step=num_step)

    return auc

def batch_pck_calculation(pred_keypoints, true_keypoints, treshold = 0.2, mask = None, normalize = None):
    
    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21),dtype=int)
    
    if normalize == None:
        normalize = get_bb_w_and_h(true_keypoints)
      
    _, avg_acc, _ = keypoint_pck_accuracy(pred=pred_keypoints,gt=true_keypoints,mask = mask,thr= treshold,normalize= normalize)

    return avg_acc

def evaluate(model, dataloader, using_heatmaps = True, batch_size = 0):
    accuracy_all = []
    image_id = []
    pred = []
    gt = []
    pck_acc = []
    epe_lst = []
    auc_lst = []

    for data in tqdm(dataloader):
        inputs = data["image"]
        pred_heatmaps = model(inputs)
        # print(pred_heatmaps.shape)
        pred_heatmaps = pred_heatmaps.detach().numpy()
        true_keypoints = (data["keypoints"]).numpy()

        if using_heatmaps == True:
            pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)
        else:
            pred_keypoints = pred_heatmaps.reshape(batch_size,N_KEYPOINTS,2)

            # true_keypoints = torch.flatten(true_keypoints,1)#.to(torch.float)

        # print('Pred shape: ', pred_keypoints.shape)
        # print('GT shape: ', true_keypoints.shape)

        accuracy_keypoint = ((true_keypoints - pred_keypoints) ** 2).sum(axis=2) ** (1 / 2)
        accuracy_image = accuracy_keypoint.mean(axis=1)
        accuracy_all.extend(list(accuracy_image))

        # Calculate PCK@02
        avg_acc = batch_pck_calculation(pred_keypoints, true_keypoints, treshold = 0.2, mask = None, normalize = None)
        pck_acc.append(avg_acc)

        # Calculate EPE mean and median, mind that it depends on what scale of input keypoints 
        epe = batch_epe_calculation(pred_keypoints, true_keypoints)
        epe_lst.append(epe)

        #TODO calculate AUC
        auc = batch_auc_calculation(pred_keypoints, true_keypoints, num_step=20, mask = None)
        auc_lst.append(auc)

    pck = sum(pck_acc) / len(pck_acc)
    epe_final = sum(epe_lst) / len(epe_lst)
    auc_final = sum(auc_lst) / len(auc_lst)

    print (f'PCK@2: {pck}, EPE: {epe_final}, AUC: {auc_final}')
    return accuracy_all, pck