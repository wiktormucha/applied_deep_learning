''' 
File with functiosn for hand detection
'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from config import *
from utils.utils import project_points_3D_to_2D


def pil_to_cv(pil_image):
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image

def cv_to_pil(cv_image):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    return pil_image

def crop_image(image, bbox):

    cropped_image = image[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']]
    return cropped_image


def gt_to_segments(gt, bbox):
    '''
    Function applyes segmentic keypoints from main image using given boundigboxes
    '''
    gt[:,0] = gt[:,0] - bbox['x_min']
    gt[:,1] = gt[:,1] - bbox['y_min']
    
    return gt


def compress_gt_pts(gt,img_dimm):
  
    compressed_gt = gt / img_dimm
    
    return compressed_gt

def get_bb(hand_info, hand_landmarks, w, h, factor = BB_FACTOR, mirror = False):
    '''
    Function to get bounding boxes from mediapipe predictions.
    '''

    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
  
    hand_dict = {'index':hand_info.classification[0].index,
                'label':hand_info.classification[0].label

    }

    if mirror == True:

        if hand_dict['label'] == 'Right':
            hand_dict['label'] = 'Left'
        else:
            hand_dict['label'] = 'Right'

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y

    # Here BBs are increased by a given factor
    hand_dict['x_min'] = x_min - factor
    hand_dict['x_max'] = x_max + factor

    hand_dict['y_min'] = y_min - factor
    hand_dict['y_max'] = y_max + factor

    bb_w = hand_dict['x_max'] - hand_dict['x_min']
    bb_h = hand_dict['y_max'] - hand_dict['y_min']
    

    # Here the BBs are modified to be square shape
    diff = bb_w - bb_h
   
    # if w > h
    if diff > 0:
        hand_dict['y_min'] -= int(diff/2)
        if diff % 2 == 0:
            hand_dict['y_max'] += int(diff/2)
        else:
            hand_dict['y_max'] += int(diff/2) + 1
    # if h > w    
    elif diff < 0:
        diff = abs(diff)
        hand_dict['x_min'] -= int(diff/2)
        if diff % 2 == 0:
            hand_dict['x_max'] += int(diff/2)
        else:
            hand_dict['x_max'] += int(diff/2) + 1


    return hand_dict

def get_hands_bb(img, hands):
    frame = img
    h, w, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    hand_class = result.multi_handedness
    hand_bb_list = []
    
    if hand_landmarks:
        for handLMs,hand in zip(hand_landmarks,hand_class):       
            hand_bb = get_bb(hand, handLMs,w =w, h=h, mirror= True)
            hand_bb_list.append(hand_bb)
            # cv2.rectangle(frame, (hand_bb['x_min'], hand_bb['y_min']), (hand_bb['x_max'], hand_bb['y_max']), (0, 255, 0), 2)
            # cv2.putText(frame, hand_bb['label'], (hand_bb['x_min'], hand_bb['y_min']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return hand_bb_list

def get_hands_img(img, gt_pts, hand_model, cam_instr):
    '''
    This function segments hands from main image and return dicitonary with image hand and label (left/right)
    '''
    
    img = pil_to_cv(img)

    hand_bb_list = get_hands_bb(img, hands = hand_model)
    gt_pts= np.split(gt_pts,[1,64,65,128]) 


    hand1 = np.reshape(gt_pts[1], (21,3))
   
    hand2 = np.reshape(gt_pts[3], (21,3))
    ptsL = project_points_3D_to_2D(hand1,cam_instr)
    ptsP = project_points_3D_to_2D(hand2,cam_instr)
   
    # For each hand
    hands_seg = []
    gt = []

    for hand_bb in hand_bb_list:
        
        hand = crop_image(img, hand_bb)
        if hand_bb['label'] == 'Left':
            hand_pts = ptsL
        else:
            hand_pts = ptsP

        pts_segm = gt_to_segments(hand_pts,hand_bb)
        compress_pts = compress_gt_pts(pts_segm, hand.shape[0])
        hand = cv_to_pil(hand)
        # Convert back to PIL
        hands_seg.append(hand)
        gt.append(compress_pts)
            
    return {
            'hands_seg': hands_seg,
            'gt': gt
    }