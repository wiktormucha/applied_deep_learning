1. Error Metrics

Loss function - The model uses heatmaps, and because of this, the loss function is IoU loss for heatmaps.

Evaluation:

PCK@02 -> Percentage of key points under the threshold of 0.2. The keypoint is correctly predicted if the difference between it and GT is lower than threshold 0.2, which is a normalised value using a bounding box surrounding the hand. The original dataset does not include bounding boxes. BBs are obtained by souranding GT keypoints. It creates bounding boxes smaller than they should be and decreases the final value.

EPE -> End Point Error - > distance between prediction and ground truth normalised

AUC -> Area under the curve of accuracy over thresholds (PCK for all thresholds) 

2. Target of the metric to achieve

FreiHAND dataset is created for 3D prediction. Under this project, I have made 2D prediction, and the comparison is different.

The best is always to beat what exists. MMDetection and MMPose library in 2D prediction on FreiHAND using ResNet50 achieve:

PCK@02 = 0.993
EPE = 3.25
AUC = 0.868 

Other results I have found by Santavas et al.
https://arxiv.org/abs/2001.08047

PCK@02 = unknown
EPE = 4
AUC = 0.87

For this moment, I have decided not to evaluate these metrics with an egocentric perspective. However, I have found one dataset containing annotations for hands in egocentric vision, but they are created artificially and do not look very accurate.

3. Acquired values of metrics (using data split 80/15/5)

Best metrics using CustomHeatmapsModel with augmentations: RandomNoise, RandomBoxes and Gaussian blur from pytorch
Learning rate 0.1 with scheduler
Optimiser: 
Learning rate = 0.1 with scheduler
Weight Decay = 1e-5
Momentum = 0.9

PCK@02 = 0.9963
EPE = 1.9433
AUC = 0.9246

Examples of prediction in .jpg file example_prediction.png

4. Amount of time spent on each task

-Downloading datasets and familiarising myself with them. Creating data loaders planned 2 days -> done in 1 day 
-Network building  3 days -> including baseline model it took no more than 3 days
-Training, fine-tuning, and improvements planned 7 days -> this is hard to estimate as the model was learning for a couple of days. 
                                                            After training, I changed 1 or 2 parameters and reran it. In total, it was three weeks, but not all time working. 
-Creating annotations for egocentric datasets - 2 days -> this was not needed and now, from the perspective, seems to be a bad idea 
                                                            (how to do it accurately? The best would be synthetic data)
-Application to present the work - 1 day -> The notebook to apply it to egocentric vision took me 2 days. I was struggling with hand detection, what is the first step. 
                                            In the end I used an existing library that already included a model
-Creating final presentation - 1 day -> #TODO