# GC-ConsFlow-Leveraging-Optical-Flow-Residuals-and-Global-Context-for-Robust-Deepfake-Detection
ICME GC-ConsFlow: Leveraging Optical Flow Residuals and Global Context for Robust Deepfake Detection
Data preparation

Data_frame_DATASET: Please note that since our optical flow requires continuous video frames as input, the normal video frames here should be continuous video frames.

Data_flow_DATASET:
Cd dataset;
Python flow.py//You should modify the input address you need to get the optical flow and the address you want to save the optical flow

Train
python trainv1.py -e [EPOCH] -b [BATCH_SIZE] -l 0.0001 -w 0.0000001 -t y 

Test
python test.py
