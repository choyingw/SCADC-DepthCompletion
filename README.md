# SCADC-DepthCompletion
Scene Completeness-Aware Lidar Depth Completion for Driving Scenario, ICASSP 2021

Cho-Ying Wu and Ulrich Neumann, University of Southern California

<img src='demo.gif'>

The full example video link is here https://www.youtube.com/watch?v=FQDTdpMPKxs

Paper: https://arxiv.org/abs/2003.06945

Project page: https://choyingw.github.io/works/SCADC/index.html

**Advantages:**

👍 **First research to attend scene-completeness in depth completion**

👍 **Sensor Fusion for lidar and stereo cameras**

👍 **Structured upper scene depth**

👍 **Precise lower scene**

# Prerequisite

	Linux
	Python 3
	PyTorch 1.5+ (Tested on 1.5)
	NVIDIA GPU + CUDA CuDNN 
	Other common libraries: matplotlib, cv2, PIL

# Data Preparation

Clone the repo first.

Then, download preprocessed data from <a href="https://drive.google.com/file/d/1c78Ox6KfaUkXZf4qx5hVly9Na_QJ5VIv/view?usp=sharing">train</a> (142G) <a href="https://drive.google.com/file/d/1RXJ5GFhE0ZIIBf4wcLXhilu4OVQ1BiEg/view?usp=sharing">val</a> (11G). This data includes training/val split that follows KITTI Completion and all required pre-processed data for this work.

Extract the files under the repository. The structure should be like 'SCADC-DepthCompletion/Data/train' and 'SCADC-DepthCompletion/Data/val'

\*.h5 files are provided, including sparse depth (D), semi-dense depth (D_semi), left-right pairs (I_L and I_R), depth completed from <a href="https://github.com/fangchangma/self-supervised-depth-completion">SSDC</a> (depth_c), and disparity from <a href="https://github.com/JiaRenChang/PSMNet">PSMNet</a> (disp_c).

# Evaluation/Training Commands:

Our provided pretrained weight is under './test_ckpt/kitti/'. To quickly get our scene completeness-aware depth maps, just use the evaluation command, and it will save frame-by-frame results under './vis/'.

	python3 evaluate.py --name kitti --checkpoints_dir './test_ckpt'--test_path [test_data_dir]

This is the training command is you want ot train the network yourself.

	python3 train_depth_complete.py --name kitti --checkpoints_dir [preferred saving ckpt path] --train_path [train_data_dir] --test_path [test_data_dir]

\[train_data_dir\]: it should be 'Data/train' when you follow the recommended folder structure
\[test_data_dir\]: it should be 'Data/test' when you follow the recommended folder structure

# Customized depth completion and stereo estimation base methods:

Note that we use <a href="https://github.com/fangchangma/self-supervised-depth-completion">SSDC</a>, and disparity from <a href="https://github.com/JiaRenChang/PSMNet">PSMNet</a>. 

The pre-processed data is in the \*.h5 files. (key: 'depth_c' and 'disp_c'). If you want to make completion results from different basic methods, please prepare those data at your own and replace data stored in \*.h5 files.


If you find our work useful, please consider to cite our work.

	@inproceedings{wu2021scene,
	  title={Scene Completeness-Aware Lidar Depth Completion for Driving Scenario},
	  author={Wu, Cho-Ying and Neumann, Ulrich},
	  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
	  pages={2490--2494},
	  year={2021},
	  organization={IEEE}
	}


# Acknowledgement

The code development is based on <a href="https://github.com/choyingw/CFCNet">CFCNet</a>, <a href="https://github.com/fangchangma/self-supervised-depth-completion">Self-Supervised Depth Completion</a>, and <a href="https://github.com/JiaRenChang/PSMNet">PSMNet</a>. 
