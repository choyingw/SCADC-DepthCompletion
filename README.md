# SCADC-DepthCompletion
Scene Completeness-Aware Lidar Depth Completion for Driving Scenario, ICASSP 2021

Cho-Ying Wu and Ulrich Neumann, Universit of Southern California

<img src='demo.gif'>

The full example video link is here https://www.youtube.com/watch?v=FQDTdpMPKxs

Paper: https://arxiv.org/abs/2003.06945

**Advantages:**

\+ **First research to attend scene-completeness in depth completion**

\+ **Sensor Fusion for lidar and stereo cameras**

\+ **Structured upper scene depth**

\+ **Precise lower scene**

# Prerequisite

	Linux
	Python 3
	PyTorch 1.5+ (Tested on 1.5)
	NVIDIA GPU + CUDA CuDNN 
	Other common libraries: matplotlib, cv2, PIL

# Data Preparation

Clone the folder first.

Then, download preprocessed data from <a href="https://drive.google.com/file/d/1c78Ox6KfaUkXZf4qx5hVly9Na_QJ5VIv/view?usp=sharing">here</a>. (153G) This data includes training/testing split that follows KITTI Completion and all required pre-processed data for this work. 

Extract the files under the repository. The structure should be like 'SCADC-DepthCompletion/Data/train' and 'SCADC-DepthCompletion/Data/test'

\*.h5 files are provided, including sparse depth, semi-dense depth, left-right pairs, depth completed from <a href="https://github.com/fangchangma/self-supervised-depth-completion">SSDC</a>, and disparity from <a href="https://github.com/JiaRenChang/PSMNet">PSMNet</a>.

# Sample Evaluation/Training Commands:

	python3 evaluate.py --name kitti --checkpoints_dir './test_ckpt'--test_path [test_data_dir]

	python3 train_depth_complete.py --name kitti --checkpoints_dir [preferred saving ckpt path] --train_path [train_data_dir] --test_path [test_data_dir]

\[train_data_dir\]: it should be 'Data/train' when you follow the recommended folder structure
\[test_data_dir\]: it should be 'Data/test' when you follow the recommended folder structure

# Customized depth completion and stereo estimation base methods:

Note that we use <a href="https://github.com/fangchangma/self-supervised-depth-completion">SSDC</a>, and disparity from <a href="https://github.com/JiaRenChang/PSMNet">PSMNet</a>. The pre-processed data is in the \*.h5 files. (key: 'depth_c' and 'disp_c'). If you want to make completion results from different basic methods, please prepare those data at your own and replace data stored in \*.h5 files.


If you find our work useful, please consider to cite our work.

@article{wu2020scene,
  title={Scene Completenesss-Aware Lidar Depth Completion for Driving Scenario},
  author={Wu, Cho-Ying and Neumann, Ulrich},
  journal={arXiv preprint arXiv:2003.06945},
  year={2020}
}

# Acknowledgement

The code development is based on <a href="https://github.com/choyingw/CFCNet">CFCNet</a>, <a href="https://github.com/fangchangma/self-supervised-depth-completion">Self-Supervised Depth Completion</a>, and <a href="https://github.com/JiaRenChang/PSMNet">PSMNet</a>. 