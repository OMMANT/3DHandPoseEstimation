# 3DHandPoseEstimation

## Introduction  
This project is estimating 3D Hand Pose from a single RGB Image by using CNN(Convolutional-Neural-Network).
This work is based on <a href="https://arxiv.org/pdf/1705.01389v3.pdf">ICCV 2017 paper</a>

## Usage
1. Download <a href="https://drive.google.com/file/d/1zgdQVR73GtIQmqi86nUu2amuTwf3GUjv/view?usp=sharing">data</a>
and unzip it into the project root folder (This will create a folder "res")
2. <code python>python main.py</code> run the network with 10 sample images  
You can add some images for testing network simply adding <strong>'.png'</strong> files at <strong>"/res/img"</strong> 

## Development Enviornment
<strong>Software</strong>
* Windows10 Pro 1909
* Python 3.7.7
* TensorFlow 2.1.0
* numpy 1.18.1
* opencv 3.4.1
* matlplotlib 3.1.3
* CUDA 10.1.243 and CUDNN 7.6.5
