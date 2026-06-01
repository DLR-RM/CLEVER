# CLEVER: Stream-based Active Learning for Robust Semantic Perception from Human Instructions

This repository is the official code base for the work:

**[CLEVER: Stream-based Active Learning for Robust Semantic Perception from Human Instructions]** \
Jongseok Lee, Timo Birr, Rudolph Triebel and Tamim Asfour (IEEE Robotics and Automation Letter 2025).\
[[Paper](https://ieeexplore.ieee.org/document/11078143)] 
[[Poster](https://jlee4176901.github.io/files/publications/ral2025poster.pdf)] 
[[Website](https://sites.google.com/view/thecleversystem)]
[[Video](https://www.youtube.com/watch?v=wO1zHbYAazI)]

## Description
### When perceiving the environments, can robots ask for help and learn from humans demonstrations?

CLEVER is a stream-based active learning system that improves the robustness of deep neural network (DNN) semantic perception by continuously learning from human feedback during operation. Unlike traditional models that are trained once and then deployed, CLEVER detects uncertain or potentially incorrect predictions, requests human annotations when needed, and updates its model online to adapt to new situations and objects. The system addresses key challenges in active learning, including uncertainty estimation, catastrophic forgetting, efficient online adaptation, and informative sample selection. To achieve this, it uses Bayesian neural networks through Laplace Approximation, which provides probabilistic estimates of model uncertainty by approximating a distribution over network parameters. Furthermore, CLEVER employs Bayesian continual learning, using knowledge from previous tasks as informative priors for new tasks, enabling better generalization in small-data settings. Experiments with human participants and a humanoid robot demonstrate that CLEVER can learn new object concepts and improve semantic perception in real-world environments, making it one of the first stream-based active learning systems successfully deployed on a real robot.

## Installation

You can install this project either with conda or the Python Package Installer (pip).

1. Clone or download the repository
  ```bash
  git clone https://rmc-github.robotic.dlr.de/lee-jn/CLEVER.git
  ```
2. Install the packages either with conda (recommended) or with pip:
  - Create a new environment called `clever`
  ```bash
  conda create -n clever python=3.10
  conda activate clever
  ```
  - Install [PyTorch and Torchvision](https://pytorch.org/get-started/locally/). I recommend to install versions above 2.2.0
  - Go into the root folder and install the remaining packages:
  ```bash
  pip install ultralytics
  pip install tqdm pyscaffold prompt_toolkit toma fvcore
  ```
  - Then, install CLEVER (inside CLEVER folder with setup.py):
  ```bash
      pip3 install -e .
  ```
  - Optionally, if you want the evaluation pipeline, install:
  ```bash
  pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics"
  ```

## Project Organization

```
├── LICENSE.txt                             <- The GNU General Public License.
├── README.md                               <- The top-level README.
├── data                                    <- The datasets used in the experiments.
│   ├── model                               <- Raw data files for
│   │   ├── clever_v1                       <- Folder where classifier models are being stored
│   │   └── dinov2                          <- Folder where dinov2 weights are being saved
|   |   └── sam/weight                      <- Folder segment anything models are being saved
│   └── images                              <- All other relevant images are saved here
|   |   └── pool                            <- Folder that keeps pool of saved images
├── pyproject.toml                          <- Build system configuration.
├── setup.cfg                               <- Declarative configuration of the project.
├── setup.py
├── src                                     <- The implementation of the main functionality.
│   ├── curvature                           <- The curvature implementations for Laplace Approximation
│   │   ├── curvatures.py                   <- Different curvature approximations including K-FOC.
│   │   └── utils.py                        <- Utilities to compute the curvatures (e.g. power method).
│   └── bpnn                                <- The Bayesian Progressive Neural Networks implementation.
│       ├── criterions.py                   <- Different criterions to optimize the weights 
│       │                                       e.g. with PAC-Bayes bounds
│       ├── curvature_scalings.py           <- Different methods to scale the curvature scales.
│       ├── bpnn.py                         <- Main implementation of BPNN and utility functions to fit it.
│       ├── pnn.py                          <- Implementation of PNN (also with MC Dropout) and the 
│       │                                       fitting of general PNN and its adaptions.
│       └── utils.py                        <- Utility functions for BPNN (e.g. metrics, training loop)
│   └── clever                              <- The CLEVER implementation.
│       ├── data                            <- Contains different data loaders
│       └── activelearner.py                <- Implementation of acquisition including batchbald
│       └── baselines.py                    <- Baseline implementations
│       └── clever.py                       <- Main implementation of CLEVER
│       └── metric.py                       <- Utility functions for evaluation
│       └── model.py                        <- Progressive model with Bayesian priors
│       └── predictions.py                  <- Different prediction mechanisms
│       └── priors.py                       <- Prior learning components
│       └── trainer.py                      <- Functions to train BPNN
│       └── utils.py                        <- Utility functions for BPNN (e.g. metrics, training loop)
├── tools                                   <- Tools to run experiments.
│   ├── bpnn                                <- CLEVER using Bayesian progressive neural networks
│       ├── demo.py                         <- Demo file with learning-based priors
│       ├── prior_all_objects.py            <- Learn priors for predefined objects
│       ├── prior_per_object.pt             <- Learn priors for predefined objects per class
│   ├── pnn                                 <- CLEVER using deterministic progressive neural networks
│       ├── demo.py                         <- Demo file without priors
│       ├── train_all_objects.py            <- Train a progressive network for predefined objects
│       ├── train_per_object.py             <- Train a progressive network for predefined objects per class
│   ├── data                                <- Generates the dataloaders used to train the small-scale 
│   ├── scripts                             <- Generates the dataloaders used to train the small-scale 
└── .coveragerc                             <- Configuration for coverage reports of unit tests.
```

## Citation
If you find this project useful, please cite us in the following ways:
```
@ARTICLE{clever2025,
  author={Lee, Jongseok and Birr, Timo and Triebel, Rudolph and Asfour, Tamim},
  journal={IEEE Robotics and Automation Letters},
  title={CLEVER: Stream-Based Active Learning for Robust Semantic Perception From Human Instructions},
  year={2025},
  volume={10},
  number={9},
  pages={8906-8913}
}
```