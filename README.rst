.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/clever.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/clever
    .. image:: https://readthedocs.org/projects/clever/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://clever.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/clever/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/clever
    .. image:: https://img.shields.io/pypi/v/clever.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/clever/
    .. image:: https://img.shields.io/conda/vn/conda-forge/clever.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/clever
    .. image:: https://pepy.tech/badge/clever/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/clever
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/clever

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

======
clever
======


    Add a short description here!


A longer description of your project goes here...


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.


1. install pytorch and torchvision

2. isntall mobile SAM

git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM; pip install -e .

3. install bpnn
pip install -e .

4. pip install mmsegmentation==0.27.0
 
5. pip install -U openmim
mim install mmcv==1.5.0
mim install mmcv-full==1.5.0
# maybe you dont install mmcv and mmsegmentation
# just go with what David had ; otherwise, lots of weird troubles.

6. pip install --upgrade pyscaffold
pip install pytest

7. install devel version of clever

8. pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics"


  641  2024-04-03 15:30:51  export CUDA_HOME=/common/homes/external/dy6919_lee/armarx_ws/deps/cuda/cuda-11.7-cudnn-8.5/
  642  2024-04-03 15:30:55  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
  643  2024-04-03 15:32:15  pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
  644  2024-04-03 15:34:55  python3
  645  2024-04-03 15:35:20  export CUDA_HOME=/common/homes/external/dy6919_lee/armarx_ws/deps/cuda/cuda-11.7-cudnn-8.5/
  646  2024-04-03 15:35:28  pip3 install spatial-correlation-sampler
  647  2024-04-03 15:38:18  python3
  648  2024-04-03 15:38:34  cd ..
  649  2024-04-03 15:38:36  cd 
  650  2024-04-03 15:38:37  cd Packages/
  651  2024-04-03 15:38:40  cd Pytorch-Correlation-extension/
  652  2024-04-03 15:38:55  python3 setup.py install
  653  2024-04-03 15:39:37  cd
  654  2024-04-03 15:40:12  pip3 tqdm pascaffold
  655  2024-04-03 15:40:18  pip3 install tqdm
  656  2024-04-03 15:40:24  pip3 install pyscaffold
  657  2024-04-03 15:40:29  pip3 install git+https://github.com/ChaoningZhang/MobileSAM.git
  658  2024-04-03 15:40:41  pip3 install timm
  659  2024-04-03 15:40:55  cd Packages/
  660  2024-04-03 15:40:59  cd BPNN/
  661  2024-04-03 15:41:04  pip3 install -e .
  662  2024-04-03 15:42:14  pip3 install opencv-python
  663  2024-04-03 15:42:27  pip3 install pyscaffold
  664  2024-04-03 15:42:42  pip3 install toma h5py setuptools imageio prompt_toolkit
  665  2024-04-03 15:43:38  pip3 install toma timm 
  666  2024-04-03 15:43:46  pip3 install toma opencv-python
  667  2024-04-03 15:44:19  cd ..
  668  2024-04-03 15:44:20  cd clever
  669  2024-04-03 15:44:29  pip3 install -e .


----------------
install pytorch first and make sure it has all the relevant files.
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

then install ultralytics (this installs opencv 4.13.0)
pip install ultralytics

then run
pip install tqdm pyscaffold prompt_toolkit toma fvcore

robustness_metrics
pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics"

pip3 install -e .

----------------
- now change the code to incorporate SAM2 instead of SAM + AOK (major)
- test the code with phone based images
----------------
- add a detailed tutorial (minor)
----------------
- clean up and include docstring (minor)
----------------
- research blog (major)
----------------