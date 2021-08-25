# State-space deep Gaussian processes in Python

This folder contains necessary Python notebooks that could help you understand and implement SS-DGPs.

# Installation

TL;DR. Run `pip install -r requirements.txt`, or equivalently, run `pip install numpy scipy matplotlib jax jaxlib tme tensorflow gpflow`. Note that you can omit `tensorflow` and `gpflow`, as they are optional in the notebooks.

# How to use?

Open `ssdgp_matern_32_ekfs_tutorial.ipynb` with Jupyter notebook, then you are good to go.

# What are these files?

1. `ssdgp_matern_32_ekfs_tutorial.ipynb`. This is a notebook that shows you how to solve a Matern 3/2 SS-DGP regression problem by using an extended Kalman filter and smoother. 

2. `filters_smoothers.py`. This file contains implementations of (extended) Kalman filters and smoothers.

3. `test_fs.py`. Unittest file.

# TODOs

1. Add more examples with different filters and smoothers.
   
2. Add an example demonstrating parameter learning.

# Citation

Please refer to the parent folder for the citation informaton.
