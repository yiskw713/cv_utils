# Computer Vision Utilities

This repo is for keeping my utilties written with python and pytorch.
Please see [`this repository`](https://github.com/yiskw713/pytorch_template) for pytorch project template.

## Requirements

* poetry
* python >= 3.8
* pytorch >= 1.0
* pytest
* pytest-mock
* pre-commit (for pre-commit formatting, type check and testing)
* opencv
* opencv-contrib-python = 3.x

Please run `poetry install` to install the necessary packages.

## List

* mixup
* Motion History Image
* optical flow
  * Lucas-Kanade
  * Gunnar Farneback
  * Dual TVL1 (NOTE: This is implemented by opencv and takes a lot of time)

## TODO

* [ ] While pytest runs without any errors using poetry,
  it fails in github actions though dependencies are installed with poetry as well.
  This may be caused by `opencv-python` or `opencv-contrib-python`.
* [ ] use GPUs for calculating Dual TVL1 optical flow

## License

This repository is released under the [MIT License](./LICENSE)

## Reference

* James W. Davis and Aaron F. Bobick "The Representation and Recognition of Action Using Temporal Templates" in CVPR 1997
* H. Zhang+ "mixup: BEYOND EMPIRICAL RISK MINIMIZATION" in ICLR2018
* B. D. Lucas and T. Kanade (1981), "An iterative image registration technique with an application to stereo vision" in Proceedings of Imaging Understanding Workshop, pages 121--130
* Gunnar Farneb¨ack, "Two-frame motion estimation based on polynomial expansion", in Scandinavian conference on Image analysis (pp. 363-370). Springer, Berlin, Heidelberg, 2002.
