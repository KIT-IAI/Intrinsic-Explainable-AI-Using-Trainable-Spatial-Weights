# Intrinsic Explainable Artificial Intelligence Using Trainable Spatial Weights

This repository contains the Python implementation of experiments to manage anomalies in energy time series presented in the following paper:
>O. Neumann, M. Beichter, B. Heidrich, N. Friederich, V. Hagenmeyer, and R. Mikut, 2024, "Intrinsic Explainable Artificial Intelligence Using Trainable Spatial Weights on Numerical Weather Predictions," in Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems (e-Energy '24). Association for Computing Machinery, Singepore, pp. 551-559. doi: [10.1145/3632775.3662161](https://doi.org/10.1145/3632775.3662161).


## Installation

To install this project, perform the following steps:
1. Clone the project
2. Create an conda virtual environment.
3. Activate the environment.
4. Install Torch as described on their website [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
5. Install all additional dependencies listed in requirements.txt (e.g. using `pip install -r requirements.txt`)


## Downloading the data.

For the experiments we need to download the OPSD and ECMWF datasets.

The OPSD dataset is freely available and can be downloaded using `python download.py` in `data/opsd`.

The ECMWF datasets we need consists of the ERA5 and HRES datasets.
The ERA5 is a analysis dataset that is optimized in advance.
It is freely available after registration.
Please follow the registration guideline and download the ERA5 dataset using `python download_era5.py` in `data/ecmwf`.

The HRES dataset, however, is not free.
If you pay for the API access or applied for free access due to educational purpose, you can also download it after registration via `python download_hres.py` in `data/hres`.


## Excecution

All experiments can be run using `python src/large_scale_forecasting.py`.

For configuration we use the [Hydra](https://hydra.cc) python framework.
Please have a look in the config folder for all available parameters.

The most important parameters are:
- `targets` (e.g. de+load) to denote with target to forecast
- `model` (early_fusion_model, late_fusion_model, or alpha_combined_fusion_model) for the model selection

To change the parameters you can edit the `config/run.yaml` or pass it to the python programm (e.g. `python src/large_scale_forecasting.py targets=[de+load] model=late_fusion_model`)


## Funding

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI and the Helmholtz Association under the Program “Energy System Design”.


## License
This code is licensed under the [MIT License](LICENSE).
