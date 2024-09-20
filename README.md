# autoencoders-CARMENES

This repository contains the necessary code to easily reproduce the methodology and figures (for the results section) presented in Mas-Buitrago et al. 2024 &nbsp; [![paper](https://zenodo.org/badge/DOI/10.1051/0004-6361/202449865.svg)](https://doi.org/10.1051/0004-6361/202449865)

#

### Methodology

The `python` code used to develop the methodology described in Section 3 is available in the directory [./Methodology](https://github.com/pedromasb/autoencoders-CARMENES/tree/main/Methodology).

| File | Description | 
| --- | --- | 
| autoencoder_fine_tuning.py | Fine-tuning, using the spectra observed with CARMENES, of the pre-trained autoencoders. | 
| autoencoder_grid_search.py | Grid search for the best combinations of autoencoder hyperparameters. | 
| autoencoder_training.py | Training of the autoencoders for the best hyperparameters combinations. | 
| cnn_training.py | Training of the CNN regression models. This file shows the training for the effective temperature. | 

#

### Figures

- The data compiled from the literature is available in the directory [./literature_data](https://github.com/pedromasb/autoencoders-CARMENES/tree/main/literature_data).
- **Code for figures in Section 4.1 - stellar parameters analysis:** &nbsp; [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](Section4_1_figs.ipynb)
- **Code for figures in Section 4.2 - comparison with the literature:** &nbsp; [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](Section4_2_figs.ipynb)
