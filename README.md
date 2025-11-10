# autoencoders-CARMENES

This repository contains the necessary code to easily reproduce the methodology and figures (for the results section) presented in Mas-Buitrago et al. 2024 &nbsp; [![paper](https://zenodo.org/badge/DOI/10.1051/0004-6361/202449865.svg)](https://doi.org/10.1051/0004-6361/202449865)

#


### Data

| Folder | Description | 
| --- | --- | 
| src_files | Input data for the methodology. `carmf_norm.npy`: normalised CARMENES spectra. `ph_pars.npy`: Phoenix stellar parameters for the 22,933 input synthetic spectra. `ph_flux_norm.npy`: available [here](https://cloud.cab.inta-csic.es/s/bEe54r4tEDZpCtP) - normalised 22,933 Phoenix input spectra.| 
| literature_data | Different collections from the literature used in the paper. | 

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
- **Code for figures in Section 4.1 - stellar parameters analysis:** &nbsp; [Section4_1_figs.ipynb](https://github.com/pedromasb/autoencoders-CARMENES/blob/main/Section4_1_figs.ipynb)
- **Code for figures in Section 4.2 - comparison with the literature:** &nbsp; [Section4_2_figs.ipynb](https://github.com/pedromasb/autoencoders-CARMENES/blob/main/Section4_2_figs.ipynb)

### Key Concepts
1. **Contractive Autoencoders**  
   - Encoders/decoders are built symmetrically; a custom contractive loss regularizes the bottleneck layer, encouraging robustness to input perturbations.  
   - Hyperparameters (neurons per layer, L1 regularization strength, learning rate) are tuned via grid search with scikit‑learn wrappers.  

2. **Fine‑tuning on Observations**  
   - Pretrained models are frozen and partially retrained on real CARMENES spectra; latent representations and reconstruction MSEs are saved for later analysis.  

3. **CNN Regression**  
   - Latent vectors (size 32) from each autoencoder feed a small 1‑D CNN to predict stellar parameters (example: effective temperature). Multiple runs per autoencoder provide ensemble estimates.

### Getting Started
1. **Set up the environment**  
   - Install dependencies listed in `requirements.txt` and ensure TensorFlow 2.13/Keras 2.13 compatibility.  
2. **Prepare data arrays**  
   - Scripts expect NumPy arrays such as `phf_good_norm` (synthetic spectra), `php_good` (synthetic parameter labels), and `carmf_norm` (observed spectra). These must be loaded or generated prior to running.  
3. **Reproduce methodology**  
   - Run `autoencoder_grid_search.py` to find optimal hyperparameters.  
   - Train autoencoders with `autoencoder_training.py`, producing latent encodings.  
   - Fine‑tune selected models on CARMENES data via `autoencoder_fine_tuning.py`.  
   - Train CNN regressors (`cnn_training.py`) to map latent vectors to stellar parameters, then explore results in the provided notebooks.

### Pointers for Further Exploration
- **Understanding contractive loss**: Study the mathematical derivation of contractive autoencoders to see how the penalty term is formed and why the bottleneck weights are used.  
- **Extending to other parameters**: The CNN example targets effective temperature; adapt `build_cnn` hyperparameters and labels for log g, metallicity, and rotational velocity.  
- **Data comparison and visualization**: Dive into `Section4_1_figs.ipynb` and `Section4_2_figs.ipynb` to learn how latent representations and predictions are evaluated against literature data.  
- **Performance considerations**: Scripts may require substantial compute power; consider batching or distributed strategies when training on large spectral datasets.  
- **Code robustness**: Several scripts assume variables like `os`, `time`, `GridSearchCV`, `load_model`, or `K` are already imported. Ensure these dependencies are explicitly added in your run environment.
