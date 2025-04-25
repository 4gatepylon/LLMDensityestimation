# LLMDensityestimation
A toolkit to estimate the density of LLM activations based on activations. This can help do things like anomaly detection and generally pushes forward the field of interpretability.

# Guide to this codebase
The original PCA etc... EDA code is availabie in `pca_finder.py` which (somewhat slowly) finds PCA decompositions and stores them (along with atactivations) for (mostly) openwebtext data on gpt2. It operates on one layer. The usage is that you run it (read for arguments) and then you use `density_plot.py` to get some nice heatmaps (2d histograms) of the density on pairs of PCs. Some old examples are in `heatmaps` (layer 8) and `heatmaps_copy8` (layer 8).

`normal_flow_sandbox.py` and `sae_plot.py` are not really used but can be used to test normalizing flow models and/or test some SAEs... (in some way?)

You can use the `sans_sae.ipynb` and `sans_sae.py` notebook and script (which leverage code from `sans_sae_lib/`) to generate SAE and non-SAE PCAs for multiple layers. These will be stored in `sae_sans_plots`.

For more documentation on the SAE vs. no-SAE stuff, please consult the `sae_sans*` code snippets.