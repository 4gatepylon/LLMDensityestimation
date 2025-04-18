# LLMDensityestimation
A toolkit to estimate the density of LLM activations based on activations. This can help do things like anomaly detection and generally pushes forward the field of interpretability.

Please set up ideally with conda:
```
conda create -n llm-density python=3.12 -y && \
    conda activate llm-density && \
    pip3 install -r requirements.txt
```

Then you can run the `demo.ipynb` notebook to try (1) calculating thet relative loss in KL and variance explained by performing PCA on activations from GPT-2 (which is our initial testing ground for this stuff), which we use to pick an optimal PCA under a KL constraint (a caveat here is that we are basing this on a more limited dataset than our full dataset), (2) train a simple normalizing flow model on these on data from i.e. webtext*, (3) visualize for a set of inputs from (a) webtext, (b) custom datasets (tbd) the logprobs. This last demo basically is meant to demonstrate the possibility for us to do anomaly detection using the normalizing flow model (only one of the many applications of this system... I think).

_* GPT-2 should have been trained on webtext FYI: https://huggingface.co/datasets/Skylion007/openwebtext_

TODOs:
1. Do the (1) PCA experiments above
2. Write the model architecture for a simple normalizing flow model
3. Write the training loop (2)
4. Do the anomaly detection experiment (i.e. (a) create the out of domain datasets, (b) get some nice plots)
5. Create a pip package that we can be modifying after `pip install -e .` here...