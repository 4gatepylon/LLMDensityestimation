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
1. (1) Be able to collect activations on non-EOS tokens (i.e. realistic chat for both input and output) on:
    - Pliny prompts
    - Other adversarial prompts that make it through
    - Prompts that don't make it through
    - Ideally have a method that can generate some prompts using NanoGCG
    - Some normal prompts
    - (basically for each of these store TOKENS but don't worry about storing activations for now; make a method that can take in some strings or something like this and then just create this stuff... store it in a folder somewhere)
3. (2) Plot perplexity over tokens
4. (3) Have a function to find the refusal direction using the method from the paper, it is literally not even looking for refusal but just the last token: https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction. Ensure that we are able to reproduce refusing (this should be possible to rip from the paper).
5. (3) Plot refusal direction magnitude over time per-layer
6. Implement code to get gram matrices for sequences of activations (curious in d_model x d_model and n_seq x n_seq)
    - We can use the d_model x d_model ones to look for similarities
    - We can use the d_token x d_token one to plot using circuit-vis (ripped) how much correlation with prior inputs

Commit and then switch to being able to use obfuscated activations etc.... we will want to get mahalanobis gcg stuff etc... working fine