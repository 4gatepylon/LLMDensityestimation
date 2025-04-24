from pathlib import Path
import sys, numpy as np, torch
from sae_lens import SAE
import torch.nn as nn
import tqdm
import torch
import click
from transformers import AutoModel
from tansformer_lens import HookedTransformer

def encode_dir(input_dir: Path, output_dir_l2: Path, output_dir_kl: Path, device: str, layer_idx: int) -> None:
    with torch.no_grad():
        assert input_dir.exists(), f"Input directory {input_dir} does not exist"
        assert not output_dir_l2.exists(), f"Output directory {output_dir_l2} already exists"
        assert not output_dir_kl.exists(), f"Output directory {output_dir_kl} already exists"
        model = AutoModel.from_pretrained("gpt2")
        model_tl = HookedTransformer.from_pretrained("gpt2")
        # print(model) # Debug
        model_mlp = model.h[layer_idx].mlp
        model_mlp.to(torch.float32)
        model_mlp.to(device)
        model_mlp.eval()
        assert isinstance(model_mlp, nn.Module)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            "callummcdougall/arena-demos-transcoder",
            "gpt2-small-layer-8-mlp-transcoder-folded-b_dec_out")
        sae.to(torch.float32)
        sae.to(device)
        sae.eval()
        # print("="*100) # XXX
        # print(sae)
        # print("="*100)
        # print(cfg_dict)
        # print("="*100)
        # print(sparsity)
        # print("="*100)
        # print(model)
        # print("="*100) # XXX
        output_dir_l2.mkdir(parents=True, exist_ok=False)
        output_dir_kl.mkdir(parents=True, exist_ok=False)
        # NOTE to get histogram for these you should basically just get the 2-axis of the gpt2_...._pca10 and then you would
        # discretize them and then you would put the error there...
        for f in tqdm.tqdm(list(input_dir.glob("*.npy")), desc="Encoding"):
            rel_path = f.relative_to(input_dir)
            out_path_l2 = output_dir_l2 / rel_path
            out_path_kl = output_dir_kl / rel_path
            acts = torch.from_numpy(np.load(f)).float().to(device)
            # Transcode and take the MLP of these and then plot the error
            assert acts.ndim == 2, f"acts.shape={acts.shape} != 2D" # batch , d
            transcoded = sae(acts)
            assert transcoded.ndim == 2, f"transcoded.shape={transcoded.shape} != 2D" # batch , d
            mlped = model_mlp(acts) # NOTE Must be on ACTS and not transcoded since we will COMPARE
            assert mlped.ndim == 2, f"mlped.shape={mlped.shape} != 2D" # batch , d
            assert mlped.shape == acts.shape, f"mlped.shape={mlped.shape} != acts.shape={acts.shape}"
            err = (mlped - acts).abs().norm(dim=-1) # mean over the d dimension
            assert err.ndim == 1 and err.shape[0] == acts.shape[0], f"err.shape={err.shape} != acts.shape[0]={acts.shape[0]}"
            err_np = err.detach().cpu().numpy()
            np.save(out_path_l2, err_np)

            # Now run the forward pass up the the rest of the model
            # prev_res = model_tl() # TODO(Adriano) also save KL to the KL folder somehow....



# XXX also please make sure to plot a hisotgram of the values (log transformed)
# XXX finish this shit below please for histogram
def __plot_err_hist2d(err_dir: Path, pca_dir: Path, output_dir: Path):
    np_files_err = list(err_dir.glob("act_*.npy"))
    np_files_pca = list(pca_dir.glob("act_*.npy"))
    assert len(np_files_err) == len(np_files_pca), f"len(np_files_err)={len(np_files_err)} != len(np_files_pca)={len(np_files_pca)}" # fmt: skip
    assert set(x.name for x in np_files_err) == set(x.name for x in np_files_pca), f"Different file names" # fmt: skip
    np_files_err = sorted(np_files_err, key=lambda x: x.name)
    np_files_pca = sorted(np_files_pca, key=lambda x: x.name)
    # NOTE: this must be the SAME
    assert list(map(lambda x: x.name, np_files_err)) == list(map(lambda x: x.name, np_files_pca)), f"Different file names" # fmt: skip
    output_dir.mkdir(parents=True, exist_ok=False)

    # XXX do the histo stuff lmao
    for err_file, pca_file in zip(np_files_err, np_files_pca):
        rel_path = err_file.relative_to(err_dir)
        rel_path_sans = pca_file.relative_to(pca_dir)
        _ = rel_path.as_posix()
        __ = rel_path_sans.as_posix()
        assert _ == __, f"Different file names... err_file={_} != pca_file={__}" # fmt: skip
        out_path = output_dir / rel_path # Place where we will store
        err = np.load(err_file)
        pca = np.load(pca_file)
        # discretize the pca 
        # plot the error histogram
        raise NotImplementedError("Not implemented") # XXX
    pass

@click.group()
def cli():
    pass

@cli.command()
@click.option("--input_dir", "-i", type=str, help="Input directory")
@click.option("--output_dir", "-o-l2", type=str, help="Output directory")
@click.option("--output_dir", "-o-kl", type=str, help="Output directory")
@click.option("--device", "-d", default="cuda", type=str, help="Device") # If you cuda make sure to set CUDA_VISIBLE_DEVICES
@click.option("--layer_idx", "-l", default=7, type=int, help="Layer index")
def get_transcoder_errs(input_dir, output_dir_l2, output_dir_kl, device, layer_idx):
    print(f"input_dir={input_dir}")
    print(f"output_dir_l2={output_dir_l2}")
    print(f"output_dir_kl={output_dir_kl}")
    # print(f"device={device}")
    input_dir, output_dir_l2, output_dir_kl = Path(input_dir), Path(output_dir_l2), Path(output_dir_kl)
    assert not output_dir_l2.exists(), f"Output directory {output_dir_l2} already exists"
    assert not output_dir_kl.exists(), f"Output directory {output_dir_kl} already exists"
    output_dir_l2.mkdir(parents=True, exist_ok=False)
    output_dir_kl.mkdir(parents=True, exist_ok=False)
    encode_dir(input_dir, output_dir_l2, output_dir_kl, device, layer_idx)

@cli.command()
@click.option("--err_dir", "-e", type=str, help="Error directory")
@click.option("--pca_dir", "-p", type=str, help="PCA directory")
@click.option("--output_dir", "-o", type=str, help="output directory for the plots")
def plot_err_hist2d(err_dir, pca_dir, output_dir):
    err_dir, pca_dir, output_dir = Path(err_dir), Path(pca_dir), Path(output_dir)
    assert err_dir.exists(), f"Error directory {err_dir} does not exist"
    assert pca_dir.exists(), f"PCA directory {pca_dir} does not exist"
    assert not output_dir.exists(), f"Output directory {output_dir} already exists"
    __plot_err_hist2d(err_dir, pca_dir, output_dir)

if __name__ == "__main__":
    cli()
