from __future__ import annotations
from pathlib import Path
import shutil
import re
from typing import Literal, Tuple, List, Dict, Optional
import click
import tqdm
import pydantic
import itertools
import json
"""
This module is a makeshift filetree parser+html generator for a static HTML
website that we can use to easily display/view/compare the different SAE
results. It allows for two forms of the static HTML generation:
1. A form where the images are loaded from the local directory
2. A form where the images are loaded from AWS S3 in an analogous location
    (basically the root folder is not <folder> but instead logically equivalent to s3://<bucket>/<folder> which
    in HTTP form resolves to: `https://gpt2-pro-website-images.s3.us-east-2.amazonaws.com/<folder>/...`
    where we use AWS S3 name `gpt2-pro-website-images` (look below).

NOTE if you want to use S3 the expectation is thatt you will have a local folder where you will
do your analysis (passed as root) and then you will use something like
`aws s3 sync <local-folder> s3://<bucketname>/<folder>` to sync the images to S3,
and you will pass the correct url and the use_s3 flag to the constructor of the
WebsiteCompiler class.
"""

class HookpointParser:
    res_hp_patt = r"^blocks\.(\d+)\.hook_resid_pre$"
    ln2_hp_patt = r"^blocks\.(\d+)\.ln2.hook_scale$"
        
    @classmethod
    def is_supported_hookpoint(cls, hookpoint: str) -> bool:
        return re.match(cls.res_hp_patt, hookpoint) is not None or re.match(cls.ln2_hp_patt, hookpoint) is not None

    @classmethod
    def type_layer2hookpoint(cls, layer: int, type: Literal["res", "ln2"]) -> str:
        if type == "res":
            return f"blocks.{layer}.hook_resid_pre"
        elif type == "ln2":
            return f"blocks.{layer}.ln2.hook_scale"
        else:
            raise ValueError(f"Invalid type: {type}")

    @classmethod
    def hookpoint2type_layer(cls, hookpoint: str) -> Tuple[int, Literal["res", "ln2"]]:
        if not cls.is_supported_hookpoint(hookpoint):
            raise ValueError(f"Hookpoint {hookpoint} is not supported")
        if re.match(cls.res_hp_patt, hookpoint):
            layer = int(re.match(cls.res_hp_patt, hookpoint).group(1))
            return layer, "res"
        elif re.match(cls.ln2_hp_patt, hookpoint):
            layer = int(re.match(cls.ln2_hp_patt, hookpoint).group(1))
            return layer, "ln2"
        else:
            raise ValueError(f"Hookpoint {hookpoint} is not supported")

class FileTreeMap(pydantic.BaseModel):
    # Root paths/urls/etc..
    local_root_path: str
    s3_root_url: Optional[str] = None
    # Things that are available to visualize
    pcs: List[int]
    k_values: List[int | str]
    # NOTE: hookpoint_layer_values is things like
    # `blocks.1.hook_resid_pre` or `blocks.1.ln2.hook_scale
    # and are guaranteed to be sorted by layer
    hookpoint_layer_values: List[str]
    error_types: List[str]
    # Maps
    local_canonical_path2real_path: Dict[str, str]
    s3_canonical_path2real_path: Dict[str, str]
    rel_canonical_path2real_path: Dict[str, str]

class FiletreeParser:
    """
    Filetree parsing happens in 3 steps:
    1. Index all the available combinations that we will need:
        - What PC pairs are available to visualize?
        - What no-SAE and k values are available to visualize?
        - What hookpoints and layers are available to visualize?
            (note technically this one is hardcoded)
        - What error types (or histogram) are available to visualize?
            (note this is also hardcoded)
    2. Generate a map from all combinations of the above to the actual paths that
        should contain these. Assert that all these actually exist. Then, return
        a pydantic schema that can be saved to a json file to map the findings.

    """
    # patterns used in various places
    pc_patt = r"^pc(\d+)_pc(\d+)\.png$" # For files
    ends_with_k_patt = r"^.*_k_(\d+)$" # For folders

    # Special names that determine things other than k-sweep
    no_sae_name: str = "no-sae"
    histogram_density_error_type: str = "histogram"
    # Other useful constants
    s3_root_url: str = "gpt2.pro.s3.us-east-2.amazonaws.com"

    @classmethod
    def __get_pngs(cls, root: str | Path) -> List[Path]:
        root = Path(root)
        pngs = set(list(root.glob("**/*.png")) + list(root.glob("*.png")))
        pngs = [p for p in pngs if re.match(cls.pc_patt, p.name)]
        return pngs

    @classmethod
    def find_available_pc_pairs(cls, root: str | Path, get_pairs: bool = True) -> List[Tuple[int, int]] | List[int]: # fmt: skip
        pngs = cls.__get_pngs(root)
        # print(pngs) # DEBUG
        pairs = set(
            (int(re.match(cls.pc_patt, p.name).group(1)), int(re.match(cls.pc_patt, p.name).group(2))) # fmt: skip
            for p in tqdm.tqdm(pngs, desc="Finding available PCs")
        )
        pairs_str = "\n".join(sorted(f"pc{p1}_pc{p2}.png" for p1, p2 in pairs))
        # print("PAIRS ARE", pairs)
        parent_folders = set(p.parent for p in pngs)
        # print(parent_folders)
        for p in tqdm.tqdm(parent_folders, desc="Checking Parents for PCs"):
            children_str = "\n".join(f.name for f in p.iterdir())
            assert all(
                (p / f"pc{p1}_pc{p2}.png").exists() or (p / f"pc{p2}_pc{p1}.png").exists()
                for p1, p2 in pairs
            ), f"Missing PCs in {p.as_posix()}... Target:\n\n{pairs_str}\n\nChildren:\n\n{children_str}" # fmt: skip
        if get_pairs:
            return pairs
        else:
            pcs_left = set(p1 for p1, p2 in pairs)
            pcs_right = set(p2 for p1, p2 in pairs)
            pcs = list(sorted(pcs_left | pcs_right))
            for parent in tqdm.tqdm(parent_folders, desc="Checking Parents for PCs"):
                children_str = "\n".join(f.name for f in parent.iterdir())
                for p1, p2 in itertools.combinations(pcs, 2):
                    assert (
                        (parent / f"pc{p1}_pc{p2}.png").exists() or
                        (parent / f"pc{p2}_pc{p1}.png").exists()
                    ), f"Missing PCs in {parent.as_posix()}... Target:\n\n{pairs_str}\n\nChildren:\n\n{children_str}" # fmt: skip
            return pcs
    
    @classmethod
    def find_available_ks(cls, root: str | Path) -> List[int | str]:
        # NOTE: this is not going to look at all valid places that have ks
        # instead it will look only at what should be a sufficient subset to find all
        # the ks; later down the line, the real paths are generated and there the correct
        # versions (sometimes the letter `k` is not there) are used.
        pngs = cls.__get_pngs(root)
        parents = set(p.parent for p in pngs)
        parents = [p for p in tqdm.tqdm(parents, desc="Finding available Ks") if re.match(cls.ends_with_k_patt, p.name)] # fmt: skip
        k_values = set(int(re.match(cls.ends_with_k_patt, p.name).group(1)) for p in parents)
        assert len(k_values) > 0, f"No Ks found\n{'\n'.join(p.as_posix() for p in parents)}"
        return list(sorted(k_values)) + [cls.no_sae_name]
    
    @classmethod
    def find_available_layer_hookpoints(cls, root: str | Path) -> List[int | str]:
        resids = [f"blocks.{i}.hook_resid_pre" for i in range(0, 12)]
        ln2s = [f"blocks.{i}.ln2.hook_scale" for i in range(0, 12)]
        # Return resids interleaved with ln2s and resids come first and layers
        # take precedence over hook name
        assert len(resids) == len(ln2s), "Resids and ln2s must have the same length"
        # 0 => 0, 1 => 0, 2 => 1, 3 => 1, etc...
        return [resids[i // 2] if i % 2 == 0 else ln2s[i // 2] for i in range(len(resids) + len(ln2s))] # fmt: skip

    @classmethod
    def find_available_error_types(cls, root: str | Path) -> List[str]:
        return [
            "normalized_error_norm",
            "normalized_mse",
            "normalized_variance_explained",
            "unnormalized_error_norm",
            "unnormalized_mse",
            "unnormalized_variance_explained",
            # NOTE: density means the values you proejct and accumulate are "+1" i.e.
            # density
            cls.histogram_density_error_type
        ]
    @classmethod
    def __combo2relative_path(cls, pc_pair: Tuple[int, int], k: int | str, layer_hookpoint: str, error_type: str) -> str: # fmt: skip
        """
        Relative path is a relative path from your root (S3 or local) to the image you care about.

        Unfortunately this is a little complicated right now because the folder structure is complicated
        for no good reason (smh) in the real folder, but we will fix this later.
        """
        assert HookpointParser.is_supported_hookpoint(layer_hookpoint), f"Hookpoint {layer_hookpoint} is not supported" # fmt: skip
        hookpoint_layer, hookpoint_type = HookpointParser.hookpoint2type_layer(layer_hookpoint) # fmt: skip
        
        #### 1. Find the subroot ####
        # Subtroot header + subroot footer determines the subroot: first folder to
        # go into
        subroot_hookpoint_header = f"{hookpoint_type}_" + (
            "sae_in_" if k == cls.no_sae_name and hookpoint_type == "res" else 
            "sae_out_" if k != cls.no_sae_name and hookpoint_type == "res" else
            "" if k == cls.no_sae_name and hookpoint_type == "ln2" else
            "sae_effect_" if k != cls.no_sae_name and hookpoint_type == "ln2" else
            ValueError(f"Invalid k: {k} and hookpoint type: {hookpoint_type}")
        )
        subroot_error_footer = ("" if error_type == cls.histogram_density_error_type else f"_err") + "_pca_histograms" # fmt: skip
        subroot = subroot_hookpoint_header + subroot_error_footer
        subroot = subroot.replace("__", "_")
        #### 2. Find the layer ####
        layer_folder = (
            f"layer_{hookpoint_layer}" if k == cls.no_sae_name else(
                # No error means loud k
                f"layer_{hookpoint_layer}_k_{k}" if error_type == cls.histogram_density_error_type else # fmt: skip
                # Error means silent k
                f"layer_{hookpoint_layer}_{k}" if error_type != cls.histogram_density_error_type else # fmt: skip
                ValueError(f"Invalid error_type: {error_type}")
            )
        )
        #### 3. Find the error type folder (no folder if histogram density) ####
        error_type_folder = (
            "" if error_type == cls.histogram_density_error_type else # fmt: skip
            f"{error_type}_k_{k}" if error_type != cls.histogram_density_error_type and k != cls.no_sae_name else # fmt: skip
            # NOTE: it is not possible to have a k no-sae because then there is no error to plot
            ValueError(f"Invalid error_type: {error_type} and k: {k}")
        )
        #### 4. Find the total folder path and fix any // in it ####
        assert not subroot.startswith("_") and not subroot.endswith("_"), "Subroot should not start or end with an underscore" # fmt: skip
        assert not layer_folder.startswith("_") and not layer_folder.endswith("_"), "Layer folder should not start or end with an underscore" # fmt: skip
        assert not error_type_folder.startswith("_") and not error_type_folder.endswith("_"), "Error type folder should not start or end with an underscore" # fmt: skip
        assert "/" not in subroot, f"Subroot should not contain a slash: {subroot}"
        assert "/" not in layer_folder, f"Layer folder should not contain a slash: {layer_folder}" # fmt: skip
        assert "/" not in error_type_folder, f"Error type folder should not contain a slash: {error_type_folder}" # fmt: skip
        total_folder_path = "/".join([subroot, layer_folder, error_type_folder])
        total_folder_path = total_folder_path.replace("//", "/") # Sometimes empty if no error subfolder
        if total_folder_path.endswith("/"):
            total_folder_path = total_folder_path[:-1]
        assert "//" not in total_folder_path, f"Total folder path should not contain a double slash: {total_folder_path}" # fmt: skip

        #### 5. Find the image name ####
        image_name = f"pc{pc_pair[0]}_pc{pc_pair[1]}.png"

        #### 6. Return the final path ####
        full_path = total_folder_path + "/" + image_name
        assert "//" not in full_path, f"Full path should not contain a double slash: {full_path}" # fmt: skip
        assert not "__" in full_path, f"Full path should not contain a double underscore: {full_path}" # fmt: skip
        return full_path
    
    @classmethod
    def __combo2canonical_path(cls, pc_pair: Tuple[int, int], k: int | str, layer_hookpoint: str, error_type: str) -> str: # fmt: skip
        """Canonical path is something you use given known combinations to find a path you care about."""
        return f"pc{pc_pair[0]}_pc{pc_pair[1]}:k_{k}:{layer_hookpoint}:{error_type}"
    
    @classmethod
    def __combo2real_path(cls, root: str | Path, pc_pair: Tuple[int, int], k: int | str, layer_hookpoint: str, error_type: str, using_s3: bool, relative_path: bool = False) -> str: # fmt: skip
        """HTML path is something like file://<real path> or <url> and points to an image."""
        relative_path = cls.__combo2relative_path(pc_pair, k, layer_hookpoint, error_type)
        real_path_obj = root / relative_path
        if not real_path_obj.exists():
            relative_path = cls.__combo2relative_path(tuple(reversed(pc_pair)), k, layer_hookpoint, error_type)
            real_path_obj = root / relative_path
        assert real_path_obj.exists(), f"Real path {real_path_obj} does not exist"
        return (
            # NOTE: in s3 we group by root.name
            f"https://{cls.s3_root_url}/{root.name}/{relative_path}" if using_s3 else 
            f"file://{real_path_obj.resolve().as_posix()}" if not relative_path else
            # NOTE your root must have a parent: we will copy root into the output folder
            f"file://{real_path_obj.relative_to(root.parent).as_posix()}"
        )

    @classmethod
    def create_canonical_map(cls, root: str | Path) -> FileTreeMap:
        """
        Create a FileTreeMap and ensure that all the necessary paths exist. This depends
        on a local root directory (that should be replicated in S3) and it will return
        something that can be used both in local and S3 mode since all the filemaps
        are provided.
        """
        # Get all the available things
        avail_pc_pairs = cls.find_available_pc_pairs(root, get_pairs=True) # fmt: skip
        avail_ks = cls.find_available_ks(root)
        avail_layer_hookpoints = cls.find_available_layer_hookpoints(root) # fmt: skip
        avail_error_types = cls.find_available_error_types(root)

        local_map: Dict[str, str] = {}
        s3_map: Dict[str, str] = {}
        rel_map: Dict[str, str] = {}
        for pc_pair, k, layer_hookpoint, error_type in tqdm.tqdm(
            itertools.product(avail_pc_pairs, avail_ks, avail_layer_hookpoints, avail_error_types), # fmt: skip
            desc="Creating canonical map",
            total=len(avail_pc_pairs) * len(avail_ks) * len(avail_layer_hookpoints) * len(avail_error_types) # fmt: skip
        ):
            # NOTE: it is simply not valid to look at the "error" for no-sae because error
            # is always w.r.t. no-sae
            if k == cls.no_sae_name and error_type != cls.histogram_density_error_type:
                continue
            canonical_path = cls.__combo2canonical_path(pc_pair, k, layer_hookpoint, error_type) # fmt: skip
            realpath = cls.__combo2real_path(root, pc_pair, k, layer_hookpoint, error_type, using_s3=False, relative_path=False) # fmt: skip
            s3_path = cls.__combo2real_path(root, pc_pair, k, layer_hookpoint, error_type, using_s3=True) # fmt: skip
            relpath = cls.__combo2real_path(root, pc_pair, k, layer_hookpoint, error_type, using_s3=False, relative_path=True) # fmt: skip
            local_map[canonical_path] = realpath
            s3_map[canonical_path] = s3_path
            rel_map[canonical_path] = relpath
            
        # Find the needed pcs
        pcs_left = set(p1 for p1, p2 in avail_pc_pairs)
        pcs_right = set(p2 for p1, p2 in avail_pc_pairs)
        avail_pcs = list(sorted(pcs_left | pcs_right))
        # Format and return
        return FileTreeMap(
            # Root paths/urls/etc..
            local_root_path=Path(root).resolve().as_posix(),
            s3_root_url=f"https://{cls.s3_root_url}",
            # Available things
            pcs=avail_pcs,
            k_values=avail_ks,
            hookpoint_layer_values=avail_layer_hookpoints,
            error_types=avail_error_types,
            # Maps
            local_canonical_path2real_path=local_map,
            s3_canonical_path2real_path=s3_map,
            rel_canonical_path2real_path=rel_map
        )

#### DEBUG version of main ####
# if __name__ == "__main__":
#     root = Path("sae_sans_old_plots_v2")
#     # root = Path("sae_sans_plots")
#     avail_pc_pairs = FiletreeParser.find_available_pc_pairs(root, get_pairs=True)
#     print("\n".join(f"{p1} {p2}" for p1, p2 in avail_pc_pairs))
#     print("=" * 100)

#     avail_pcs = FiletreeParser.find_available_pc_pairs(root, get_pairs=False)
#     print("\n".join(f"{p}" for p in avail_pcs))
#     print("=" * 100)

#     avail_ks = FiletreeParser.find_available_ks(root)
#     print("\n".join(f"{k}" for k in avail_ks))
#     print("=" * 100)

#     avail_layer_hookpoints = FiletreeParser.find_available_layer_hookpoints(root)
#     print("\n".join(avail_layer_hookpoints))
#     print("=" * 100)

#     avail_error_types = FiletreeParser.find_available_error_types(root)
#     print("\n".join(avail_error_types))
#     print("=" * 100)

#     filetree_map = FiletreeParser.create_canonical_map(root)
#     print(filetree_map.model_dump_json(indent=2)) # DEBUG

class WebsiteCompiler:
    template_html_file = Path(__file__).parent / "sans_sae_viz_template.html"
    # This will be filled in by the template engine
    template_html_fillin_k_values: str = "K_VALUES_2281C139-3A94-43E8-8556-50055385279"
    template_html_fillin_hookpoint_layer_values: str = "HOOKPOINT_LAYER_VALUES_5E00BA9F-4CAE-412D-9DC1-4EF643EBE2E2"
    template_html_fillin_error_types: str = "ERROR_TYPES_CB2547A2-7C67-4D5C-8059-DCDA565B746E"
    template_html_fillin_local_canonical_path2real_path: str = "LOCAL_CANONICAL_PATH2_REAL_PATH_C3F8C34C-0570-496B-B459-1026E4E50C7E"

    def __init__(self, root_folder: str | Path, use_s3: bool = True, output_folder: str | Path = "output"):
        """Pass use_s3=True vs. False to decide whether the generated HTML/JS will load from S3 or local."""
        self.s3_url = FiletreeParser.s3_root_url
        self.use_s3 = use_s3
        self.root_folder = Path(root_folder)
        self.output_folder = Path(output_folder)
        if self.output_folder.exists():
            if len([f for f in self.output_folder.glob("**/*") if f.is_file()]) > 0 or len([f for f in self.output_folder.iterdir() if f.is_file()]) > 0: # fmt: skip
                raise ValueError(f"Output folder {self.output_folder} already exists")
            else:
                shutil.rmtree(self.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=False)
        self.output_filetree_mapfile = self.output_folder / "filetree_map.json"
        self.copied_images_folder = self.output_folder / self.root_folder.name

        # NOTE: we will include all the JS here too eh...
        self.index_html_file = self.output_folder / "index.html"

    def compile_index_html(self, filetree_map: FileTreeMap):
        # 0. Read the template
        template = self.template_html_file.read_text()

        # 1. Ensure this can be filled in OK
        assert self.template_html_fillin_k_values in template, f"K_VALUES_2281C139-3A94-43E8-8556-50055385279 not found in template" # fmt: skip
        assert self.template_html_fillin_hookpoint_layer_values in template, f"HOOKPOINT_LAYER_VALUES_5E00BA9F-4CAE-412D-9DC1-4EF643EBE2E2 not found in template" # fmt: skip
        assert self.template_html_fillin_error_types in template, f"ERROR_TYPES_CB2547A2-7C67-4D5C-8059-DCDA565B746E not found in template" # fmt: skip
        assert self.template_html_fillin_local_canonical_path2real_path in template, f"LOCAL_CANONICAL_PATH2_REAL_PATH_C3F8C34C-0570-496B-B459-1026E4E50C7E not found in template" # fmt: skip
        
        # 2. Extract the values
        k_values = filetree_map.k_values
        hookpoint_layer_values = filetree_map.hookpoint_layer_values
        error_types = filetree_map.error_types
        local_canonical_path2real_path = filetree_map.s3_canonical_path2real_path if self.use_s3 else filetree_map.local_canonical_path2real_path # fmt: skip
        
        # 3. Turn these into strings
        k_values_str = json.dumps(k_values) # one line
        hookpoint_layer_values_str = json.dumps(hookpoint_layer_values) # one line
        error_types_str = json.dumps(error_types) # one line
        local_canonical_path2real_path_str = json.dumps(local_canonical_path2real_path) # one line
        assert isinstance(k_values_str, str) and isinstance(hookpoint_layer_values_str, str) and isinstance(error_types_str, str) and isinstance(local_canonical_path2real_path_str, str), "All values must be strings" # fmt: skip
        
        # 4. Fill in the template
        template = template.replace(self.template_html_fillin_k_values, k_values_str)
        template = template.replace(self.template_html_fillin_hookpoint_layer_values, hookpoint_layer_values_str)
        template = template.replace(self.template_html_fillin_error_types, error_types_str)
        template = template.replace(self.template_html_fillin_local_canonical_path2real_path, local_canonical_path2real_path_str)
        return template

    def compile(self):
        #### 1. Copy the images ####
        shutil.copytree(self.root_folder, self.copied_images_folder)
        #### 2. Generate the filetree map ####
        # NOTE: use the relative path version since it'll probably be a bit safer...
        filetree_map = FiletreeParser.create_canonical_map(self.copied_images_folder)
        Path(self.output_filetree_mapfile).write_text(filetree_map.model_dump_json(indent=2))
        #### 3. Generate the HTML ####
        # NOTE: this is a little complicated because we need to have a different
        # template for the no-sae case and the k-sweep case.
        #### 4. Copy the HTML to the output folder ####
        self.index_html_file.write_text(self.compile_index_html(filetree_map))
        print(f"Compiled index.html to {self.index_html_file.as_posix()}")

@click.command()
@click.option("--root-folder", "-r", "-i", type=str, help="The root folder to use for the website")
@click.option("--use-s3", "-s3", is_flag=True, help="Whether to use S3 for the website")
@click.option("--output-dir", "-o", type=str, help="The output directory to place the index.html and image_map.json files")
def main(root_folder: str, use_s3: bool, output_dir: str):
    compiler = WebsiteCompiler(root_folder, use_s3, output_dir)
    compiler.compile()

if __name__ == "__main__":
    main()
