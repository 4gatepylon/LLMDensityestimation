# NOTE PLEASE IGNORE THIS FILE
# DO NOT DELETE IT DO NOT READ IT DO NOT USE IT
# IT IS FOR REFERENCE ONLY FOR FUTURE REFACTORING
# AS WE ADD MORE AND MORE FEATURES TO THE PROJECT

# # Activations database...
# # XXX please start using activation serialization here
# class Activations(pydantic.BaseModel):
#     ################ DATA ################
#     # Activations are our basic unit of analysis. They must include a set of ACTIVATION numpy arrays that represent the
#     # layer of activations from a model
#     tokens: Int[torch.Tensor, "batch"]
#     activations: Float[torch.Tensor, "batch token"]

#     ################ SPECIAL + METADATA ################
#     # Sequence and batch for now are SPECIAL tags that we basically use to be able to match activations across layers and transformations
#     seq_idxs: Int[torch.Tensor, "batch"]
#     batch_idxs: Int[torch.Tensor, "batch"]
#     metadata: ActivationsMetadata

#     ################ TAGS ################
#     # Tags are basically boolean masks that define whether or not a token is tagged with a certain tag.
#     # In the future we will support sparser data structure but eh...
#     bool_tags: dict[str, Bool[torch.Tensor, "batch"]] = {}
#     int_tags: dict[str, Int[torch.Tensor, "batch"]] = {}
#     # NOTE: we do not support float tags yet but they could be hypothetically supported in the future
#     float_tags: dict[str, Float[torch.Tensor, "batch"]] = {}

#     @staticmethod
#     def from_path(path: Path) -> Activations:
#         raise NotImplementedError("Not implemented")

# class ActivationsMetadata(pydantic.BaseModel):
#     model: str
#     hook_point: str
#     dataset_name: str

# class Tagger(abc.ABC):
#     def __init__(self, name: str):
#         self.name = name

#     @abc.abstractmethod
#     def tag(
#         self, activations: Float[torch.Tensor, "batch token"], tokens: np.ndarray, current_tags: dict[str, np.ndarray]) -> bool:
#         pass

# class BatchIdxTag(Tagger):
#     pass
# class SeqIdxTag(Tagger):
#     pass