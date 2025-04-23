import pytest
import torch

from obf_reps.metrics import ObfMetric
from obf_reps.optimize.loss import BehaviorLoss, LossFunctionBase


class DummyObfMetric(ObfMetric):
    def evaluate(self, reps):
        b_size, layers, seq_len, hidden_dim = reps.shape
        assert isinstance(b_size, int)

        return torch.ones(b_size, layers, seq_len, hidden_dim)


@pytest.fixture
def loss_inputs():
    batch_size = 2
    beh_seq_len = 5
    rep_seq_len = 45
    vocab_size = 1000
    num_layers = 3
    hidden_dim = 10
    """
        behavior_logits: Float[Tensor, "b_size beh_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_seq_len"],
        reps: Float[Tensor, "b_size layers rep_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_seq_len"],
    """

    return {
        "behavior_logits": torch.rand(batch_size, beh_seq_len, vocab_size),
        "behavior_target": torch.randint(vocab_size, (batch_size, beh_seq_len)),
        "reps": torch.rand(batch_size, num_layers, rep_seq_len, hidden_dim),
        "behavior_loss_mask": torch.ones(batch_size, beh_seq_len, dtype=torch.bool),
        "rep_loss_mask": torch.ones(batch_size, rep_seq_len, dtype=torch.bool),
    }


@pytest.fixture
def obf_metric():
    return DummyObfMetric()


def test_behavior_loss(loss_inputs, obf_metric):
    loss_fn = BehaviorLoss(obf_metric)
    loss = loss_fn.compute_loss(**loss_inputs)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
