import pytest
import torch

from src.model import CaModule
from src.utils import insert_cellular_padding
from src.ca_simulator import Simulator


inputs = [
    torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]
    ]),
    torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.]
    ]),
]

results_size1 = [
    torch.tensor([
        [9., 7., 8., 9., 7.],
        [3., 1., 2., 3., 1.],
        [6., 4., 5., 6., 4.],
        [9., 7., 8., 9., 7.],
        [3., 1., 2., 3., 1.],
    ]),
    torch.tensor([
        [16., 13., 14., 15., 16., 13.],
        [4., 1., 2., 3., 4., 1.],
        [8., 5., 6., 7., 8., 5.],
        [12., 9., 10., 11., 12., 9.],
        [16., 13., 14., 15., 16., 13.],
        [4., 1., 2., 3., 4., 1.],
    ]),
]
results_size2 = [
    torch.tensor([
        [5., 6., 4., 5., 6., 4., 5.],
        [8., 9., 7., 8., 9., 7., 8.],
        [2., 3., 1., 2., 3., 1., 2.],
        [5., 6., 4., 5., 6., 4., 5.],
        [8., 9., 7., 8., 9., 7., 8.],
        [2., 3., 1., 2., 3., 1., 2.],
        [5., 6., 4., 5., 6., 4., 5.],
    ]),
    torch.tensor([
        [11., 12., 9., 10., 11., 12., 9., 10.],
        [15., 16., 13., 14., 15., 16., 13., 14.],
        [3., 4., 1., 2., 3., 4., 1., 2.],
        [7., 8., 5., 6., 7., 8., 5., 6.],
        [11., 12., 9., 10., 11., 12., 9., 10.],
        [15., 16., 13., 14., 15., 16., 13., 14.],
        [3., 4., 1., 2., 3., 4., 1., 2.],
        [7., 8., 5., 6., 7., 8., 5., 6.],
    ]),
]

inputs1 = list(zip(inputs, [1, 1], results_size1))
inputs2 = list(zip(inputs, [2, 2], results_size2))


@pytest.mark.parametrize("matrix, size, expected", inputs1 + inputs2)
def test_padding(matrix: torch.Tensor, size: int, expected: torch.Tensor):
    expected: torch.Tensor = expected.view((1, 1, expected.shape[0], -1))
    result = insert_cellular_padding(matrix.view((1, 1, matrix.shape[0], -1)), size)
    assert result.shape == expected.shape
    assert torch.all(result == expected)


ca_inputs = [
    torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ]),
    torch.tensor([
        [1., 1., 0.],
        [1., 1., 0.],
        [0., 0., 0.]
    ]),
    torch.tensor([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 1.],
        [0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1., 0.],
    ]),
    torch.tensor([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 1.],
        [0., 0., 0., 0., 0., 1., 1.],
    ]),
    torch.tensor([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 1.],
    ])
]

ca_expectations = [
    torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ]),
    torch.tensor([
        [1., 1., 0.],
        [1., 1., 0.],
        [0., 0., 0.]
    ]),
    torch.tensor([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 1.],
        [0., 0., 0., 0., 0., 1., 1.],
    ]),
    torch.tensor([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 1.],
    ]),
    torch.tensor([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 1., 1.],
    ])
]


@pytest.mark.parametrize("prev_state, expected_state", zip(ca_inputs, ca_expectations))
def test_game_of_live(prev_state, expected_state):
    prev_state: torch.Tensor = prev_state.view((1, 1, prev_state.shape[0], -1))
    expected_state: torch.Tensor = expected_state.view((1, 1, expected_state.shape[0], -1))
    sim = Simulator((2, 4), (1, 4), 1)
    result = sim.get_next_state(prev_state)
    assert torch.all(result == expected_state)


@pytest.mark.parametrize("shape", [(10, 1, 12, 12), (1, 1, 20, 20)])
def test_sanity_model(shape):
    inputs = torch.zeros(shape)
    model = CaModule()
    results = model(inputs)
    assert inputs.shape == results.shape
