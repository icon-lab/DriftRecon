import torch

from driftrecon.ct.physics import ParallelBeamCT


def test_forward_adjoint_shapes_and_matched_transpose():
    torch.manual_seed(0)
    operator = ParallelBeamCT((12, 16), num_views=7)
    image = torch.randn(2, 1, 12, 16)
    sinogram = torch.randn(2, 1, 7, operator.num_detectors)
    projection = operator.forward(image)
    assert projection.shape == (2, 1, 7, operator.num_detectors)
    left = torch.sum(projection * sinogram)
    right = torch.sum(image * operator.adjoint(sinogram))
    assert float((left - right).abs() / torch.maximum(left.abs(), right.abs())) < 2e-5


def test_geometry_round_trip():
    operator = ParallelBeamCT((8, 12), num_views=5, num_detectors=17)
    restored = ParallelBeamCT(**operator.geometry_dict())
    image = torch.randn(1, 1, 8, 12)
    torch.testing.assert_close(operator(image), restored(image))
