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


def test_sparse_projector_matches_view_loop():
    # The CSR path is used on CUDA; its matrices can be checked on CPU.
    torch.manual_seed(1)
    operator = ParallelBeamCT((12, 16), num_views=7)
    matrix, matrix_t = operator._sparse_matrices(torch.device("cpu"))
    image = torch.randn(3, 1, 12, 16)
    sinogram = torch.randn(3, 1, 7, operator.num_detectors)
    sparse_forward = torch.sparse.mm(matrix, image.reshape(3, -1).t()).t() * operator.projection_scale
    sparse_adjoint = torch.sparse.mm(matrix_t, sinogram.reshape(3, -1).t()).t() * operator.projection_scale
    torch.testing.assert_close(sparse_forward.reshape(sinogram.shape), operator.forward(image),
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sparse_adjoint.reshape(image.shape), operator.adjoint(sinogram),
                               rtol=1e-5, atol=1e-5)
