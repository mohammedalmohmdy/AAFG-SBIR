import torch
from src.models.aafg_sbir import AAFG_SBiR

def test_forward_shapes():
    model = AAFG_SBiR()
    sk = torch.randn(2,3,224,224)
    im = torch.randn(2,3,224,224)
    zs, zi = model(sk, im)
    assert zs.shape == (2,512) and zi.shape == (2,512)
    assert torch.allclose(zs.norm(dim=1), torch.ones(2), atol=1e-4)
    assert torch.allclose(zi.norm(dim=1), torch.ones(2), atol=1e-4)
    print('forward ok')

if __name__ == '__main__':
    test_forward_shapes()
