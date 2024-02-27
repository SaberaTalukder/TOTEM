import torch

def pearsoncor(
    x: torch.Tensor, y: torch.Tensor, reduction: str = "mean", eps: float = 1e-4
) -> torch.Tensor:
    """
    Args:
        x: tensor of shape (B, T, S)
        y: tensor of shape (B, T, S)
        reduction: str of "mean" or "sum"
    Returns:
        corr: tensor of shape (1,)
    """

    mux = x.mean(dim=1, keepdim=True)
    muy = y.mean(dim=1, keepdim=True)

    u = torch.sum((x - mux) * (y - muy), dim=1)  # (B, S)
    d = torch.sqrt(
        torch.sum((x - mux) ** 2, dim=1) * torch.sum((y - muy) ** 2, dim=1)
    )  # (B, S)

    corr = u / (d + eps)  # (B, S)

    if reduction == "sum":
        corr = corr.sum()
    elif reduction == "mean":
        corr = corr.mean()
    else:
        raise ValueError("Uknown reduction mode")
    return corr


def test_pearsoncorr():
    import scipy

    B, T, S = 2, 16, 12
    x = torch.randn((B, T, S))
    y = torch.randn((B, T, S))

    our_corr = pearsoncor(x, y)

    # scipy corr is not vectorized
    x_np = x.numpy()
    y_np = y.numpy()
    scipy_corr = []
    for b in range(B):
        for s in range(S):
            cc = scipy.stats.pearsonr(x_np[b, :, s], y_np[b, :, s])
            scipy_corr.append(cc.statistic)
    scipy_corr = sum(scipy_corr) / B / S  # the mean

    assert torch.allclose(
        our_corr, torch.zeros_like(our_corr).fill_(scipy_corr)
    ), "Pearson test failed"


if __name__ == "__main__":
    test_pearsoncorr()