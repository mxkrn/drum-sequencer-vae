def linear_anneal(epoch: int, max_epoch: int):
    """Linear annealing strategy based on the current epoch."""
    if epoch > max_epoch:
        ratio = 1.0
    else:
        ratio = epoch / max_epoch
    return 1.0 - ratio
