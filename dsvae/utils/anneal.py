def linear_anneal(epoch: int, max_epoch: int):
    """Linear annealing strategy based on the current epoch."""
    if epoch > max_epoch:
        note_dropout = 1.0
    else:
        note_dropout = epoch / max_epoch
    return 1.0 - note_dropout
