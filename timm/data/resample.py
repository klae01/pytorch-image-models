from torch import Tensor


def sample_level_weight(P: Tensor, resample: float) -> Tensor:
    """Oversampling and undersampling to adjust the frequency of occurrence of each class.
    not change = 1
    """
    return P**resample / ((P ** (1 + resample)).sum())


def class_level_weight(P: Tensor, resample: float) -> Tensor:
    """ The final weight of each class when adjusting the frequency of occurrence of each class through oversampling and undersampling
    sum of return tensor = 1
    """
    return P ** (1 + resample) / ((P ** (1 + resample)).sum())
