import pandas as pd
from .nbaiot import load_nbaiot
def load_iot23(use_synthetic=False):
    return load_nbaiot(use_synthetic)
