import os
import pathlib
from ._model import KeyBERT

__version__ = "0.6.0"
os.environ['TORCH_HOME'] = pathlib.Path(__file__).resolve().parent.as_posix()