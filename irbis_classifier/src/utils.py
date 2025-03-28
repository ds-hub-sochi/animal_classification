from __future__ import annotations

from pathlib import Path
from random import sample
from typing import Sequence
from urllib.parse import urlparse, ParseResult

import pandas as pd
import torch
from loguru import logger


def filter_non_images(image_paths: Sequence[Path]) -> list[Path]:
    """
    This function filters non-images according to file's extension

    Args:
        image_paths (Sequence[Path]): a sequence of strings representing filenames

    Returns:
        list[Path]: a sequence of only images filenames
    """

    return [
        image_path for image_path in image_paths
        if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}
    ]


def fix_rus_i_naming(filename: str) -> str:
    """
    This function fixes russian i symbol problem
    Fix inconsistency with 'й' symbol.
    First one is quite normal in Ubuntu/Mac and presents in .json markup
    The second one is different and presents in original filenames

    Args:
        filename (str): a string (typically a filename) where you want to fix "russian i" symbol

    Returns:
        str: a fixed string
    """

    return filename.replace('й', 'й')


def sample_from_dataframe(
    df: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """
    This function samples from given pandas dataframe

    Args:
        df (pd.DataFrame): DataFrame from which you want to sample data 
        sample_size (int): size of sample you want to produce

    Raises:
        ValueError: raised if sample_size > df.shape[0]

    Returns:
        pd.DataFrame: dataframe with sampled rows
    """

    if sample_size > df.shape[0]:
        raise ValueError("can't sample from the dataset with smaller number of rows")

    indices: list[int] = sample(
        list(df.index),
        sample_size,
    )

    return df.loc[indices]


def save_model_as_traced(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    save_path: str | Path,
) -> None:
    """
    This function saves given torch.nn.Module as a torch.jit.ScriptModule

    Args:
        model (torch.nn.Module): model you want to save as a TracedModel
        sample_input (torch.Tensor): required sample input (only shape matters)
        save_path (str | Path): path where you want to save new model
    """

    traced_model: torch.jit.ScriptModule = torch.jit.trace(
        model,
        sample_input,
    )
    traced_model.save(save_path)

    logger.success('model saved as traced model')


def is_valid_url(url: str) -> bool:
    """
    This function checks if given string a valid URL or not

    Args:
        url (str): string you want to check

    Returns:
        bool: True if given string is a valid URL else False
    """

    try: 
        result: ParseResult = urlparse(url)
        return all([result.scheme, result.netloc]) 
    except Exception: 
        return False
