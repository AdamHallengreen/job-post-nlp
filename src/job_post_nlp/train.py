import pathlib
from pathlib import Path
from typing import Any

import polars as pl
import scipy.sparse as ss  # type: ignore  # noqa: PGH003
from corextopic import corextopic as ct  # type: ignore  # noqa: PGH003
from omegaconf import DictConfig, ListConfig, OmegaConf
from spacy.tokens import DocBin

from job_post_nlp.interactive import try_inter

try_inter()
from job_post_nlp.utils.find_project_root import find_project_root  # noqa: E402


def load_corpus(file_path: Path) -> DocBin:
    """
    Load a preprocessed corpus from a .spacy binary file.
    Args:
        file_path (Path): Path to the .spacy file.
    Returns:
        DocBin: The loaded corpus.
    """
    doc_bin = DocBin().from_disk(file_path)
    return doc_bin


def load_tdm(file_path: Path) -> pl.DataFrame:
    """
    Load a Term Document Matrix (TDM) from a CSV file.
    Args:
        file_path (Path): Path to the TDM CSV file.
    Returns:
        pl.DataFrame: The loaded TDM.
    """
    if not file_path.exists():
        raise FileNotFoundError()
    return pl.read_parquet(file_path)


class UnsupportedAnchorTypeError(Exception):
    """Exception raised when an unsupported anchor type is encountered."""

    def __init__(self, anchor: Any):
        message = f"Unsupported anchor type: {type(anchor).__name__}.Anchor must be a string or a list of strings."
        super().__init__(message)


def convert_anchors(anchors: list | ListConfig) -> list:
    converted_anchors = []
    for anchor in anchors:
        if isinstance(anchor, (str, list)):
            converted_anchors.append(anchor)
        elif isinstance(anchor, ListConfig):
            converted_anchors.append(list(anchor))
        else:
            raise UnsupportedAnchorTypeError(anchor)
    return converted_anchors


def train_corex(tdm: pl.DataFrame, par: DictConfig) -> object:
    words = tdm.columns[1:]
    docs = tdm.select(pl.col("doc_id")).to_series().to_list()
    X = tdm.select(pl.exclude("doc_id")).to_numpy()
    X = ss.csr_matrix(X)

    # check_anchors_in_vocab(words, par)
    model = ct.Corex(
        n_hidden=par.corex.n_topics,
        max_iter=par.settings.max_iter,
        verbose=par.settings.verbose,
        seed=par.settings.seed,
    )
    anchors = convert_anchors(par.corex.anchors) if par.corex.anchors is not None else None
    model.fit(X, words=words, docs=docs, anchors=anchors, anchor_strength=par.corex.anchor_strength)
    return model


def export_model(model: object, output_file: str | pathlib.Path) -> None:
    model.save(output_file, ensure_compatibility=False)  # type: ignore  # noqa: PGH003


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    params_path = project_root / "params.yaml"
    corpus_file = data_dir / "corpus.spacy"  # Use corpus file
    output_file = data_dir / "most_common_words.json"
    tdm_file = data_dir / "tdm.parquet"

    # Load parameters
    par = OmegaConf.load(params_path).train

    # Process
    tdm = load_tdm(tdm_file)
    model = train_corex(tdm, par)
    export_model(model, models_dir / "corex_model.pkl")
