import pathlib
from pathlib import Path

import polars as pl
import scipy.sparse as ss  # type: ignore  # noqa: PGH003
import yaml
from corextopic import corextopic as ct  # type: ignore  # noqa: PGH003
from spacy.tokens import DocBin

from job_post_nlp.utils.find_project_root import find_project_root


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


def train_corex(tdm: pl.DataFrame, params: dict) -> object:
    words = tdm.columns[1:]
    docs = tdm.select(pl.col("doc_id")).to_series().to_list()
    X = tdm.select(pl.exclude("doc_id")).to_numpy()
    X = ss.csr_matrix(X)

    model = ct.Corex(
        n_hidden=params["n_topics"], max_iter=params["max_iter"], verbose=params["verbose"], seed=params["seed"]
    )
    model.fit(X, words=words, docs=docs, anchors=params["anchors"], anchor_strength=params["anchor_strength"])
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
    with open(params_path) as file:
        params = yaml.safe_load(file)["train"]

    # Process
    tdm = load_tdm(tdm_file)
    model = train_corex(tdm, params)
    export_model(model, models_dir / "corex_model.pkl")
