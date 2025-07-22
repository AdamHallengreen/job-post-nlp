import json
import pathlib
from pathlib import Path
from typing import Any

import scipy.sparse as ss
import spacy
from corextopic import corextopic as ct
from omegaconf import DictConfig, ListConfig, OmegaConf
from spacy.tokens import DocBin

from job_post_nlp.utils.interactive import try_inter

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


def load_corpus_split(corpus_dir: Path) -> DocBin:
    """
    Load and combine split corpus files from a directory into a single DocBin.
    Args:
        corpus_dir (Path): Directory containing .spacy files.
    Returns:
        DocBin: Combined corpus.
    """
    combined_corpus = DocBin(store_user_data=True)
    nlp = spacy.blank("da")

    # Get all .spacy files and sort them by name to maintain order
    spacy_files = sorted(corpus_dir.glob("*.spacy"))

    for spacy_file in spacy_files:
        chunk_corpus = DocBin().from_disk(spacy_file)
        for doc in chunk_corpus.get_docs(nlp.vocab):
            combined_corpus.add(doc)

    return combined_corpus


def load_tdm(tdm_file: Path, tdm_info_file: Path) -> tuple[ss.csr_matrix, dict]:
    """
    Load a Term Document Matrix (TDM) from a .npz file as a sparse matrix,
    and load vocab and ids from JSON files.
    Args:
        tdm_file (Path): Path to the TDM .npz file.
        tdm_info_file (Path): Path to the tdm_info JSON file.
    Returns:
        tuple: (csr_matrix, tdm_info dict of vocab and ids)
    """
    if (not tdm_file.exists()) or (not tdm_info_file.exists()):
        raise FileNotFoundError()
    with open(tdm_info_file, encoding="utf-8") as f:
        tdm_info = json.load(f)

    tdm = ss.load_npz(str(tdm_file))
    return tdm, tdm_info


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


def train_corex(tdm: ss.csr_matrix, tdm_info: dict, par: DictConfig) -> object:
    """
    Train a Corex topic model using a sparse matrix, vocabulary, and document ids.
    Args:
        tdm: sparse matrix (csr_matrix)
        tdm_info: dict containing 'vocab' and 'ids'
            - vocab: list of terms
            - ids: list of document IDs
        par: DictConfig with model parameters
    Returns:
        Corex model object
    """
    model = ct.Corex(
        n_hidden=par.corex.n_topics,
        max_iter=par.settings.max_iter,
        verbose=par.settings.verbose,
        seed=par.settings.seed,
    )
    anchors = convert_anchors(par.corex.anchors) if par.corex.anchors is not None else None
    model.fit(
        tdm, words=tdm_info["vocab"], docs=tdm_info["ids"], anchors=anchors, anchor_strength=par.corex.anchor_strength
    )
    return model


def export_model(model: object, output_file: str | pathlib.Path) -> None:
    model.save(output_file, ensure_compatibility=False)  # type: ignore  # noqa: PGH003


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    params_path = project_root / "params.yaml"
    corpus_dir = data_dir / "corpus_split"  # Updated to use split corpus directory
    output_file = data_dir / "most_common_words.json"
    tdm_file = data_dir / "tdm.npz"
    tdm_info_file = data_dir / "tdm_info.json"

    # Load parameters
    par = OmegaConf.load(params_path).train

    print("Loading tdm and tdm_info")
    # load tdm, vocab, and ids
    tdm, tdm_info = load_tdm(tdm_file, tdm_info_file)

    print("Training Corex model")
    # Process
    model = train_corex(tdm, tdm_info, par)

    print("Exporting Corex model")
    export_model(model, models_dir / "corex_model.pkl")
