from pathlib import Path

import polars as pl
import spacy
import yaml
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from job_post_nlp.utils.find_project_root import find_project_root


class FileNotFoundErrorMessage:
    def __init__(self, file_path: Path) -> None:
        self.message = f"File {file_path} does not exist."

    def __str__(self) -> str:
        return self.message


class UnsupportedFileTypeError(Exception):
    def __init__(self, file_suffix: str) -> None:
        self.message = f"Unsupported file type: {file_suffix}"
        super().__init__(self.message)


def load_data(file_path: Path) -> list[tuple[str, str]]:
    """
    Load data from an Excel file and extract the specified column.

    Args:
        file_path (Path): Path to the Excel file.
        column_name (str): Name of the column to extract.
        raise FileNotFoundError(FileNotFoundErrorMessage(file_path))
    Returns:
        list: A list of texts from the specified column.
    """

    # check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(FileNotFoundErrorMessage(file_path))
    # check if the file is an Excel file
    if file_path.suffix in [".xlsx", ".xls"]:
        df = load_excel(file_path)
    else:
        raise UnsupportedFileTypeError(file_path.suffix)

    text_col = df.select(pl.col("text")).to_series().to_list()
    id_col = df.select(pl.col("id")).to_series().to_list()
    texts = [(t, i) for t, i in zip(text_col, id_col)]

    return texts


def load_excel(file_path: Path, sheet_name: str = "Sheet1") -> pl.DataFrame:
    """
    Load data from an Excel file and return it as a list of dictionaries.

    Args:
        file_path (Path): Path to the Excel file.

    Returns:
        list: A list of dictionaries representing the rows in the Excel file.
    """

    df = pl.read_excel(
        file_path, sheet_name=sheet_name, columns=["ID", "Text"], schema_overrides={"ID": pl.String, "Text": pl.String}
    ).rename({"ID": "id", "Text": "text"})
    return df


def preprocess_texts(texts: list[tuple[str, str]], params: dict) -> DocBin:
    """
    Preprocess a list of texts using spaCy, including tokenization, lemmatization,
    stopword removal, and lowercasing.

    Args:
        texts (list): A list of texts to preprocess.

    Returns:
        list: A list of preprocessed texts as lists of tokens.
    """
    if not Doc.has_extension("text_id"):
        Doc.set_extension("text_id", default=None)

    doc_bin = DocBin(store_user_data=True)

    # Load the spaCy language model
    nlp = spacy.load(params["model"], enable=params["pipeline"])

    for doc, text_id in tqdm(
        nlp.pipe(texts, as_tuples=True, batch_size=params["batch_size"], n_process=params["threads"]),
        total=len(texts),
        desc="Preprocessing texts",
    ):
        doc._.text_id = text_id
        doc_bin.add(doc)

    return doc_bin


def export_corpus(corpus: DocBin, output_file: Path) -> None:
    """
    Export the preprocessed corpus to a .spacy binary file.
    Args:
        corpus (DocBin): The preprocessed corpus.
        output_file (Path): Path to the output file.
    """
    corpus.to_disk(output_file)


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    params_path = project_root / "params.yaml"
    data_dir = project_root / "data"
    file_path = data_dir / "Jobnet.xlsx"
    output_file = data_dir / "corpus.spacy"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["prepare"]

    # Process the data
    texts = load_data(file_path)[: params["nobs"]]
    preprocessed_corpus = preprocess_texts(texts, params)
    export_corpus(preprocessed_corpus, output_file)

    print(f"Preprocessed corpus exported to {output_file}")
