import json
from collections.abc import Generator
from pathlib import Path

import polars as pl
import spacy
import yaml
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


def process_texts_generator(
    texts: list[tuple[str, str]], nlp: spacy.language.Language, batch_size: int = 1000, threads: int = 1
) -> Generator[tuple[str, list[str]], None, None]:
    """
    Processes texts using spaCy and yields a list of lemmatized and filtered tokens for each document.

    Args:
        texts (list): A list of input text strings.
        nlp (spacy.language.Language): The loaded spaCy language model.
        batch_size (int): The batch size for spaCy's nlp.pipe().
        threads (int): The number of parallel processes for spaCy's nlp.pipe().

    Yields:
        list: A list of lowercase lemmas of alphabetic and non-stop tokens for each document.
    """
    for doc, context in tqdm(
        nlp.pipe(texts, as_tuples=True, batch_size=batch_size, n_process=threads),
        total=len(texts),
        desc="Preprocessing texts",
    ):
        yield (context, [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop])


def preprocess_texts(texts: list[tuple[str, str]], params: dict) -> list:
    """
    Preprocess a list of texts using spaCy, including tokenization, lemmatization,
    stopword removal, and lowercasing.

    Args:
        texts (list): A list of texts to preprocess.

    Returns:
        list: A list of preprocessed texts as lists of tokens.
    """
    # Load the spaCy language model
    nlp = spacy.load("da_core_news_sm", enable=params["pipeline"])

    # Process the texts in batches and collect the results
    processed_texts = list(
        process_texts_generator(
            texts,
            nlp,
            batch_size=params["batch_size"],
            threads=params["threads"],
        )
    )

    return processed_texts


def export_corpus(corpus: list, output_file: Path) -> None:
    """
    Export the preprocessed corpus to a file in JSON format.

    Args:
        corpus (list): A list of preprocessed texts.
        output_file (Path): Path to the output file.
    """
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)


if __name__ == "__main__":
    # Define file paths
    project_root = find_project_root(__file__)
    params_path = Path(project_root) / "params.yaml"
    file_path = project_root / "data" / "Jobnet.xlsx"
    output_file = project_root / "data" / "corpus.json"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["prepare"]
    nobs = params["nobs"]

    # Process the data
    texts = load_data(file_path)
    preprocessed_corpus = preprocess_texts(texts[:nobs], params)
    export_corpus(preprocessed_corpus, output_file)

    print(f"Preprocessed corpus exported to {output_file}")
