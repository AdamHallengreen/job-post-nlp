import json
from collections.abc import Generator
from pathlib import Path

import spacy
import yaml
from tqdm import tqdm

from job_post_nlp.utils.find_project_root import find_project_root


class ColumnNotFoundError(Exception):
    def __init__(self, column_name: str) -> None:
        super().__init__(f"Column '{column_name}' not found in the Excel file.")


def load_data(file_path: Path, column_name: str = "Text") -> list:
    """
    Load data from an Excel file and extract the specified column.

    Args:
        file_path (Path): Path to the Excel file.
        column_name (str): Name of the column to extract.

    Returns:
        list: A list of texts from the specified column.
    """
    import polars as pl  # Import here to avoid dependency issues if unused

    df = pl.read_excel(file_path, sheet_name="Sheet1")

    if column_name not in df.columns:
        raise ColumnNotFoundError(column_name)
    return df[column_name].to_list()


def process_texts_generator(
    texts: list[str], nlp: spacy.language.Language, batch_size: int = 1000, threads: int = 1
) -> Generator[list[str], None, None]:
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
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=threads),
        total=len(texts),
        desc="Preprocessing texts",
    ):
        yield [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]


def preprocess_texts(texts: list, params: dict) -> list:
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
