from collections.abc import Generator
from pathlib import Path
from typing import Optional

import polars as pl
import spacy
from lingua import LanguageDetectorBuilder
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


class UnsupportedLinguaOutput(Exception):
    def __init__(self, output: str) -> None:
        self.message = f"Unsupported Lingua output: {output}"
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

    # for 3 obs there is a duplciate id called: Virksomheden har valgt at rekruttere via jobcentret
    count = 1
    for i in range(len(id_col)):
        if id_col[i] == "Virksomheden har valgt at rekruttere via jobcentret":
            id_col[i] = f"jobcentret_{count}"
            count += 1

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


def detect_language(texts: list[tuple[str, str]]) -> dict:
    """
    Detect the language of the texts using Lingua.
    Args:
        texts : A list of tuples containing text IDs and their corresponding texts.
    Returns:
        list: A dictionary of detected languages for each text linked to id.
    """
    detector = LanguageDetectorBuilder.from_all_languages().build()
    languages = {}

    # We could possibly speed this up by setting the languages to detect
    outputs = detector.detect_languages_in_parallel_of([text[0] for text in texts])
    for i, output in enumerate(outputs):
        if output is not None:
            if hasattr(output, "iso_code_639_3"):
                languages[texts[i][1]] = output.iso_code_639_3.name
            else:
                raise UnsupportedLinguaOutput(str(output))
        else:
            languages[texts[i][1]] = "unknown"

    if False:
        # For interactive checking
        for i, language in enumerate(languages):
            if language == "unknown":
                print(f"Text {i} is detected as unknown language.")
            elif (language != "DAN") and (language != "ENG"):
                print(f"Text {i} is detected as {language}.")
                print(texts[i][0])

    return languages


def register_extensions(extensions: tuple = ("text_id", "clean_tokens", "language")) -> None:
    for extension in extensions:
        if not Doc.has_extension(extension):
            Doc.set_extension(extension, default=None)


def preprocess_texts(texts: list[tuple[str, str]], languages: dict, par: DictConfig) -> DocBin:
    """
    Preprocess a list of texts using spaCy, including tokenization, lemmatization,
    stopword removal, and lowercasing.
    """
    doc_bin = DocBin(store_user_data=True)

    # Load the spaCy language model
    nlp = spacy.load(par.preprocessing.model, enable=par.preprocessing.pipeline)

    # Set stop words
    if par.preprocessing.keep_negations is not None:
        negation_words = {"ikke", "nej", "ingen", "intet", "aldrig"}
        nlp.Defaults.stop_words -= negation_words

    # Register the extension for text_id if not already set
    register_extensions()

    for doc, text_id in tqdm(
        nlp.pipe(texts, as_tuples=True, batch_size=par.settings.batch_size, n_process=par.settings.threads),
        total=len(texts),
        desc="Preprocessing texts",
    ):
        doc._.text_id = text_id
        doc._.clean_tokens = get_clean_tokens(doc)
        doc._.language = languages[text_id]

        doc_bin.add(doc)

    return doc_bin


def corpus_unpack(corpus: DocBin) -> Generator[Doc, None, None]:
    """
    Unpack the corpus and return a list of tuples containing text IDs and their corresponding lemmas.
    Args:
        corpus (DocBin): The preprocessed corpus.
    Returns:
        list: A list of tuples containing text IDs and their corresponding lemmas.
    """
    nlp = spacy.blank("da")
    register_extensions()

    yield from corpus.get_docs(nlp.vocab)


def get_clean_tokens(doc: Doc) -> list[str]:
    """
    Clean the tokens by removing non-alphabetic characters and stop words.
    Args:
        doc (Doc): A spaCy Doc object.
    Returns:
        list: A list of cleaned tokens.
    """
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]


def _build_tdm(
    corpus: DocBin, tdm_cell: str = "binary", ngram: int = 1, min_df: Optional[int | float] = None
) -> pl.DataFrame:
    """
    Build a Term-Document Matrix (TDM) using sklearn's CountVectorizer and return a dense polars DataFrame.
    Rows: document IDs
    Columns: unique terms (lemmas)
    Values: term frequency in each document
    """

    # Prepare documents and IDs
    texts = []
    text_ids = []
    for doc in tqdm(
        corpus_unpack(corpus),
        total=corpus.__len__(),
        desc="Building Term-Document Matrix",
    ):
        text_ids.append(doc._.text_id)
        tokens = doc._.clean_tokens
        texts.append(";".join(tokens))

    # Build the Term-Document
    if tdm_cell == "binary":
        vectorizer = CountVectorizer(token_pattern="[^;]+", binary=True, ngram_range=(ngram, ngram), min_df=min_df)  # noqa: S106
    elif tdm_cell == "tf":
        vectorizer = CountVectorizer(token_pattern="[^;]+", ngram_range=(ngram, ngram), min_df=min_df)  # noqa: S106
    elif tdm_cell == "tfidf":
        vectorizer = TfidfVectorizer(token_pattern="[^;]+", ngram_range=(ngram, ngram), min_df=min_df)  # noqa: S106
    else:
        raise ValueError()  # f"Unsupported TDM cell type: {par.tdm.tdm_cell}"

    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out().tolist()

    # Convert to dense and build polars DataFrame
    X_dense = X.toarray()
    data = {"doc_id": text_ids}
    for idx, term in enumerate(vocab):
        data[term] = X_dense[:, idx]
    return pl.DataFrame(data)


def build_tdm(corpus: DocBin, par: DictConfig) -> pl.DataFrame:
    """
    Build a Term-Document Matrix (TDM) using sklearn's CountVectorizer and return a dense polars DataFrame.
    Rows: document IDs
    Columns: unique terms (lemmas)
    Values: term frequency in each document
    """

    dfs = []
    for i, n in enumerate(range(1, par.tdm.ngram_n + 1)):
        # Build the Term-Document Matrix
        tdm = _build_tdm(corpus, tdm_cell=par.tdm.tdm_cell, ngram=n, min_df=par.tdm.min_df[i])
        # concat the dataframes
        dfs.append(tdm)
    # append the dataframes together (but only use first column from the first dataframe)
    tdm = dfs[0]
    for i in range(1, len(dfs)):
        tdm = tdm.hstack(dfs[i].select(pl.exclude("doc_id")))
    return tdm


def export_tdm(tdm: pl.DataFrame, output_file: Path) -> None:
    """
    Export the Term-Document Matrix (TDM) to a CSV file.
    Args:
        tdm (pl.DataFrame): The Term-Document Matrix.
        output_file (Path): Path to the output CSV file.
    """
    tdm.write_parquet(output_file)


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
    corpus_file = data_dir / "corpus.spacy"
    tdm_file = data_dir / "tdm.parquet"

    # Load parameters
    par = OmegaConf.load(params_path).prepare

    # Process the data
    texts = load_data(file_path)[: par.settings.nobs]
    languages = detect_language(texts)

    preprocessed_corpus = preprocess_texts(texts, languages, par)
    tdm = build_tdm(preprocessed_corpus, par)
    export_corpus(preprocessed_corpus, corpus_file)
    print(f"Preprocessed corpus exported to {corpus_file}")
    export_tdm(tdm, tdm_file)
    print(f"Term-Document Matrix exported to {tdm_file}")
