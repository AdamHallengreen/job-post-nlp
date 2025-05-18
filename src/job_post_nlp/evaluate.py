import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import yaml
from corextopic import corextopic as ct  # type: ignore  # noqa: PGH003
from dvclive import Live
from matplotlib.figure import Figure

from job_post_nlp.utils.find_project_root import find_project_root


class InvalidInputFileError(Exception):
    def __init__(self) -> None:
        super().__init__("Input file must contain a list of lists.")


def load_model(model_path: str | Path) -> object:
    """
    Load a model from a file.

    Args:
        model_path (str | Path): Path to the model file.

    Returns:
        object: The loaded model.
    """
    return ct.load(model_path)


def plot_TC(model: object) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(range(model.tcs.shape[0]), model.tcs, color="#4e79a7", width=0.5)  # type: ignore  # noqa: PGH003
    ax.set_xlabel("Topic", fontsize=16)
    ax.set_ylabel("Total Correlation (nats)", fontsize=16)
    ax.set_title("Total Correlation of Topics", fontsize=20)
    return fig


def plot_num_job_posts_per_topic(model: object) -> Figure:
    n_docs_per_topic = model.labels.sum(axis=0)  # type: ignore  # noqa: PGH003
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(range(model.n_hidden), n_docs_per_topic, color="#4e79a7", width=0.5)  # type: ignore  # noqa: PGH003
    ax.set_xlabel("Topic", fontsize=16)
    ax.set_ylabel("Number of Job Posts", fontsize=16)
    ax.set_title("Number of Job Posts per Topic", fontsize=20)
    return fig


def get_top_words(model: object, n_words: int = 10) -> dict:
    """
    Get the top words for each topic.

    Args:
        model (object): The trained model.
        n_words (int): Number of top words to retrieve.

    Returns:
        dict: A dictionary with topic numbers as keys and lists of top words as values.
    """
    topics = model.get_topics(n_words)  # type: ignore  # noqa: PGH003
    top_words = {}
    for n, topic in enumerate(topics):
        topic_words, _, _ = zip(*topic)
        top_words[n] = list(topic_words)
    return top_words


def save_top_words(model: object, output_file: Path) -> None:
    """
    Save the top words for each topic to a YAML file, with each key on a new line.

    Args:
        model (object): The trained model.
        output_file (Path): Path to the output file.
    """
    top_words = get_top_words(model)
    if output_file.suffix == ".yaml":
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(top_words, f, allow_unicode=True, default_flow_style=False)
    elif output_file.suffix == ".txt":
        with open(output_file, "w", encoding="utf-8") as f:
            for topic_n, words_list in top_words.items():
                f.write(f"Topic {topic_n}: {', '.join(words_list)}\n")
    elif output_file.suffix == ".json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(top_words, f, ensure_ascii=False, indent=4)
    else:
        raise ValueError()


def get_best_match(model: object, texts: pl.DataFrame, j: int, x: int) -> str:
    top_docs = model.get_top_docs()  # type: ignore  # noqa: PGH003
    topic_docs, _ = zip(*top_docs[j])
    doc_id = topic_docs[x]

    # return value where text_id is doc_id
    text = texts.filter(pl.col("id") == doc_id).select(pl.col("text")).to_series().to_list()[0]
    return str(text)


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    metrics = project_root / "metrics.yaml"

    # Process
    model = load_model(models_dir / "corex_model.pkl")
    save_top_words(model, output_dir / "top_words.yaml")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True) as live:
        live.log_metric("Overall TC", model.tc, plot=False)  # type: ignore  # noqa: PGH003

        # TC plot
        TC_df = pd.DataFrame(model.tcs, columns=["TC"])  # type: ignore  # noqa: PGH003
        live.log_plot(
            "Total Correlation",
            TC_df,
            x="TC",
            y="Topic",
            template="bar_horizontal_sorted",
            title="Total Correlation of Topics",
        )
        fig = plot_TC(model)
        live.log_image("total_correlation.png", fig)

        # Number of job posts per topic
        n_docs_per_topic = model.labels.sum(axis=0)  # type: ignore  # noqa: PGH003
        n_docs_per_topic_df = pd.DataFrame(n_docs_per_topic, columns=["Number of Job Posts"])
        live.log_plot(
            "Number of Job Posts per Topic",
            n_docs_per_topic_df,
            x="Number of Job Posts",
            y="Topic",
            template="bar_horizontal_sorted",
            title="Number of Job Posts per Topic",
        )
        fig = plot_num_job_posts_per_topic(model)
        live.log_image("num_job_posts_per_topic.png", fig)
