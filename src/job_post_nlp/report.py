from pathlib import Path

from dvclive import Live

from job_post_nlp.evaluate import (
    get_top_words,
    load_model,
    most_common_languages,
    plot_num_job_posts_per_topic,
    plot_TC,
)
from job_post_nlp.prepare import load_excel
from job_post_nlp.train import load_corpus
from job_post_nlp.utils.find_project_root import find_project_root

if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = Path(project_root) / "output"
    report_path = output_dir / "report.md"
    metrics = Path(project_root) / "metrics.yaml"

    # Load the most common words
    model = load_model(models_dir / "corex_model.pkl")
    texts = load_excel(data_dir / "Jobnet.xlsx", sheet_name="Sheet1")
    corpus = load_corpus(data_dir / "corpus.spacy")  # Use corpus file

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, report="md", save_dvc_exp=False) as live:
        live.log_metric("Overall TC", model.tc, plot=False)  # type: ignore  # noqa: PGH003

        fig = plot_TC(model)
        live.log_image("total_correlation.png", fig)

        fig = plot_num_job_posts_per_topic(model)
        live.log_image("num_job_posts_per_topic.png", fig)

        live.make_report()

    # Add text onto the report
    with open(report_path, "a", encoding="utf-8") as report_file:
        report_file.write("# Topic words\n")
        top_words = get_top_words(model, 10)
        text = "\n".join([f"Topic {i}: {', '.join(words)}\n" for i, words in top_words.items()])
        report_file.write(text)
        report_file.write("\n")

        report_file.write("# Most common langauges in text\n")
        text = most_common_languages(corpus)
        report_file.write(text)
        report_file.write("\n")
