from pathlib import Path

from dvclive import Live

from job_post_nlp.evaluate import load_most_common_words, plot_most_common_words
from job_post_nlp.utils.find_project_root import find_project_root

if __name__ == "__main__":
    # Define file paths
    project_root = find_project_root(__file__)
    input_file = Path(project_root) / "data" / "most_common_words.json"
    output_path = Path(project_root) / "output"
    metrics = Path(project_root) / "metrics.yaml"

    # Load the most common words
    common_words = load_most_common_words(input_file)

    # Log metrics using DVCLive
    with Live(dir=str(output_path), cache_images=True, report="md", save_dvc_exp=False) as live:
        for i, (word, _) in enumerate(common_words):
            live.log_metric(f"Top {i + 1}", word, plot=False)

        fig = plot_most_common_words(common_words)
        live.log_image("most_common_words.png", fig)

        live.make_report()
