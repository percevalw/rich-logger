import sys
from typing import IO, Any, Callable, Dict, Optional, Tuple

import spacy
from spacy import Errors, registry
from tqdm import tqdm

from .table_printer import RichTablePrinter


@registry.loggers("rich-logger")
def rich_logger(
    progress_bar: bool = False,
) -> Callable[
    [spacy.Language],
    Tuple[Callable[[Optional[Dict[str, Any]]], None], Callable[[], None]],
]:
    """
    A rich based logger that renders nicely in Jupyter notebooks and console

    Parameters
    ----------
    progress_bar: bool
        Whether to show a training progress bar or not

    Returns
    -------
    Tuple[Callable[[Optional[Dict[str, Any]]], None], Callable[[], None]]]
    """

    def setup_printer(
        nlp: spacy.Language, stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Optional[Dict[str, Any]]], None], Callable[[], None]]:

        # ensure that only trainable components are logged
        logged_pipes = [
            name
            for name, proc in nlp.pipeline
            if hasattr(proc, "is_trainable") and proc.is_trainable
        ]
        eval_frequency = nlp.config["training"]["eval_frequency"]
        score_weights = nlp.config["training"]["score_weights"]
        score_cols = [
            col for col, value in score_weights.items() if value is not None
        ] + ["speed"]

        fields = {"epoch": {}, "step": {}}
        for pipe in logged_pipes:
            fields[f"loss_{pipe}"] = {
                "format": "{0:.2f}",
                "name": f"Loss {pipe}".upper(),
                "goal": "lower_is_better",
            }
        for score, weight in score_weights.items():
            if score != "speed" and weight is not None:
                fields[score] = {
                    "format": "{0:.2f}",
                    "name": score.upper(),
                    "goal": "higher_is_better",
                }
        fields["speed"] = {"name": "WPS"}
        fields["duration"] = {"name": "DURATION"}
        table_printer = RichTablePrinter(fields=fields)
        table_printer.hijack_tqdm()

        progress: Optional[tqdm] = None
        last_seconds = 0

        def log_step(info: Optional[Dict[str, Any]]) -> None:
            nonlocal progress, last_seconds

            if info is None:
                # If we don't have a new checkpoint, just return.
                if progress is not None:
                    progress.update(1)
                return

            data = {
                "epoch": info["epoch"],
                "step": info["step"],
            }

            for pipe in logged_pipes:
                data[f"loss_{pipe}"] = float(info["losses"][pipe])

            for col in score_cols:
                score = info["other_scores"].get(col, 0.0)
                try:
                    score = float(score)
                except TypeError:
                    err = Errors.E916.format(name=col, score_type=type(score))
                    raise ValueError(err) from None
                if col != "speed":
                    score *= 100
                data[col] = score
            data["duration"] = info["seconds"] - last_seconds
            last_seconds = info["seconds"]

            table_printer.log(data)

            if progress_bar:
                # Set disable=None, so that it disables on non-TTY
                progress = tqdm(
                    total=eval_frequency, disable=None, leave=False, file=stderr
                )
                progress.set_description(f"Epoch {info['epoch'] + 1}")

        def finalize() -> None:
            table_printer.finalize()

        return log_step, finalize

    return setup_printer
