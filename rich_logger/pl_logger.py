from typing import Dict, Optional, Union

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from .table_printer import RichTablePrinter


class RichTableLogger(LightningLoggerBase):
    def __init__(
        self, fields: Dict[str, Union[Dict, bool]] = {}, key: Optional[str] = None
    ):
        """
        Logger based on `rich` tables

        Parameters
        ----------
        key: Optional[str]
            main key to group results by row
        fields: Dict[str, Union[Dict, bool]]
            Field descriptors containing goal ("lower_is_better" or "higher_is_better"),
             format and display name
            The key is a regex that will be used to match the fields to log
            Each entry of the dictionary should match the following scheme:
            - key: a regex to match columns
            - value: either a Dict or False to hide the column, the dict format is
                - name: the name of the column
                - goal: "lower_is_better" or "higher_is_better"
                - goal_wait: how many logging row should we wait before coloring cells
                - format: a format string to display the value
        """
        super().__init__()
        self.printer = RichTablePrinter(key=key, fields=fields)

    @property
    def name(self):
        return "RichTableLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.printer.log_metrics({"step": step, **metrics})

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        self.printer.finalize()
