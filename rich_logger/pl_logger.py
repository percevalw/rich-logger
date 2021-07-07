from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from .table_printer import RichTablePrinter


class RichTableLogger(LightningLoggerBase):
    def __init__(self, key=None, fields=None):
        """
        Pytorch lightning logger based on RichTablePrinter
        :param key: str or None
            main key to group results by row
        :param fields: dict of (dict or False)
            Field descriptors containing goal ("lower_is_better" or "higher_is_better"), format and display name
            The key is a regex that will be used to match the fields to log
        """
        super().__init__()
        self.printer = RichTablePrinter(key=key, fields=fields)

    @property
    def name(self):
        return 'RichTableLogger'

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return '0.1'

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.printer.log({"step": step, **metrics})

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
