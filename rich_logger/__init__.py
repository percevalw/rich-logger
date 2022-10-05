from warnings import warn

from .table_printer import RichTablePrinter

try:
    from .pl_logger import RichTableLogger
except ImportError as e:

    class RichTableLogger:
        def __init__(self, key=None, fields=None):
            raise Exception(
                "Could not import RichTableLogger, some packages might be missing: {}".format(
                    e.name
                )
            )
