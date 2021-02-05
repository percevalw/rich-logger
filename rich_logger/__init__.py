from warnings import warn

from .table_printer import RichTablePrinter

try:
    from .pl_logger import RichTableLogger
except ImportError as e:
    warn("Cannot import RichTableLogger, some packages might be missing: {}".format(e.name))
