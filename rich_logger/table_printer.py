import os
import re
import typing
from collections import OrderedDict
from enum import Enum
from numbers import Number
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from pydantic import BaseModel, Extra
from rich import get_console, reconfigure
from rich.console import RenderableType
from rich.control import Control
from rich.jupyter import _render_segments
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table
from rich.text import Text

T = typing.TypeVar("T")

# If we're in a slurm job, force the console to act as in a standard terminal
if "SLURM_JOBID" in os.environ:
    reconfigure(force_terminal=True)


def check_is_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class RichToHTML:
    def __init__(self, renderable, console):
        self.renderable = renderable
        self.console = console

    def _repr_mimebundle_(
        self,
        include: Sequence[str],
        exclude: Sequence[str],
        **kwargs: Any,
    ) -> Dict[str, str]:
        console = get_console()

        # In the browser, we can scroll horizontally so we don't need a limit
        console_options = console.options
        console_options.max_width = 2**32 - 1

        segments = list(console.render(self.renderable, console_options))
        html = _render_segments(segments)
        text = console._render_buffer(segments)
        data = {"text/plain": text, "text/html": html}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class FixedLive(Live):
    def refresh(self) -> None:
        """Update the display of the Live Render."""
        with self._lock:
            self._live_render.set_renderable(self.renderable)
            if self.console.is_jupyter:  # pragma: no cover
                try:
                    from IPython.display import display
                except ImportError:
                    import warnings

                    warnings.warn('install "ipywidgets" for Jupyter support')
                else:

                    if self.ipy_widget is None:
                        self.ipy_widget = display(None, display_id=True)

                    self.ipy_widget.update(
                        RichToHTML(
                            renderable=self._live_render.renderable,
                            console=self.console,
                        )
                    )

            elif self.console.is_terminal and not self.console.is_dumb_terminal:
                with self.console:
                    self.console.print(Control())
            elif (
                not self._started and not self.transient
            ):  # if it is finished allow files or dumb-terminals to see final result
                with self.console:
                    self.console.print(Control())


class Goal(str, Enum):
    lower_is_better = "lower_is_better"
    higher_is_better = "higher_is_better"


class Rule(BaseModel):
    class Config:
        extra = Extra.forbid

    name: Optional[str] = None
    goal: Optional[Goal] = None
    goal_wait: Optional[int] = None
    format: Optional[str] = None


class RuleConfig(BaseModel):
    rules: Dict[str, Union[Rule, bool]]


def validate_rules(rules: Dict[str, Union[Rule, bool]]):
    return RuleConfig(rules=rules).rules


def get_last_matching_index(matchers: Sequence[str], name: str):
    for i, matcher in reversed(list(enumerate(matchers))):
        if re.match(matcher, name):
            return i
    raise ValueError()


def get_last_matching_value(matchers, name, field, default):
    for matcher, rule in reversed(list(matchers.items())):
        if re.match(matcher, name):
            if rule is False:
                return matcher, False
            elif rule is True:
                return matcher, default
            else:
                field_value = getattr(rule, field)
                if field_value is not None:
                    return matcher, field_value
    return None, default


class PostfixColumn(ProgressColumn):
    """A column containing text."""

    def __init__(
        self,
        table_column: Optional[Column] = None,
    ) -> None:
        super().__init__(table_column=table_column or Column(no_wrap=True))

    def render(self, task: "Task") -> Text:
        _text = task.fields.get("postfix", "")
        text = Text(_text)
        return text


class RichTablePrinter(Progress):
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
        """

        # Logging attributes
        self.key: str = key
        self.key_to_row_idx: Dict[str, int] = {}
        self.name_to_column_idx: Dict[str, int] = {}
        self.best = {}
        self._old_tqdm_new: Any = None
        self.displayed_tasks: typing.List[Dict] = []
        if key is not None and key not in fields:
            fields = {key: {}, **fields}
        self.rules = validate_rules(fields)

        # Display attributes
        self.logger_table: Optional[Table] = None
        self.display_handle: Any = None

        super().__init__(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            PostfixColumn(),
        )
        self.live.__class__ = FixedLive

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        if self.logger_table is not None:
            yield self.logger_table
        tasks_table = self.make_tasks_table(self.tasks)
        yield tasks_table

    def prune_tasks(self):
        # Prune displayed & finished tasks
        remaining_tasks = []
        for last_task in self.displayed_tasks:
            if last_task["disposable"]:
                self.remove_task(last_task["task_id"])
            else:
                remaining_tasks.append(last_task)
        self.displayed_tasks = remaining_tasks

    def progress_bar(
        self,
        iterable: Iterable[T],
        description: str = "",
        leave: bool = False,
        total: int = None,
    ) -> Iterable[T]:
        """
        A progress bar that plays nicely with the rendered table

        Parameters
        ----------
        iterable: Iterable
            Iterable for the progress bar
        description: str
            Description that will be displayed
        leave: bool
            Should we leave the progress bar when the iterable is over
        total: Optional[int]
            Total length of the iterable

        Returns
        -------
        Iterable
        """

    def log_metrics(self, info: Dict[str, Any]):
        """
        Adds or update a row in the table

        Parameters
        ----------
        info: Dict[str, Any]
            The value of each column
        """
        self.ensure_live()

        for name, value in info.items():
            if name not in self.name_to_column_idx:
                matcher, column_name = get_last_matching_value(
                    self.rules, name, "name", default=name
                )
                if column_name is False:
                    self.name_to_column_idx[name] = -1
                    continue
                if (
                    matcher is not None
                    and re.compile(matcher).groups > 0
                    and re.findall(r"\\\d+", column_name)
                ):
                    column_name = re.sub(matcher, column_name, name)
                self.logger_table.add_column(column_name, no_wrap=True)
                self.logger_table.columns[-1]._cells = [""] * (
                    len(self.logger_table.columns[0]._cells)
                    if len(self.logger_table.columns)
                    else 0
                )
                self.name_to_column_idx[name] = (
                    (max(self.name_to_column_idx.values()) + 1)
                    if len(self.name_to_column_idx)
                    else 0
                )
        new_name_to_column_idx = {}
        columns = []

        def get_name_index(name):
            try:
                return get_last_matching_index(self.rules, name)
            except ValueError:
                return len(self.name_to_column_idx)

        for name in sorted(self.name_to_column_idx.keys(), key=get_name_index):
            if self.name_to_column_idx[name] >= 0:
                columns.append(self.logger_table.columns[self.name_to_column_idx[name]])
                new_name_to_column_idx[name] = (
                    (max(new_name_to_column_idx.values()) + 1)
                    if len(new_name_to_column_idx)
                    else 0
                )
            else:
                new_name_to_column_idx[name] = -1
        self.logger_table.columns = columns
        self.name_to_column_idx = new_name_to_column_idx

        if (
            self.key is not None
            and self.key in info
            and info[self.key] in self.key_to_row_idx
        ):
            idx = self.key_to_row_idx[info[self.key]]
        elif self.key is not None and self.key not in info and self.key_to_row_idx:
            idx = list(self.key_to_row_idx.values())[-1]
        else:
            self.logger_table.add_row()
            idx = len(self.logger_table.rows) - 1
            if self.key is not None:
                self.key_to_row_idx[info[self.key]] = idx
        for name, value in info.items():
            if self.name_to_column_idx[name] < 0:
                continue
            prev_count = len(
                self.logger_table.columns[self.name_to_column_idx[name]]._cells
            )
            formatted_value = get_last_matching_value(self.rules, name, "format", "{}")[
                1
            ].format(value)
            goal = get_last_matching_value(self.rules, name, "goal", None)[1]
            goal_wait = get_last_matching_value(self.rules, name, "goal_wait", 1)[1]
            if goal is not None:
                if name not in self.best or prev_count <= goal_wait:
                    self.best[name] = value
                else:
                    diff = (value - self.best[name]) * (
                        -1 if goal == "lower_is_better" else 1
                    )
                    if diff > 0:
                        self.best[name] = value
                        formatted_value = "[green]" + formatted_value + "[/green]"
                    elif diff <= 0:
                        formatted_value = "[red]" + formatted_value + "[/red]"
            self.logger_table.columns[self.name_to_column_idx[name]]._cells[
                idx
            ] = formatted_value

        self.refresh()

    @property
    def log(self):
        return self.log_metrics

    @log.setter
    def log(self, logger):
        self.rich_log = logger

    def ensure_live(self):
        if self.logger_table is None:
            self.logger_table = Table()
            self.console.clear_live()
            self.start()

    def finalize(self):
        if self.logger_table is not None:
            self.refresh()
            self.stop()
            try:
                import tqdm
            except ImportError:
                pass
            else:
                if self._old_tqdm_new is not None:
                    tqdm.tqdm.__new__ = self._old_tqdm_new
                    self._old_tqdm_new = None
            if self.console.is_interactive and self.console.is_terminal:
                self.console.print()
            self.logger_table = None

    def hijack_tqdm(self):
        """
        Monkey patch tqdm to use the progress bar method of this class instead
        """
        import tqdm

        if self._old_tqdm_new is not None:
            return

        self._old_tqdm_new = tqdm.tqdm.__new__

        logger = self

        def __new__(
            self,
            iterable=None,
            desc=None,
            total=None,
            leave=False,
            file=None,
            ncols=None,
            mininterval=0.1,
            maxinterval=10.0,
            miniters=None,
            ascii=None,
            disable=False,
            unit="it",
            unit_scale=False,
            dynamic_ncols=False,
            smoothing=0.3,
            bar_format=None,
            initial=0,
            position=None,
            postfix=None,
            unit_divisor=1000,
            write_bytes=None,
            lock_args=None,
            nrows=None,
            colour=None,
            delay=0,
            gui=False,
            **kwargs,
        ):
            logger.ensure_live()

            return TqdmShim(
                iterable=iterable,
                description=desc,
                total=total,
                leave=leave,
                disable=disable,
                printer=logger,
            )

        tqdm.tqdm.__new__ = __new__

    def __enter__(self) -> "RichTablePrinter":
        self.hijack_tqdm()
        self.ensure_live()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


class TqdmShim:
    def __init__(
        self,
        total,
        iterable,
        disable,
        description,
        leave,
        printer: "RichTablePrinter",
    ):
        self.total = total
        self.iterable = iterable
        self.disable = disable
        self.description = description
        self.leave = leave
        self.printer = printer

        if self.disable:
            self.task = self.task_id = None
        else:
            if self.total is None:
                try:
                    self.total = len(self.iterable)
                except (AttributeError, TypeError):
                    pass

                self.printer.prune_tasks()

            self.task_id = self.printer.add_task(
                description=self.description or "",
                total=self.total,
            )
            self.task = {"task_id": self.task_id, "disposable": False}
            self.printer.displayed_tasks.append(self.task)

    def __iter__(self):
        if self.disable:
            yield from self.iterable
        else:
            try:
                for item in self.iterable:
                    yield item
                    self.printer.update(self.task_id, advance=1)
            finally:
                if not self.leave:
                    self.task["disposable"] = True

    def reset(self, total=0):
        if self.task_id is None:
            return
        self.printer.reset(self.task_id, total=total)

    def update(self, n=1):
        if self.task_id is None:
            return
        self.printer.update(self.task_id, advance=n)

    def set_description(self, desc, refresh=True):
        if self.task_id is None:
            return
        self.printer.update(self.task_id, description=desc)
        if refresh:
            self.printer.refresh()

    def refresh(self):
        if self.task_id is None:
            return
        self.printer.refresh()

    @staticmethod
    def format_num(n):
        """
        Intelligent scientific notation (.3g).
        Taken straight from tqdm's format_num

        Parameters
        ----------
        n  : int or float or Numeric
            A Number.

        Returns
        -------
        out  : str
            Formatted number.
        """
        f = "{0:.3g}".format(n).replace("+0", "+").replace("-0", "-")
        n = str(n)
        return f if len(f) < len(n) else n

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict or OrderedDict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
        # Sort in alphabetical order to be more deterministic
        if self.task_id is None:
            return

        postfix = OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = self.format_num(postfix[key])
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        # Stitch together to get the final postfix
        self.printer.update(
            self.task_id,
            postfix=", ".join(
                key + "=" + postfix[key].strip() for key in postfix.keys()
            ),
        )
        if refresh:
            self.printer.refresh()

    def close(self):
        if self.task is not None:
            self.task["disposable"] = True
        self.printer.prune_tasks()
