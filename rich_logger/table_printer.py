import re
import typing
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from pydantic import BaseModel, Extra
from rich import get_console
from rich.console import RenderableType
from rich.control import Control
from rich.jupyter import _render_segments
from rich.live import Live
from rich.progress import Progress
from rich.table import Table

T = typing.TypeVar("T")


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

        super().__init__()
        self.live.__class__ = FixedLive

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        if self.logger_table is not None:
            yield self.logger_table
        tasks_table = self.make_tasks_table(self.tasks)
        yield tasks_table

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
        if total is None:
            try:
                total = len(iterable)
            except AttributeError:
                pass

        # Prune displayed & finished tasks
        remaining_tasks = []
        for last_task in self.displayed_tasks:
            if last_task["disposable"]:
                self.remove_task(last_task["task_id"])
            else:
                remaining_tasks.append(last_task)
        self.displayed_tasks = remaining_tasks

        task_id = self.add_task(
            description=description or "",
            total=total,
        )
        task = {"task_id": task_id, "disposable": False}
        self.displayed_tasks.append(task)
        try:
            for item in iterable:
                yield item
                self.update(task_id, advance=1)
        finally:
            if not leave:
                task["disposable"] = True

    def log_metrics(self, info: Dict[str, Any]):
        """
        Adds or update a row in the table

        Parameters
        ----------
        info: Dict[str, Any]
            The value of each column
        """
        if self.logger_table is None:
            self.logger_table = Table()
            self.console.clear_live()
            self.start()

            # if is_in_notebook:
            #     self.display_handle = display(None, display_id=True)

        for name, value in info.items():
            if name not in self.name_to_column_idx:
                matcher, column_name = get_last_matching_value(
                    self.rules, name, "name", default=name
                )
                if column_name is False:
                    self.name_to_column_idx[name] = -1
                    continue
                self.logger_table.add_column(
                    re.sub(matcher, column_name, name) if matcher is not None else name,
                    no_wrap=True,
                )
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

    def finalize(self):
        self.refresh()
        if self.live is not None:
            self.live.stop()
        try:
            import tqdm
        except ImportError:
            pass
        else:
            if self._old_tqdm_new is not None:
                tqdm.tqdm.__new__ = self._old_tqdm_new

    def hijack_tqdm(self):
        """
        Monkey patch tqdm to use the progress bar method of this class instead
        """
        import tqdm

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
            return logger.progress_bar(
                iterable=iterable,
                description=desc,
                total=total,
                leave=leave,
            )

        tqdm.tqdm.__new__ = __new__

    def __enter__(self) -> "RichTablePrinter":
        super().__enter__()
        self.hijack_tqdm()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.finalize()
