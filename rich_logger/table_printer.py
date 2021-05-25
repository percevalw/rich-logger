import re

from rich.console import Console
from rich.jupyter import _render_segments
from rich.live import Live
from rich.table import Table


def check_is_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_last_matching_index(matchers, name):
    for i, matcher in reversed(list(enumerate(matchers))):
        if re.match(matcher, name):
            return i
    raise ValueError()


def get_last_matching_value(matchers, name, field, default):
    for matcher, value in reversed(list(matchers.items())):
        if re.match(matcher, name):
            if value is False:
                return matcher, False
            elif field in value:
                return matcher, value[field]
    return None, default


class RichTablePrinter(object):
    def __init__(self, fields={}, key=None):
        """
        Logger based on `rich` tables

        :param key: str or None
            main key to group results by row
        :param fields: dict of (dict or False)
            Field descriptors containing goal ("lower_is_better" or "higher_is_better"), format and display name
            The key is a regex that will be used to match the fields to log
        """

        self.fields = dict(fields)
        self.key = key
        self.key_to_row_idx = {}
        self.name_to_column_idx = {}
        self.table = None
        self.console = None
        self.live = None
        self.best = {}
        if key is not None and key not in self.fields:
            self.fields = {key: {}, **fields}

    def _repr_html_(self) -> str:
        if self.console is None:
            return "Empty table"
        segments = list(self.console.render(self.table, self.console.options))  # type: ignore
        html = _render_segments(segments)
        return html

    def log(self, info):
        if self.table is None:
            is_in_notebook = check_is_in_notebook()
            self.table = Table()
            self.console = Console(width=2 ** 32 - 1 if is_in_notebook else None)

            if is_in_notebook:
                dh = display(None, display_id=True)
                self.refresh = lambda: dh.update(self)
            else:
                self.live = Live(self.table, console=self.console)
                self.live.start()
                self.refresh = lambda: self.live.refresh()
            # self.console = Console()
            # table_centered = Columns((self.table,), align="center", expand=True)
            # self.live = Live(table_centered, console=console)
            # self.live.start()

        for name, value in info.items():
            if name not in self.name_to_column_idx:
                matcher, column_name = get_last_matching_value(self.fields, name, "name", default=name)
                if column_name is False:
                    self.name_to_column_idx[name] = -1
                    continue
                self.table.add_column(re.sub(matcher, column_name, name) if matcher is not None else name, no_wrap=True)
                self.table.columns[-1]._cells = [''] * (len(self.table.columns[0]._cells) if len(self.table.columns) else 0)
                self.name_to_column_idx[name] = (max(self.name_to_column_idx.values()) + 1) if len(self.name_to_column_idx) else 0
        new_name_to_column_idx = {}
        columns = []

        def get_name_index(name):
            try:
                return get_last_matching_index(self.fields, name)
            except ValueError:
                return len(self.name_to_column_idx)

        for name in sorted(self.name_to_column_idx.keys(), key=get_name_index):
            if self.name_to_column_idx[name] >= 0:
                columns.append(self.table.columns[self.name_to_column_idx[name]])
                new_name_to_column_idx[name] = (max(new_name_to_column_idx.values()) + 1) if len(new_name_to_column_idx) else 0
            else:
                new_name_to_column_idx[name] = -1
        self.table.columns = columns
        self.name_to_column_idx = new_name_to_column_idx

        if self.key is not None and self.key in info and info[self.key] in self.key_to_row_idx:
            idx = self.key_to_row_idx[info[self.key]]
        elif self.key is not None and self.key not in info and self.key_to_row_idx:
            idx = list(self.key_to_row_idx.values())[-1]
        else:
            self.table.add_row()
            idx = len(self.table.rows) - 1
            if self.key is not None:
                self.key_to_row_idx[info[self.key]] = idx
        for name, value in info.items():
            if self.name_to_column_idx[name] < 0:
                continue
            formatted_value = get_last_matching_value(self.fields, name, "format", "{}")[1].format(value)
            goal = get_last_matching_value(self.fields, name, "goal", None)[1]
            if goal is not None:
                if name not in self.best:
                    self.best[name] = value
                else:
                    diff = (value - self.best[name]) * (-1 if goal == "lower_is_better" else 1)
                    if diff > 0:
                        self.best[name] = value
                        formatted_value = "[green]" + formatted_value + "[/green]"
                    elif diff <= 0:
                        formatted_value = "[red]" + formatted_value + "[/red]"
            self.table.columns[self.name_to_column_idx[name]]._cells[idx] = formatted_value

        self.refresh()

    def finalize(self):
        if self.live is not None:
            self.live.stop()


if __name__ == "__main__":
    import time

    printer = RichTablePrinter(key="step", fields={"col2": {"name": "COLUMN2"}, "col1": {"name": "COLUMN1"}})
    printer.log({"step": 1, "col1": 4})
    time.sleep(1)
    printer.log({"step": 1, "col1": 3})
    time.sleep(1)
    printer.log({"step": 2, "col2": "ok"})
    time.sleep(1)
    printer.log({"step": 2, "col3": 3.5})
    time.sleep(1)
    printer.log({"step": 3, "col2": "ko", "col1": 4})
    printer.finalize()
