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


class RichTablePrinter(object):
    def __init__(self, fields={}, key=None):
        self.fields = dict(fields)
        self.key = key
        self.key_to_row_idx = {}
        self.name_to_column_idx = {}
        self.table = None
        self.console = None
        self.live = None
        self.best = {}
        if key not in self.fields:
            self.fields = {key: {}, **fields}

    def _repr_html_(self) -> str:
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
                self.table.add_column(self.fields.get(name, {}).get("name", name), no_wrap=True)
                self.table.columns[-1]._cells = [''] * (len(self.table.columns[0]._cells) if len(self.table.columns) else 0)
                self.name_to_column_idx[name] = len(self.name_to_column_idx)
        new_name_to_column_idx = {}
        columns = []
        def get_name_index(name):
            try:
                return list(self.fields.keys()).index(name)
            except ValueError:
                return len(self.name_to_column_idx)
        for name in sorted(self.name_to_column_idx.keys(), key=get_name_index):
            columns.append(self.table.columns[self.name_to_column_idx[name]])
            new_name_to_column_idx[name] = len(new_name_to_column_idx)
        self.table.columns = columns
        self.name_to_column_idx = new_name_to_column_idx

        if self.key is not None and info[self.key] in self.key_to_row_idx:
            idx = self.key_to_row_idx[info[self.key]]
        else:
            self.table.add_row()
            idx = len(self.table.rows) - 1
            if self.key is not None:
                self.key_to_row_idx[info[self.key]] = idx
        for name, value in info.items():
            field = self.fields.get(name, {})
            formatted_value = field.get("format", "{}").format(value)
            if "goal" in field:
                goal = field["goal"]
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
