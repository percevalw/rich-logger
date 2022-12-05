import time

from tqdm import trange

from rich_logger import RichTablePrinter


def test_simple(capsys):
    with RichTablePrinter(
        key="step",
        fields={
            "col2": {"name": "COLUMN2"},
            "col1": {"name": "COLUMN1"},
            "col4": False,
            "col5": True,
        },
    ) as printer:
        printer.log({"step": 1, "col1": 4})
        time.sleep(1)
        printer.log({"step": 1, "col1": 3})
        time.sleep(1)
        printer.log({"step": 2, "col2": "ok"})
        time.sleep(1)
        printer.log({"step": 2, "col3": 3.5})
        time.sleep(1)
        printer.log({"step": 3, "col2": "ko", "col1": 4, "col4": "ok", "col5": 3})
    captured = capsys.readouterr()
    assert captured.out == (
        "┏━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓\n"
        "┃ step ┃ COLUMN2 ┃ COLUMN1 ┃ col5 ┃\n"
        "┡━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩\n"
        "│ 1    │         │ 3       │      │\n"
        "│ 2    │ ok      │         │      │\n"
        "│ 3    │ ko      │ 4       │ 3    │\n"
        "└──────┴─────────┴─────────┴──────┘\n"
    )
    assert captured.err == ""


def test_no_context(capsys):
    logger_fields = {
        "step": {},
        "(.*)_precision": {
            "goal": "higher_is_better",
            "format": "{:.4f}",
            "name": r"\1_p",
        },
        "(.*)_recall": {
            "goal": "higher_is_better",
            "format": "{:.4f}",
            "name": r"\1_r",
        },
        "duration": {"format": "{:.1f}", "name": "dur(s)"},
        ".*": True,
    }

    def optimization():
        printer = RichTablePrinter(key="step", fields=logger_fields)
        t = time.time()
        for i in range(10):
            time.sleep(0.1)
            printer.log(
                {
                    "step": i,
                    "task_precision": i / 10.0 if i < 5 else 0.5 - (i - 5) / 10.0,
                }
            )
            time.sleep(0.1)
            printer.log(
                {
                    "step": i,
                    "task_recall": 0.0 if i < 3 else (i - 3) / 10.0,
                    "duration": time.time() - t,
                }
            )
            t = time.time()
        printer.finalize()

    optimization()

    captured = capsys.readouterr()

    assert captured.out == (
        "┏━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓\n"
        "┃ step ┃ task_p ┃ task_r ┃ dur(s) ┃\n"
        "┡━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩\n"
        "│ 0    │ 0.0000 │ 0.0000 │ 0.2    │\n"
        "│ 1    │ 0.1000 │ 0.0000 │ 0.2    │\n"
        "│ 2    │ 0.2000 │ 0.0000 │ 0.2    │\n"
        "│ 3    │ 0.3000 │ 0.0000 │ 0.2    │\n"
        "│ 4    │ 0.4000 │ 0.1000 │ 0.2    │\n"
        "│ 5    │ 0.5000 │ 0.2000 │ 0.2    │\n"
        "│ 6    │ 0.4000 │ 0.3000 │ 0.2    │\n"
        "│ 7    │ 0.3000 │ 0.4000 │ 0.2    │\n"
        "│ 8    │ 0.2000 │ 0.5000 │ 0.2    │\n"
        "│ 9    │ 0.1000 │ 0.6000 │ 0.2    │\n"
        "└──────┴────────┴────────┴────────┘\n"
    )
    assert captured.err == ""


def test_tqdm(capsys):
    logger_fields = {
        "step": {},
        "(.*)_recall": {
            "goal": "higher_is_better",
            "format": "{:.4f}",
            "name": r"\1_r",
        },
        "(.*)_precision": {
            "goal": "higher_is_better",
            "format": "{:.4f}",
            "name": r"\1_p",
        },
        "duration": {"format": "{:.1f}", "name": "dur(s)"},
        ".*": True,
    }

    def optimization():
        with RichTablePrinter(key="step", fields=logger_fields) as printer:
            t = time.time()
            for i in trange(10):
                time.sleep(0.1)
                printer.log(
                    {
                        "step": i,
                        "task_precision": i / 10.0 if i < 5 else 0.5 - (i - 5) / 10.0,
                    }
                )
                time.sleep(0.1)
                printer.log(
                    {
                        "step": i,
                        "task_recall": 0.0 if i < 3 else (i - 3) / 10.0,
                        "duration": time.time() - t,
                    }
                )
                printer.log({"test": i})
                t = time.time()
                for j in trange(5):
                    pass

    optimization()

    captured = capsys.readouterr()

    assert captured.out == (
        "┏━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━┓\n"
        "┃ step ┃ task_r ┃ task_p ┃ dur(s) ┃ test ┃\n"
        "┡━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━┩\n"
        "│ 0    │ 0.0000 │ 0.0000 │ 0.2    │ 0    │\n"
        "│ 1    │ 0.0000 │ 0.1000 │ 0.2    │ 1    │\n"
        "│ 2    │ 0.0000 │ 0.2000 │ 0.2    │ 2    │\n"
        "│ 3    │ 0.0000 │ 0.3000 │ 0.2    │ 3    │\n"
        "│ 4    │ 0.1000 │ 0.4000 │ 0.2    │ 4    │\n"
        "│ 5    │ 0.2000 │ 0.5000 │ 0.2    │ 5    │\n"
        "│ 6    │ 0.3000 │ 0.4000 │ 0.2    │ 6    │\n"
        "│ 7    │ 0.4000 │ 0.3000 │ 0.2    │ 7    │\n"
        "│ 8    │ 0.5000 │ 0.2000 │ 0.2    │ 8    │\n"
        "│ 9    │ 0.6000 │ 0.1000 │ 0.2    │ 9    │\n"
        "└──────┴────────┴────────┴────────┴──────┘\n"
        " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00  \n"
        " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00  \n"
    )
