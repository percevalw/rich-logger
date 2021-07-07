# rich_logger
Table logger using Rich, aimed at Pytorch Lightning logging

## Features

- display your training logs with pretty [rich](https://github.com/willmcgugan/rich) tables 
- describe your fields with `goal` ("higher_is_better" or "lower_is_better"), `format` and `name`
- a field descriptor can be matched with any regex
- a field name can be computed as a regex substitution
- works in Jupyter notebooks as well as in a command line
- integrates easily with [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## Demo
```python
import time
import random
from rich_logger import RichTablePrinter
logger_fields = {
    "step": {},
    "(.*)_precision": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_p"},
    "(.*)_recall": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_r"},
    "duration": {"format": "{:.1f}", "name": "dur(s)"},
}

def optimization():
    printer = RichTablePrinter(key="step", fields=logger_fields)
    t = time.time()
    for i in range(10):
        time.sleep(random.random())
        printer.log({"step": i, "task_precision": i/10. if i < 5 else 0.5-(i-5)/10.})
        time.sleep(random.random())
        printer.log({"step": i, "task_recall": 0. if i < 3 else (i-3)/10., "duration": time.time() - t})
        t = time.time()
    printer.finalize()
        
optimization()
```
![Demo](demo.gif)

## Use it with PytorchLightning
```python
from rich_logger import RichTableLogger
trainer = pl.Trainer(..., logger=[RichTableLogger(key="epoch", fields=logger_fields)])
```
