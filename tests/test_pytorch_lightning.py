import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.demos.boring_classes import BoringModel

from rich_logger import RichTableLogger


def test_pytorch_lightning_logger(tmpdir, capsys):
    seed_everything(43)

    class Model(BoringModel):
        def __init__(self):
            super().__init__()
            self.validation_step_outputs = []

        def validation_step(self, batch, batch_idx):
            self.layer(batch)
            try:
                output = self.layer(batch)
                loss = self.loss(batch, output)
            except Exception:
                pass
            loss = batch_idx * 0.1
            self.validation_step_outputs.append(loss)
            return {"loss": loss, "x": torch.zeros(0)}

        def on_validation_epoch_end(self) -> None:
            self.log(
                "val_loss",
                int(100 * (torch.as_tensor(self.validation_step_outputs).mean())),
            )
            self.validation_step_outputs = []

    model = Model()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=3,
        limit_val_batches=3,
        max_epochs=10,
        log_every_n_steps=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        logger=RichTableLogger(
            fields={
                "epoch": True,
                "step": True,
                ".*": True,
            }
        ),
    )
    trainer.fit(model)

    captured = capsys.readouterr()
    assert captured.err + captured.out == (
        "┏━━━━━━━┳━━━━━━┳━━━━━━━━━━┓\n"
        "┃ epoch ┃ step ┃ val_loss ┃\n"
        "┡━━━━━━━╇━━━━━━╇━━━━━━━━━━┩\n"
        "│ 0     │ 2    │ 10.0     │\n"
        "│ 1     │ 5    │ 10.0     │\n"
        "│ 2     │ 8    │ 10.0     │\n"
        "│ 3     │ 11   │ 10.0     │\n"
        "│ 4     │ 14   │ 10.0     │\n"
        "│ 5     │ 17   │ 10.0     │\n"
        "│ 6     │ 20   │ 10.0     │\n"
        "│ 7     │ 23   │ 10.0     │\n"
        "│ 8     │ 26   │ 10.0     │\n"
        "│ 9     │ 29   │ 10.0     │\n"
        "└───────┴──────┴──────────┘\n"
    )
