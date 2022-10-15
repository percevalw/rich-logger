import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.demos.boring_classes import BoringModel

from rich_logger import RichTableLogger


def test_pytorch_lightning_logger(tmpdir, capsys):
    seed_everything(43)

    class Model(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def validation_epoch_end(self, outputs) -> None:
            self.log(
                "val_loss",
                int(100 * (torch.stack([x["loss"] for x in outputs]).mean())),
            )

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
            }
        ),
    )
    trainer.fit(model)

    captured = capsys.readouterr()
    assert captured.err + captured.out == (
        "┏━━━━━━━┳━━━━━━┳━━━━━━━━━━┓\n"
        "┃ epoch ┃ step ┃ val_loss ┃\n"
        "┡━━━━━━━╇━━━━━━╇━━━━━━━━━━┩\n"
        "│ 0     │ 2    │ 622.0    │\n"
        "│ 1     │ 5    │ 390.0    │\n"
        "│ 2     │ 8    │ 379.0    │\n"
        "│ 3     │ 11   │ 379.0    │\n"
        "│ 4     │ 14   │ 378.0    │\n"
        "│ 5     │ 17   │ 378.0    │\n"
        "│ 6     │ 20   │ 378.0    │\n"
        "│ 7     │ 23   │ 378.0    │\n"
        "│ 8     │ 26   │ 378.0    │\n"
        "│ 9     │ 29   │ 378.0    │\n"
        "└───────┴──────┴──────────┘\n"
    )
