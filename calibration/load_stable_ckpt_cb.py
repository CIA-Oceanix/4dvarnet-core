import pytorch_lightning as pl

class LoadStableCheckpoint(pl.Callback):
    def __init__(self, model_ckpt_cb, threshold=1.5):
        prev_val_loss = None
        prev_state_dict = None
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        print(f'{dir(trainer)}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(f'{outputs=}

