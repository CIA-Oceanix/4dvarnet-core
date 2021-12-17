import torch
from torch import tensor

outputs0 = [(tensor([1, 3, 5, 7], device='cuda:6'), 0), (tensor([ 8, 10, 12, 14], device='cuda:6'), 1), (tensor([16, 18], device='cuda:6'), 2)]

outputs1 = [(tensor([0, 2, 4, 6], device='cuda:6'), 0), (tensor([ 8, 10, 12, 14], device='cuda:6'), 1), (tensor([16, 18], device='cuda:6'), 2)]



x = [outputs0, outputs1]

def iter_item(outputs):
    n_batch_chunk = len(outputs)
    n_batch = len(outputs[0])
    for b in range(n_batch):
        bs = outputs[0][b][0].shape[0]
        for i in range(bs):
            for bc in range(n_batch_chunk):
                print(b, i, bc)
                yield (
                        outputs[bc][b][0][i].cpu().item(),
                        outputs[bc][b][1],
                )

print(list(iter_item(x)))
# torch.tensor
# ds = torch.utils.data.TensorDataset(torch.arange(20))
# dl = torch.utils.data.DataLoader(ds, batch_size=4)

# tr = pl.Trainer(gpus=[6,7], strategy='ddp')

# mod = LitModel()

# tr.test(mod, dataloaders=dl)
