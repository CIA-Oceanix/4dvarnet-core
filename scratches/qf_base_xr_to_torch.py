import xarray as xr
import contextlib
import numpy as np
import torch.utils.data
import itertools
import tqdm

class IncompleteScanConfiguration(Exception):
    pass

class DangerousDimOrdering(Exception):
    pass

class XrDataset(torch.utils.data.Dataset):
    """
    torch Dataset based on an xarray.DataArray with on the fly slicing.

    ### Usage: #### 
    If you want to be able to reconstruct the input

    the input xr.DataArray should:
        - have coordinates
        - have the last dims correspond to the patch dims in same order
        - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)

    the batches passed to self.reconstruct should:
        - have the last dims correspond to the patch dims in same order
    """
    def __init__(
            self, da, patch_dims, domain_limits=None, strides=None,
            check_full_scan=False, check_dim_order=False,
            prepro_fn=None, postpro_fn=None
            ):
        """
        da: xarray.DataArray with patch dims at the end in the dim orders
        patch_dims: dict of da dimension to size of a patch 
        domain_limits: dict of da dimension to slices of domain to select for patch extractions
        strides: dict of dims to stride size (default to one)
        check_full_scan: Boolean: if True raise an error if the whole domain is not scanned by the patch size stride combination
        """
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.prepro_fn = prepro_fn
        self.da = da.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        da_dims = dict(zip(da.dims, da.shape))
        self.ds_size = {
                dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
                for dim in patch_dims
                }


        if check_full_scan:
            for dim in patch_dims:
                if (da_dims[dim] - self.patch_dims[dim]) % self.strides.get(dim, 1) != 0:
                    raise IncompleteScanConfiguration(
                        f"""
                        Incomplete scan in dimension dim {dim}:
                        dataarray shape on this dim {da_dims[dim]}
                        patch_size along this dim {self.patch_dims[dim]}
                        stride along this dim {self.strides.get(dim, 1)}
                        [shape - patch_size] should be divisible by stride
                        """
                    )

        if check_dim_order:
            for dim in patch_dims:
                if not '#'.join(da.dims).endswith('#'.join(list(patch_dims))): 
                    raise DangerousDimOrdering(
                        f"""
                        input dataarray's dims should end with patch_dims 
                        dataarray's dim {da.dims}:
                        patch_dims {list(patch_dims)}
                        """
                )
    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {
                dim: slice(self.strides.get(dim, 1) * idx,
                           self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                for dim, idx in zip(self.ds_size.keys(),
                                    np.unravel_index(item, tuple(self.ds_size.values())))
                }
        item =  self.da.isel(**sl)
        if self.prepro_fn is not None:
            item = self.prepro_fn(item)

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

    def reconstruct(self, batches, weight=None):
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims

    batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting 
        """

        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    def reconstruct_from_items(self, items, weight=None):
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

        coords = self.get_coords()

        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)

        das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
               for  it, co in zip(items, coords)]

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
                np.zeros([*new_shape.values(), *da_shape.values()]),
                dims=dims,
                coords={d: self.da[d] for d in self.patch_dims} 
        )
        count_da = xr.zeros_like(rec_da)

        for da in tqdm.tqdm(das):
            rec_da.loc[da.coords] = rec_da.sel(da.coords) + da * w
            count_da.loc[da.coords] = count_da.sel(da.coords) + w

        return rec_da / count_da

class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """
    def reconstruct(self, batches, weight=None):
        """
        Returns list of xarray object, reconstructed from batches
        """
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.reconstruct_from_items(ds_items, weight))
    
        return rec_das

def simple_test():
    """
    Creates a DataArray with 3 dimensions
    Creates a torch.dataset that scan 3x3 patches from the first 2 dimensions
    """

    xrds = xr.Dataset(
        dict(
            v1=(('d1', 'd2', 'd3'), np.random.rand(1000, 1000, 100)),
        ),
        coords=dict(d1=np.arange(1000, 2000)
                    ,d2=np.arange(1000)
                    ,d3=np.arange(100))
    )

    torch_ds = XrDataset(
        xrds.v1, patch_dims=dict(d1=3, d2=3),
        domain_limits=dict(d3=slice(0,50)),
        strides=dict(d1=3, d2=3),
    )

    item = torch_ds[0]


    assert len(torch_ds) == (1000//3) **2, "The dataset does all possible 3x3 patches -> the last coordinates is not in a patch"
    assert torch_ds[0].shape == (3, 3, 51), "The first 2 dimensions are patch dimensions the last one is the full domain within domain limits"


def reconstruction_test():
    """
    Creates a  torch dataset from a xarray.DataArray
    Create a dataloader a get all the batches
    reconstruct the inputs from the batches
    """
    xrds = xr.Dataset(
        dict(
            v1=(('d1', 'd2', 'd3'), np.random.rand(100, 100, 100)),
        ),
        coords=dict(d1=np.arange(100, 200)
                    ,d2=np.arange(100)
                    ,d3=np.arange(100))
    )

    torch_ds = XrDataset(
        xrds.v1.transpose('d3', 'd1', 'd2'), 
        patch_dims=dict(d1=5, d2=5),
        domain_limits=dict(d3=slice(0,50)),
        strides=dict(d1=5, d2=5),
    )

    torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=2)

    batches = [b for b in torch_dl]
    rec_da = torch_ds.reconstruct(batches)

    assert np.allclose(rec_da.values, torch_ds.da.values), "the reconstruction matches the original dataarray"

def reconstruction_with_overlap_test():
    """
    Creates a  torch dataset with overlap from a xarray.DataArray
    Create a dataloader a get all the batches
    reconstruct the inputs from the batches
    """
    xrds = xr.Dataset(
        dict(
            v1=(('d1', 'd2', 'd3'), np.random.rand(100, 100, 100)),
        ),
        coords=dict(d1=np.arange(100)
                    ,d2=np.arange(100)
                    ,d3=np.arange(100))
    )

    torch_ds = XrDataset(
            xrds.v1.transpose('d3', 'd1', 'd2'), 
            patch_dims=dict(d1=10, d2=10),
            domain_limits=dict(d3=slice(0,50)),
            strides=dict(d1=5, d2=5),
            )

    torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=2)
    batches = [b for b in torch_dl]
    rec_da = torch_ds.reconstruct(batches)

    assert np.allclose(rec_da.values, torch_ds.da.values), "the reconstruction matches the original dataarray"

def check_complete_scan_test():
    """
    Creates a configuration with incomplete scan
    Checks the Exception is raised

    """

    xrds = xr.Dataset(
            dict(
                v1=(('d1', 'd2', 'd3'), np.random.rand(1000, 1000, 100)),
                ),
            coords=dict(d1=np.arange(1000)
                        ,d2=np.arange(1000)
                        ,d3=np.arange(100))
            )
    try:
        torch_ds = XrDataset(
                xrds.v1, patch_dims=dict(d1=3, d2=3),
                domain_limits=dict(d3=slice(0,50)),
                strides=dict(d1=3, d2=3),
                check_full_scan=True,
                )
        assert False, "The dataset creation should fail an exception should be raised"
    except Exception as e:
        assert isinstance(e, IncompleteScanConfiguration), "The exception raised should be specific"

def check_dim_order_test():
    """
    Creates a configuration with wrong dim_order
    Checks the Exception is raised
    """

    xrds = xr.Dataset(
            dict(
                v1=(('d1', 'd2', 'd3'), np.random.rand(1000, 1000, 100)),
                ),
            coords=dict(d1=np.arange(1000)
                        ,d2=np.arange(1000)
                        ,d3=np.arange(100))
            )
    try:
        torch_ds = XrDataset(
                xrds.v1, patch_dims=dict(d1=3, d2=3),
                domain_limits=dict(d3=slice(0,50)),
                strides=dict(d1=3, d2=3),
                check_dim_order=True,
                )
        assert False, "The dataset creation should fail an exception should be raised"
    except Exception as e:
        print(e)
        assert isinstance(e, DangerousDimOrdering), "The exception raised should be specific"


def concat_ds_test():
    """
    Creates dataset from multiple dataarrays 
    reconstruct inputs from batches
    """
    xrds1 = xr.Dataset(
        dict(
            v1=(('d1', 'd2', 'd3'), np.random.rand(100, 100, 100)),
        ),
        coords=dict(d1=np.arange(100)
                    ,d2=np.arange(100)
                    ,d3=np.arange(100))
    )

    xrds2 = xr.Dataset(
        dict(
            v1=(('d1', 'd2', 'd3'),
                np.random.rand(100, 100, 100)),
        ),
        coords=dict(d1=np.arange(1000, 1100)
                    ,d2=np.arange(1000, 1100)
                    ,d3=np.arange(100))
    )

    torch_ds1 = XrDataset(
        xrds1.v1.transpose('d3', 'd1', 'd2'), 
        patch_dims=dict(d1=10, d2=10),
        domain_limits=dict(d3=slice(0,50)),
        strides=dict(d1=5, d2=5),
    )

    torch_ds2 = XrDataset(
        xrds2.v1.transpose('d3', 'd1', 'd2'), 
        patch_dims=dict(d1=10, d2=10),
        domain_limits=dict(d3=slice(0,50)),
        strides=dict(d1=5, d2=5),
    )

    torch_ds = XrConcatDataset([torch_ds1, torch_ds2])
    torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=2)

    batches = [b for b in torch_dl]
    rec_das = torch_ds.reconstruct(batches)

    assert np.allclose(rec_das[0].values, torch_ds1.da.values), "the reconstruction matches the original dataarray"
    assert np.allclose(rec_das[1].values, torch_ds2.da.values), "the reconstruction matches the original dataarray"

if __name__ =='__main__':
    simple_test()
    reconstruction_test()
    reconstruction_with_overlap_test()
    check_complete_scan_test()
    check_dim_order_test()
    concat_ds_test()
    print("All test OK")
