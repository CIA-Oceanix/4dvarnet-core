import xarray as xr

def coords_to_dim(ds, dims=('time',), drop=('x',)):
    df = ds.to_dataframe()

    for d in dims:
        df = df.set_index(d, append=True)
    return (
        df.reset_index(level=drop, drop=True)
            .pipe(lambda ddf: xr.Dataset.from_dataframe(ddf))
    )

def reindex(ds, dims=('time', 'lat', 'lon')):
    df = ds.to_dataframe().reset_index()

    for i, d in enumerate(dims):
        df = df.set_index(d, append=i>0)
    return df.pipe(lambda ddf: xr.Dataset.from_dataframe(ddf))

