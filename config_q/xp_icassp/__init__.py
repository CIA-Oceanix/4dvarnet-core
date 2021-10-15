import pandas as pd

dT = 5
# Specify the dataset spatial bounds
dim_range = {
    'lat': slice(33, 43),
    'lon': slice(-65, -55),
}

# Specify the batch patch size
slice_win = {
    'time': 5,
    'lat': 200,
    'lon': 200,
}
# Specify the stride between two patches
strides = {
    'time': 1,
    'lat': 200,
    'lon': 200,
}

test_dates = [str(dt.date()) for dt in pd.date_range('2013-01-03', "2013-01-27")]
