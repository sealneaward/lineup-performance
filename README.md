# lineup-performance

Using network embedding techniques, we aim to provide methodology to predict the performance of different lineups.

# Setup

1. Extract the lineup data from *link here*

2. Add the path to the data you extracted to the `lineup/data/constant.py` file.
```py
import os
if os.environ['HOME'] == '/home/neil':
    data_dir = '/home/neil/projects/lineup-performance/lineup'
    # insert data path as branch of if statement
else:
    raise Exception("Unspecified data_dir, unknown environment")
```

3. Install the packages required from the repo dir.
```
python setup.py build
python setup.py install
```
