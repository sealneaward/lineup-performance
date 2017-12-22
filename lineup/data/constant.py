import os
if os.environ['HOME'] == '/home/neil':
    data_dir = '/home/neil/projects/lineup-performance/lineup'
else:
    raise Exception("Unspecified data_dir, unknown environment")
