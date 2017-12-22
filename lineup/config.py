from _config_section import ConfigSection
from lineup.data.constant import data_dir
import os
REAL_PATH = data_dir

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")
