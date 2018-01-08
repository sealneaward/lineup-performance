from _config_section import ConfigSection
from lineup.data.constant import data_dir
import os
REAL_PATH = data_dir

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")

data.matchups = ConfigSection("matchups")
data.matchups.dir = "%s/%s" % (data.dir, "matchups")

data.lineups = ConfigSection("lineups")
data.lineups.dir = "%s/%s" % (data.dir, "lineups")

data.config = ConfigSection("config")
data.config.dir = "%s/%s" % (data.dir, "config")
