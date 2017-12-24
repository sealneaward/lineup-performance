from _config_section import ConfigSection
from lineup.data.constant import data_dir
import os
REAL_PATH = data_dir

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")

data.matches = ConfigSection("matches")
data.matches.dir = "%s/%s" % (data.dir, "matches")

data.lineups = ConfigSection("lineups")
data.lineups.dir = "%s/%s" % (data.dir, "lineups")

data.config = ConfigSection("config")
data.config.dir = "%s/%s" % (data.dir, "config")
