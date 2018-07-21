from lineup._config_section import ConfigSection
from lineup.data.constant import data_dir
import os
REAL_PATH = data_dir

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")

data.nba = ConfigSection("nba")
data.nba.dir = "%s/%s" % (data.dir, "nba")

data.nhl = ConfigSection("nhl")
data.nhl.dir = "%s/%s" % (data.dir, "nhl")

data.nba.matchups = ConfigSection("matchups")
data.nba.matchups.dir = "%s/%s" % (data.nba.dir, "matchups")

data.nba.lineups = ConfigSection("lineups")
data.nba.lineups.dir = "%s/%s" % (data.nba.dir, "lineups")

data.nhl.matchups = ConfigSection("matchups")
data.nhl.matchups.dir = "%s/%s" % (data.nhl.dir, "matchups")

data.nhl.lineups = ConfigSection("lineups")
data.nhl.lineups.dir = "%s/%s" % (data.nhl.dir, "lineups")

data.config = ConfigSection("config")
data.config.dir = "%s/%s" % (data.dir, "config")

model = ConfigSection("model")
model.dir = "%s/%s" % (REAL_PATH, "model")

model.config = ConfigSection("config")
model.config.dir = "%s/%s" % (model.dir, "config")