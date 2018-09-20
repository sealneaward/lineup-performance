script_path='/home/neil/projects/lineup-performance/lineup/data/nba'
python $script_path/get_on_court.py lineups.yaml
python $script_path/get_lineups.py lineups.yaml
python $script_path/get_pbp.py lineups.yaml
python $script_path/get_matchups.py lineups.yaml
python $script_path/get_abilities.py lineups.yaml
