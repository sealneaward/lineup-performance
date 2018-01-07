import pandas as pd
import re

PLAYER_RE = r'\w{0,7}\d{2}'

def parse_play(play):
	"""Parse play details from a play-by-play string describing a play.
	Assuming valid input, this function returns structured data in a dictionary
	describing the play. If the play detail string was invalid, this function
	returns None.
	:param details: detail string for the play
	:param is_hm: bool indicating whether the offense is at home
	:param returns: dictionary of play attributes or None if invalid
	:rtype: dictionary or None

	SOURCE: https://github.com/MikeRa1979/SportsScrape/blob/master/sportsref/nba/pbp.py
	"""
	if play['SCORE'] is None:
		return pd.DataFrame()
	elif play['HOMEDESCRIPTION'] is None:
		aw = True
		hm = False
		is_hm = False
		details = play['VISITORDESCRIPTION']
	else:
		hm = True
		aw = False
		is_hm = True
		details = play['HOMEDESCRIPTION']

	# if input isn't a string, return None
	if not details or not isinstance(details, basestring):
		return None


	p = {}
	p['detail'] = details
	p['home'] = hm
	p['away'] = aw
	p['is_home_play'] = is_hm

	# parsing field goal attempts
	shotRE = (r'(?P<shooter>{0}) (?P<is_fgm>makes|misses) '
			  '(?P<is_three>2|3)\-pt shot').format(PLAYER_RE)
	distRE = r' (?:from (?P<shot_dist>\d+) ft|at rim)'
	assistRE = r' \(assist by (?P<assister>{0})\)'.format(PLAYER_RE)
	blockRE = r' \(block by (?P<blocker>{0})\)'.format(PLAYER_RE)
	shotRE = r'{0}{1}(?:{2}|{3})?'.format(shotRE, distRE, assistRE, blockRE)
	m = re.match(shotRE, details, re.IGNORECASE)
	if m:
		p['is_fga'] = True
		p.update(m.groupdict())
		p['shot_dist'] = p['shot_dist'] if p['shot_dist'] is not None else 0
		p['shot_dist'] = int(p['shot_dist'])
		p['is_fgm'] = p['is_fgm'] == 'makes'
		p['is_three'] = p['is_three'] == '3'
		p['is_assist'] = pd.notnull(p.get('assister'))
		p['is_block'] = pd.notnull(p.get('blocker'))
		shooter_home = p['shooter'] in ['']
		p['off_team'] = hm if shooter_home else aw
		p['def_team'] = aw if shooter_home else hm
		return p
	#
	# # parsing jump balls
	# jumpRE = ((r'Jump ball: (?P<away_jumper>{0}) vs\. (?P<home_jumper>{0})'
	# 		   r'(?: \((?P<gains_poss>{0}) gains possession\))?')
	# 		  .format(PLAYER_RE))
	# m = re.match(jumpRE, details, re.IGNORECASE)
	# if m:
	# 	p['is_jump_ball'] = True
	# 	p.update(m.groupdict())
	# 	return p
	#
	# # parsing rebounds
	# rebRE = (r'(?P<is_oreb>Offensive|Defensive) rebound'
	# 		 r' by (?P<rebounder>{0}|Team)').format(PLAYER_RE)
	# m = re.match(rebRE, details, re.I)
	# if m:
	# 	p['is_reb'] = True
	# 	p.update(m.groupdict())
	# 	p['is_oreb'] = p['is_oreb'].lower() == 'offensive'
	# 	p['is_dreb'] = not p['is_oreb']
	# 	if p['rebounder'] == 'Team':
	# 		p['reb_team'], other = (hm, aw) if is_hm else (aw, hm)
	# 	else:
	# 		reb_home = p['rebounder'] in hm_roster
	# 		p['reb_team'], other = (hm, aw) if reb_home else (aw, hm)
	# 	p['off_team'] = p['reb_team'] if p['is_oreb'] else other
	# 	p['def_team'] = p['reb_team'] if p['is_dreb'] else other
	# 	return p
	#
	# # parsing free throws
	# ftRE = (r'(?P<ft_shooter>{}) (?P<is_ftm>makes|misses) '
	# 		r'(?P<is_tech_fta>technical )?(?P<is_flag_fta>flagrant )?'
	# 		r'(?P<is_clearpath_fta>clear path )?free throw'
	# 		r'(?: (?P<fta_num>\d+) of (?P<tot_fta>\d+))?').format(PLAYER_RE)
	# m = re.match(ftRE, details, re.I)
	# if m:
	# 	p['is_fta'] = True
	# 	p.update(m.groupdict())
	# 	p['is_ftm'] = p['is_ftm'] == 'makes'
	# 	p['is_tech_fta'] = bool(p['is_tech_fta'])
	# 	p['is_flag_fta'] = bool(p['is_flag_fta'])
	# 	p['is_clearpath_fta'] = bool(p['is_clearpath_fta'])
	# 	p['is_pf_fta'] = not p['is_tech_fta']
	# 	if p['tot_fta']:
	# 		p['tot_fta'] = int(p['tot_fta'])
	# 	if p['fta_num']:
	# 		p['fta_num'] = int(p['fta_num'])
	# 	ft_home = p['ft_shooter'] in hm_roster
	# 	p['fta_team'] = hm if ft_home else aw
	# 	if not p['is_tech_fta']:
	# 		p['off_team'] = hm if ft_home else aw
	# 		p['def_team'] = aw if ft_home else hm
	# 	return p
	#
	# # parsing substitutions
	# subRE = (r'(?P<sub_in>{0}) enters the game for '
	# 		 r'(?P<sub_out>{0})').format(PLAYER_RE)
	# m = re.match(subRE, details, re.I)
	# if m:
	# 	p['is_sub'] = True
	# 	p.update(m.groupdict())
	# 	sub_home = p['sub_in'] in hm_roster or p['sub_out'] in hm_roster
	# 	p['sub_team'] = hm if sub_home else aw
	# 	return p
	#
	# # parsing turnovers
	# toReasons = (r'(?P<to_type>[^;]+)(?:; steal by '
	# 			 r'(?P<stealer>{0}))?').format(PLAYER_RE)
	# toRE = (r'Turnover by (?P<to_by>{}|Team) '
	# 		r'\((?:{})\)').format(PLAYER_RE, toReasons)
	# m = re.match(toRE, details, re.I)
	# if m:
	# 	p['is_to'] = True
	# 	p.update(m.groupdict())
	# 	p['to_type'] = p['to_type'].lower()
	# 	if p['to_type'] == 'offensive foul':
	# 		return None
	# 	p['is_steal'] = pd.notnull(p['stealer'])
	# 	p['is_travel'] = p['to_type'] == 'traveling'
	# 	p['is_shot_clock_viol'] = p['to_type'] == 'shot clock'
	# 	p['is_oob'] = p['to_type'] == 'step out of bounds'
	# 	p['is_three_sec_viol'] = p['to_type'] == '3 sec'
	# 	p['is_backcourt_viol'] = p['to_type'] == 'back court'
	# 	p['is_off_goaltend'] = p['to_type'] == 'offensive goaltending'
	# 	p['is_double_dribble'] = p['to_type'] == 'dbl dribble'
	# 	p['is_discont_dribble'] = p['to_type'] == 'discontinued dribble'
	# 	p['is_carry'] = p['to_type'] == 'palming'
	# 	if p['to_by'] == 'Team':
	# 		p['off_team'] = hm if is_hm else aw
	# 		p['def_team'] = aw if is_hm else hm
	# 	else:
	# 		to_home = p['to_by'] in hm_roster
	# 		p['off_team'] = hm if to_home else aw
	# 		p['def_team'] = aw if to_home else hm
	# 	return p
	#
	# # parsing shooting fouls
	# shotFoulRE = (r'Shooting(?P<is_block_foul> block)? foul by (?P<fouler>{0})'
	# 			  r'(?: \(drawn by (?P<drew_foul>{0})\))?').format(PLAYER_RE)
	# m = re.match(shotFoulRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p['is_shot_foul'] = True
	# 	p.update(m.groupdict())
	# 	p['is_block_foul'] = bool(p['is_block_foul'])
	# 	foul_on_home = p['fouler'] in hm_roster
	# 	p['off_team'] = aw if foul_on_home else hm
	# 	p['def_team'] = hm if foul_on_home else aw
	# 	p['foul_team'] = p['def_team']
	# 	return p
	#
	# # parsing offensive fouls
	# offFoulRE = (r'Offensive(?P<is_charge> charge)? foul '
	# 			 r'by (?P<to_by>{0})'
	# 			 r'(?: \(drawn by (?P<drew_foul>{0})\))?').format(PLAYER_RE)
	# m = re.match(offFoulRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p['is_off_foul'] = True
	# 	p['is_to'] = True
	# 	p['to_type'] = 'offensive foul'
	# 	p.update(m.groupdict())
	# 	p['is_charge'] = bool(p['is_charge'])
	# 	p['fouler'] = p['to_by']
	# 	foul_on_home = p['fouler'] in hm_roster
	# 	p['off_team'] = hm if foul_on_home else aw
	# 	p['def_team'] = aw if foul_on_home else hm
	# 	p['foul_team'] = p['off_team']
	# 	return p
	#
	# # parsing personal fouls
	# foulRE = (r'Personal (?P<is_take_foul>take )?(?P<is_block_foul>block )?'
	# 		  r'foul by (?P<fouler>{0})(?: \(drawn by '
	# 		  r'(?P<drew_foul>{0})\))?').format(PLAYER_RE)
	# m = re.match(foulRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p.update(m.groupdict())
	# 	p['is_take_foul'] = bool(p['is_take_foul'])
	# 	p['is_block_foul'] = bool(p['is_block_foul'])
	# 	foul_on_home = p['fouler'] in hm_roster
	# 	p['off_team'] = aw if foul_on_home else hm
	# 	p['def_team'] = hm if foul_on_home else aw
	# 	p['foul_team'] = p['def_team']
	# 	return p
	#
	# # TODO: parsing double personal fouls
	# # double_foul_re = (r'Double personal foul by (?P<fouler1>{0}) and '
	# #                   r'(?P<fouler2>{0})').format(PLAYER_RE)
	# # m = re.match(double_Foul_re, details, re.I)
	# # if m:
	# #     p['is_pf'] = True
	# #     p.update(m.groupdict())
	# #     p['off_team'] =
	#
	# # parsing loose ball fouls
	# looseBallRE = (r'Loose ball foul by (?P<fouler>{0})'
	# 			   r'(?: \(drawn by (?P<drew_foul>{0})\))?').format(PLAYER_RE)
	# m = re.match(looseBallRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p['is_loose_ball_foul'] = True
	# 	p.update(m.groupdict())
	# 	foul_home = p['fouler'] in hm_roster
	# 	p['foul_team'] = hm if foul_home else aw
	# 	return p
	#
	# # parsing punching fouls
	# # TODO
	#
	# # parsing away from play fouls
	# awayFromBallRE = ((r'Away from play foul by (?P<fouler>{0})'
	# 				   r'(?: \(drawn by (?P<drew_foul>{0})\))?')
	# 				  .format(PLAYER_RE))
	# m = re.match(awayFromBallRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p['is_away_from_play_foul'] = True
	# 	p.update(m.groupdict())
	# 	foul_on_home = p['fouler'] in hm_roster
	# 	# TODO: figure out who had the ball based on previous play
	# 	p['foul_team'] = hm if foul_on_home else aw
	# 	return p
	#
	# # parsing inbound fouls
	# inboundRE = (r'Inbound foul by (?P<fouler>{0})'
	# 			 r'(?: \(drawn by (?P<drew_foul>{0})\))?').format(PLAYER_RE)
	# m = re.match(inboundRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p['is_inbound_foul'] = True
	# 	p.update(m.groupdict())
	# 	foul_on_home = p['fouler'] in hm_roster
	# 	p['off_team'] = aw if foul_on_home else hm
	# 	p['def_team'] = hm if foul_on_home else aw
	# 	p['foul_team'] = p['def_team']
	# 	return p
	#
	# # parsing flagrant fouls
	# flagrantRE = (r'Flagrant foul type (?P<flag_type>1|2) by (?P<fouler>{0})'
	# 			  r'(?: \(drawn by (?P<drew_foul>{0})\))?').format(PLAYER_RE)
	# m = re.match(flagrantRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p['is_flagrant'] = True
	# 	p.update(m.groupdict())
	# 	foul_on_home = p['fouler'] in hm_roster
	# 	p['foul_team'] = hm if foul_on_home else aw
	# 	return p
	#
	# # parsing clear path fouls
	# clearPathRE = (r'Clear path foul by (?P<fouler>{0})'
	# 			   r'(?: \(drawn by (?P<drew_foul>{0})\))?').format(PLAYER_RE)
	# m = re.match(clearPathRE, details, re.I)
	# if m:
	# 	p['is_pf'] = True
	# 	p['is_clear_path_foul'] = True
	# 	p.update(m.groupdict())
	# 	foul_on_home = p['fouler'] in hm_roster
	# 	p['off_team'] = aw if foul_on_home else hm
	# 	p['def_team'] = hm if foul_on_home else aw
	# 	p['foul_team'] = p['def_team']
	# 	return p
	#
	# # parsing timeouts
	# timeoutRE = r'(?P<timeout_team>.*?) (?:full )?timeout'
	# m = re.match(timeoutRE, details, re.I)
	# if m:
	# 	p['is_timeout'] = True
	# 	p.update(m.groupdict())
	# 	isOfficialTO = p['timeout_team'].lower() == 'official'
	# 	name_to_id = season.team_names_to_ids()
	# 	p['timeout_team'] = (
	# 		'Official' if isOfficialTO else
	# 		name_to_id.get(hm, name_to_id.get(aw, p['timeout_team']))
	# 	)
	# 	return p
	#
	# # parsing technical fouls
	# techRE = (r'(?P<is_hanging>Hanging )?'
	# 		  r'(?P<is_taunting>Taunting )?'
	# 		  r'(?P<is_ill_def>Ill def )?'
	# 		  r'(?P<is_delay>Delay )?'
	# 		  r'(?P<is_unsport>Non unsport )?'
	# 		  r'tech(?:nical)? foul by '
	# 		  r'(?P<tech_fouler>{0}|Team)').format(PLAYER_RE)
	# m = re.match(techRE, details, re.I)
	# if m:
	# 	p['is_tech_foul'] = True
	# 	p.update(m.groupdict())
	# 	p['is_hanging'] = bool(p['is_hanging'])
	# 	p['is_taunting'] = bool(p['is_taunting'])
	# 	p['is_ill_def'] = bool(p['is_ill_def'])
	# 	p['is_delay'] = bool(p['is_delay'])
	# 	p['is_unsport'] = bool(p['is_unsport'])
	# 	foul_on_home = p['tech_fouler'] in hm_roster
	# 	p['foul_team'] = hm if foul_on_home else aw
	# 	return p
	#
	# # parsing ejections
	# ejectRE = r'(?P<ejectee>{0}|Team) ejected from game'.format(PLAYER_RE)
	# m = re.match(ejectRE, details, re.I)
	# if m:
	# 	p['is_ejection'] = True
	# 	p.update(m.groupdict())
	# 	if p['ejectee'] == 'Team':
	# 		p['ejectee_team'] = hm if is_hm else aw
	# 	else:
	# 		eject_home = p['ejectee'] in hm_roster
	# 		p['ejectee_team'] = hm if eject_home else aw
	# 	return p
	#
	# # parsing defensive 3 seconds techs
	# def3TechRE = (r'(?:Def 3 sec tech foul|Defensive three seconds)'
	# 			  r' by (?P<tech_fouler>{})').format(PLAYER_RE)
	# m = re.match(def3TechRE, details, re.I)
	# if m:
	# 	p['is_tech_foul'] = True
	# 	p['is_def_three_secs'] = True
	# 	p.update(m.groupdict())
	# 	foul_on_home = p['tech_fouler'] in hm_roster
	# 	p['off_team'] = aw if foul_on_home else hm
	# 	p['def_team'] = hm if foul_on_home else aw
	# 	p['foul_team'] = p['def_team']
	# 	return p
	#
	# # parsing violations
	# violRE = (r'Violation by (?P<violator>{0}|Team) '
	# 		  r'\((?P<viol_type>.*)\)').format(PLAYER_RE)
	# m = re.match(violRE, details, re.I)
	# if m:
	# 	p['is_viol'] = True
	# 	p.update(m.groupdict())
	# 	if p['viol_type'] == 'kicked_ball':
	# 		p['is_to'] = True
	# 		p['to_by'] = p['violator']
	# 	if p['violator'] == 'Team':
	# 		p['viol_team'] = hm if is_hm else aw
	# 	else:
	# 		viol_home = p['violator'] in hm_roster
	# 		p['viol_team'] = hm if viol_home else aw
	# 	return p
	#
	# p['is_error'] = True
	# return p
