prediction.py  :  predict on county and state level and learn models with whole country data using unacast data
prediction_country.py : predict on country level using google mobility data
prediction_state.py  :  predict on state level and learn models with each state data separately using unacast data
prediction_county.py  :  predict on county level and learn models with each county data separately using unacast data
parallel_state_prediction : predict on state level and learn models with each state data separately (parallel on states)
parallel_county_prediction : predict on county level and learn models with each county data separately (parallel on counties)
makeHistoricalData : base makeHistoricalData code which use unacast data and old rank
old_rank_makeHistoricalData : makeHistoricalData code which use google mobility data and old rank (only handle country mode for now)
new_rank_makeHistoricalData : makeHistoricalData code which use google mobility data and new rank (only handle country mode for now)
models : ML models