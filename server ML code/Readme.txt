prediction.py  :  predicts on county and state level and learn models with whole country data using unacast data
prediction_country.py : predicts on country level using google mobility data
prediction_state.py  :  predicts on state level and learn models with each state data separately using unacast data
prediction_county.py  :  predicts on county level and learn models with each county data separately using unacast data
parallel_state_prediction : predicts on state level and learn models with each state data separately (parallel on states)
parallel_county_prediction : predicts on county level and learn models with each county data separately (parallel on counties)
old_rank_makeHistoricalData : makeHistoricalData code which uses google mobility data and old rank (ranking based on target variable at day t, only handle country mode for now)
new_rank_makeHistoricalData : makeHistoricalData code which uses google mobility data and new rank (ranking based on target variable at day t+r, only handle country mode for now)
makeHistoricalData : base makeHistoricalData code which uses unacast data and old rank
models : ML models
