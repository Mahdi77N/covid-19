from makeHistoricalData import makeHistoricalData
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os
import pandas as pd
import numpy as np

def main():
    
    data = makeHistoricalData(4, 5, 2, 'death', 'mrmr', 'country', 'weeklyaverage', './',
                              [], 'county')
    names = pd.read_csv('./country_numbers.csv')
    methods = ['GBM', 'GLM', 'KNN', 'NN', 'MM_GLM', 'MM_NN']
    countries = names['name'].unique()
    
    for r in range(1,10):
        
        maxHistory = min(5,(10-r))
        root_add = 'r = ' + str(r) + './' + 'results/max_history=' + str(maxHistory) + '/test/all_errors/'
        if os.path.exists(root_add):
            print('exists!')
            errors = pd.DataFrame(columns=methods,index=countries)
            template = pd.DataFrame(columns = ['Unnamed: 0','date of day t','county_fips','Target','prediction','error','absoulte_error','percentage_error'])
            predictions = {i:template for i in methods}

            for method in methods:
                address = root_add + method + '/all_errors_'+method+'.csv'
                predictions[method] = pd.read_csv(address)
                
                for country in countries:
                    data = predictions[method]
                    data = pd.merge(data,names,right_on=['number'],left_on=['county_fips'], sort=True)
                    y_test = np.array(data[data['name']==country]['Target'])
                    y_prediction = np.array(data[data['name']==country]['prediction'])
                    y_prediction[y_prediction < 0] = 0
                    sumOfAbsoluteError = sum(abs(y_test - y_prediction))
                    percentageOfAbsoluteError = np.mean((abs(y_test - y_prediction)/y_test)*100)
                    errors.loc[country,method] = percentageOfAbsoluteError
                    
            for col in errors.columns:
                errors[col]=errors[col].apply(lambda x:np.round(x,2))
                
            errors = errors.dropna()
            er = errors.T
            errors['min_error']=er.min()
            errors = errors.sort_values(by=['min_error'])
            errors.to_csv('r = ' + str(r) + './' + 'results/errors.csv')
            
if __name__ == "__main__":
    
    main()
