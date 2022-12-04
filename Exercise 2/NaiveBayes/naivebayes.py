import numpy as np
import warnings
import math

class NaiveBayes:
    def __init__(self):
        self.probabilities = {}
        self.target_classes = []
        pass
    
    def normal_distr_fn(self, x, mean, sd):
        p = 1 / math.sqrt(2 * math.pi * sd**2)
        return p * np.exp(-0.5 / sd**2 * (x - mean)**2)
    
    def fit(self, df, target_name):
        # final dictionary created
        probabilities = {}

        # all variable names are stored in columns
        columns = list(df.drop(target_name, axis=1).columns)

        # all unique target classes stored
        target_classes = list(df[target_name].unique())
        self.target_classes = target_classes
        
        target_probabilities = []

        # PRIORI pobability
        # store the target probabilities generally speaking
        for _class in target_classes:
            occurence = df[df[target_name] == _class].shape[0]
            target_probabilities.append(occurence/df.shape[0])

        target_probabilities = dict(zip(target_classes, target_probabilities))

        # LIKELIHOOD table for all features
        numeric_variables = []
        nominal_variables = []

        # separate the columns into nominal and numeric
        for column in df.drop(target_name, axis=1).columns:
            if((df[column].dtype == np.int64) or (df[column].dtype == np.float64)):
                numeric_variables.append(column)
            else:
                nominal_variables.append(column)

        # nominal variables
        for variable in nominal_variables:
            dict_of_probabilities = {}
            # save unique varible values
            unique_variable_values = list(df[variable].unique())
            for variable_value in unique_variable_values:
                dict_of_probabilities[variable_value] = {}
                # save the probability for that unique variable value
                dict_of_probabilities[variable_value]['general'] = df[df[variable] == variable_value].shape[0] / df.shape[0]
                # go through every target_class for that unique variable value and calculate prior probabilities
                for _class in target_classes:
                    dict_of_probabilities[variable_value][_class] = df[(df[variable] == variable_value) & (df[target_name] == _class)].shape[0] / df.shape[0]          
            # save it finally at the end to the probabilities main dictionary under the variable key
            self.probabilities[variable] = dict_of_probabilities

        # numeric variables
        for variable in numeric_variables:
            new_dict = {}
            # go through every target class 
            # similar approach here as in nominal variables, except we don't calculate the frequencies of unique variable value
            # ocurring but the mean and std of the filtered dataframe with the target_class as the filter-condition
            for _class in target_classes:
                mean = df[df[target_name] == _class][variable].mean()
                std = df[df[target_name] == _class][variable].std()

                new_dict[_class] = {'mean': mean, 'std': std}

            general_mean = df[variable].mean()
            general_std = df[variable].std()
            new_dict['general'] = {'mean': general_mean, 'std': general_std}

            self.probabilities[variable] = new_dict

        # add the general target_class probabilities as well
        self.probabilities["targets"] = target_probabilities
        print(self.probabilities)
        
    def predict(self, df):
        
        final_predictions = []
        
        for row_index in range(df.shape[0]):
            data_instance = df.iloc[row_index, :]
            probabilities_per_class = {}
            for target_class in self.target_classes:
                # to calcualte posterior probability, first calculate likelihood and evidence
                # from values generated in 'fit' method and multiply for each row value
                prior_probability = self.probabilities['targets'][target_class]
                likelihood = 1
                evidence = 1
                for column in df.columns:
                    # for numeric prediction use previously calculated mean and std in normal distribution
                    # function to calculate probability for the new value
                    if((df[column].dtype == np.int64) or (df[column].dtype == np.float64)):
                        mean = self.probabilities[column][target_class]['mean']
                        std = self.probabilities[column][target_class]['std']
                        x = data_instance[column]
                        x_probability = self.normal_distr_fn(x, mean, std)
                        likelihood = likelihood * x_probability
                        
                        general_mean = self.probabilities[column]['general']['mean']
                        general_std = self.probabilities[column]['general']['std']
                        general_probability = self.normal_distr_fn(x, general_mean, general_std)
                        evidence = evidence * general_probability
                    else:
                        x = data_instance[column]
                        if x in self.probabilities[column]:
                            likelihood = likelihood * self.probabilities[column][x][target_class]
                            evidence = evidence * self.probabilities[column][x]['general']
                        else:
                            likelihood = 0
                            evidence = 0
                
                # calculate posterior and assign to dict for the current target class
                posterior = 0
                if evidence == 0:
                    posterior = 0
                else:
                    posterior = (likelihood * prior_probability) / (evidence)
                probabilities_per_class[target_class] = posterior
                
            # select the target class varaible with the highest probability to return for the instance
            result = max(probabilities_per_class, key = lambda cls: probabilities_per_class[cls])
            final_predictions.append(result)
        return final_predictions