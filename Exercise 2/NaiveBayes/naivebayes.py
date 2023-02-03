import numpy as np
import warnings
import math

class NaiveBayes:
    def __init__(self):
        self.feature_probabilities = {}
        self.target_probabilities = {}
        self.target_class_values = []
        pass
        
    # probability density function
    def normal_distr_fn(self, x, mean, std):
        return (1.0 / (math.sqrt(2 * math.pi) * std) * math.exp(-0.5 * ((x - mean) / std) ** 2))
        
    def fit(self, df, target_class_name):
        # get all feature names (without target class)
        feature_names = list(df.drop(target_class_name, axis=1).columns)

        # get all target class values
        self.target_class_values = list(df[target_class_name].unique())

        # calculate the target probability for each class
        for _class in self.target_class_values:
            # get number of occurrences and divide by number of instances
            occurence = df[df[target_class_name] == _class].shape[0]
            self.target_probabilities[_class] = occurence / df.shape[0]

        # separate the feature names into nominal and numeric
        numeric_feature_names = []
        nominal_feature_names = []
        for column in df.drop(target_class_name, axis=1).columns:
            if((df[column].dtype == np.int64) or (df[column].dtype == np.float64)):
                numeric_feature_names.append(column)
            else:
                nominal_feature_names.append(column)
                
        # calculate priori probabilities for nominal features
        for feature_name in nominal_feature_names:
            self.feature_probabilities[feature_name] = {}

            feature_values = list(df[feature_name].unique())
            for feature_value in feature_values:
                self.feature_probabilities[feature_name][feature_value] = {}

                for target_class in self.target_class_values:
                    # calculate occurrence of feature value and occurrence target class value
                    value_occurrence = df[(df[feature_name] == feature_value) & (df[target_class_name] == target_class)].shape[0]
                    class_occurrence = df[df[target_class_name] == target_class].shape[0]
                    # add +1 to for every feature value - class combination count to solve "zero-frequency problem"
                    self.feature_probabilities[feature_name][feature_value][target_class] = (value_occurrence + 1) / class_occurrence
    
        # calculate priori probabilities for numeric features
        for feature_name in numeric_feature_names:
            self.feature_probabilities[feature_name] = {}

            for target_class in self.target_class_values:
                mean = df[df[target_class_name] == target_class][feature_name].mean()
                std = df[df[target_class_name] == target_class][feature_name].std()

                self.feature_probabilities[feature_name][target_class] = { 'mean': mean, 'std': std }
    
    def predict(self, df):
        # predict target class values for instances without target class
        predictions = []
        # iterate all instances
        for row_index in range(df.shape[0]):
            instance = df.iloc[row_index, :]
            likelihood_per_class = {}
            probability_per_class = {}

            # iterate all target class values
            for target_class_value in self.target_class_values:
                # set init value target class probability
                likelihood = self.target_probabilities[target_class_value]

                # iterate all features
                for feature_name in df.columns:
                    feature_value = instance[feature_name]
                    # omit missing vales
                    if feature_value is None:
                        continue

                    # numeric features 
                    if((df[feature_name].dtype == np.int64) or (df[feature_name].dtype == np.float64)):
                        # omit missing vales from likelihood calculation
                        if target_class_value not in self.feature_probabilities[feature_name]:
                            continue
                        mean = self.feature_probabilities[feature_name][target_class_value]['mean']
                        std = self.feature_probabilities[feature_name][target_class_value]['std']
                        density_value = self.normal_distr_fn(feature_value, mean, std)
                        likelihood = likelihood * density_value
                    # nominal features    
                    else:
                        # omit missing vales from likelihood calculation
                        if feature_value not in self.feature_probabilities[feature_name]:
                            continue
                        if target_class_value not in self.feature_probabilities[feature_name][feature_value]:
                            continue
                        # calcilate likelihood
                        priori_probability = self.feature_probabilities[feature_name][feature_value][target_class_value]
                        likelihood = likelihood * priori_probability

                likelihood_per_class[target_class_value] = likelihood

            # calculate probability per class by normalizing likelihoods
            likelihood_sum = np.sum(list(likelihood_per_class.values()))
            for _class_name in likelihood_per_class:
                probability_per_class[_class_name] = likelihood_per_class[_class_name] / likelihood_sum

            # the highest probability is the target class prediction
            result = max(probability_per_class, key = lambda cls: probability_per_class[cls])
            predictions.append(result)
        
        return predictions