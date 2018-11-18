from __future__ import division
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split


def extract_data():
    pass


def preprocessing():
    # need to update this according to our plan #
    dt = pd.read_csv('train.csv')
    # need to update this according to our plan #
    # fill up blank age column and embarked column with mean and mode respectively #
    dt['Age'] = np.where(dt['Age'].isnull(), np.round(np.mean(dt['Age']), 0), dt['Age'])
    dt['age_bucket'] = np.where(dt['Age'] <= 10, 'children', np.where(dt['Age'] <= 30, 'Adults', np.where(dt['Age'] <= 40, 'Middle Age', 'Old')))
    dt['Embarked'] = np.where(dt['Embarked'].isnull(), stats.mode(dt['Embarked'])[0], dt['Embarked'])
    return dt

# calculate the entropy of each column #
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# calculate the information gain of each column #
def informationgain(data, target_col, attribute):
    # calcualte the entropy of target column #
    total_entropy = entropy(target_col=data[target_col])

    # calculate counts of attribute distinct values #
    elements, counts = np.unique(data[attribute], return_counts=True)

    # calculate entropy of attribute column #
    attribute_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data['Survived'][data[attribute] == elements[i]]) for i in range(len(counts))])

    # calculate the information gain #
    information_gain = total_entropy - attribute_entropy

    return information_gain


# create a decision tree #
def ID3(data, originaldata, features, parent_node_class=None, target_attribute='Survived'):
    # only one distinct value in attribute column #
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]

    elif len(features) == 0:
        return parent_node_class
    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [informationgain(data, target_attribute, feature) for feature in
                       features]  # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data, originaldata, features, parent_node_class, target_attribute)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return (tree)


# predict the target using trained decision tree #
def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return 0
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


# This is expert component which make use of every other function #
def expert(testing,features, decision_tree):
    testing_records = testing[features].to_dict(orient='records')
    predicted = pd.DataFrame(columns=['prediction'])
    for i in range(len(testing)):
        predicted.loc[i, 'Survived'] = predict(testing_records[i], decision_tree)
    accuracy = (np.sum(predicted["Survived"] == testing["Survived"])/len(testing))*100
    return accuracy


def split_data(dt):
    ## Step 01 Split the dataset ##
    split_dataset = train_test_split(dt, test_size=0.25, random_state=42)
    training = split_dataset[0].reset_index()
    testing = split_dataset[1].reset_index()
    return training, testing


def testing():
    dt = preprocessing()
    training, testing = split_data(dt)
    features = ['Sex', 'age_bucket', 'Embarked']
    decision_tree = ID3(training, training, features)
    accuracy = expert(testing, features, decision_tree)
    return accuracy