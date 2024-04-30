import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier



dataset = pd.read_csv('Churn_Modelling.csv')

dataset['Gender'] = (dataset['Gender'] == "Male").astype(int)

dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)





# Mapping the country to the minimum salary
country_salaries = {'France': 1540 * 12, 'Spain': 1050 * 12, 'Germany': 1580 * 12}

# Remove rows with 'EstimatedSalary' less than 1000
dataset = dataset[dataset['EstimatedSalary'] >= 1000]

# Multiply 'EstimatedSalary' by 12 if it's less than the corresponding country's salary
dataset.loc[dataset['EstimatedSalary'] < dataset['Geography'].map(country_salaries), 'EstimatedSalary'] *= 12

# Mapping countries to integers
countries = {'France': 0, 'Spain': 1, 'Germany': 2}
dataset['Geography'] = dataset['Geography'].map(countries)

plt.scatter(dataset['Balance'], dataset['EstimatedSalary'])
plt.xlabel('Balance')
plt.ylabel('EstimatedSalary')
plt.title('EstimatedSalary by Balance')
#plt.show()





newFile = open('cleaned_dataset.csv', 'w')
dataset.to_csv(newFile, index=False)

dataset = pd.read_csv('cleaned_dataset.csv')





# SMOTE

smote = SMOTE(random_state=5)
X = dataset.drop('Exited', axis=1)
y = dataset['Exited']

X, y = smote.fit_resample(X, y)
dataset = pd.concat([X, y], axis=1)

dataset.describe()






# We can extract the data in this format from pandas like this:
all_inputs = dataset[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values

# Similarly, we can extract the class labels
all_labels = dataset['Exited'].values

# Make sure that you don't mix up the order of the entries
# all_inputs[5] inputs should correspond to the class in all_labels[5]

# Here's what a subset of our inputs looks like:
all_inputs[:10]




from sklearn.model_selection import train_test_split

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.25, random_state=1)


###### DECISION TREE CLASSIFIER ######

print("====================================")
print("Decision Tree Classifier")
print("====================================")

# Create the classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training set
decision_tree_classifier.fit(training_inputs, training_classes)

# Validate the classifier on the testing set using classification accuracy
decision_tree_classifier.score(testing_inputs, testing_classes)

# Print performance metrics: (performance during learning, confusion matrix, precision, recall, accuracy, F1 measure)

predictions = decision_tree_classifier.predict(testing_inputs)

print(confusion_matrix(testing_classes, predictions))
print(classification_report(testing_classes, predictions))




###### K-NEAREST NEIGHBORS ######

from sklearn.neighbors import KNeighborsClassifier

print("====================================")
print("K-Nearest Neighbors Classifier")
print("====================================")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training_inputs, training_classes)

predictions = knn.predict(testing_inputs)

print(confusion_matrix(testing_classes, predictions))
print(classification_report(testing_classes, predictions))

# We can see that the Decision Tree Classifier has a better performance than the K-Nearest Neighbors Classifier
# So we can use the Decision Tree Classifier to predict the churn rate of the bank customers




###### NAIVE BAYES ######

from sklearn.naive_bayes import GaussianNB

print("====================================")
print("Naive Bayes Classifier")
print("====================================")

gnb = GaussianNB()
gnb.fit(training_inputs, training_classes)

predictions = gnb.predict(testing_inputs)

print(confusion_matrix(testing_classes, predictions))
print(classification_report(testing_classes, predictions))



###### Support Vector Machines ######

from sklearn.svm import SVC

print("====================================")
print("Support Vector Machines Classifier")
print("====================================")

svc = SVC()
svc.fit(training_inputs, training_classes)

predictions = svc.predict(testing_inputs)

print(confusion_matrix(testing_classes, predictions))
print(classification_report(testing_classes, predictions))


######  Neural Networks ######

from sklearn.neural_network import MLPClassifier

print("====================================")
print("Neural Networks Classifier")
print("====================================")

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(training_inputs, training_classes)

predictions = mlp.predict(testing_inputs)

print(confusion_matrix(testing_classes, predictions))
print(classification_report(testing_classes, predictions))