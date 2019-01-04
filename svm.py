import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.neighbors as kNN
import sklearn.svm as svm
import statsmodels.api as sm
from scipy import stats

data = pandas.read_csv('/Users/snehamitta/Desktop/ML/Assignment5/WineQuality.csv', delimiter = ',')

# To create boxplots
data.boxplot(column='fixed_acidity', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("fixed_acidity")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='volatile_acidity', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("volatile_acidity")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='citric_acid', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("citric_acid")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='residual_sugar', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("residual_sugar")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='chlorides', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("chlorides")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='free_sulfur_dioxide', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("free_sulfur_dioxide")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='total_sulfur_dioxide', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("total_sulfur_dioxide")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='density', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("density")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='pH', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("pH")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='sulphates', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("sulphates")
plt.ylabel("quality_grp")
plt.show()

data.boxplot(column='alcohol', by='quality_grp', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("alcohol")
plt.ylabel("quality_grp")
plt.show()

col = data.columns

dict1 = {}
for i in range(11):
     (a,b)=stats.ttest_ind(data[data["quality_grp"]==1][col[i]],
     data[data["quality_grp"]==0][col[i]])
     dict1.update({i: (a,b)})

sortedDict = sorted(dict1.items(),key = lambda x:x[1][1],reverse = True)
for j in sortedDict:
    print("The input attribute is {}, the t statistics is {}, the two-sided p-values is {}".format(col[j[0]], j[1][0], j[1][1]))   

trainData = pandas.DataFrame(data.iloc[:,0:11])
yTrain = pandas.DataFrame(data.iloc[:,13]).astype('category')

svm_Model = svm.LinearSVC(verbose = 1, random_state = 20181111, max_iter = 10000)
# thisFit = svm_Model.fit(trainData, yTrain)

trainData.drop([col[5]], axis=1, inplace = True)
# thisFit = svm_Model.fit(trainData, yTrain)

trainData.drop([col[8]], axis=1, inplace = True)
# thisFit = svm_Model.fit(trainData, yTrain)

trainData.drop([col[9]], axis=1, inplace = True)
# thisFit = svm_Model.fit(trainData, yTrain)

trainData.drop([col[0]], axis=1, inplace = True)
# thisFit = svm_Model.fit(trainData, yTrain)

trainData.drop([col[6]], axis=1, inplace = True)
# thisFit = svm_Model.fit(trainData, yTrain)

trainData.drop([col[2]], axis=1, inplace = True)
# thisFit = svm_Model.fit(trainData, yTrain)

trainData.drop([col[3]], axis=1, inplace = True)
thisFit = svm_Model.fit(trainData, yTrain)

print('Intercept:\n', thisFit.intercept_)
print('Weight Coefficients:\n', thisFit.coef_)

y_predictClass = thisFit.predict(trainData)
print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

des = trainData.loc[0:3].describe()
mean = {'volatile_acidity':[des.loc['mean']['volatile_acidity']],
        'chlorides':[des.loc['mean']['chlorides']],
        'density':[des.loc['mean']['density']],
        'alcohol':[des.loc['mean']['alcohol']]}
mean = pandas.DataFrame(mean)
print(mean)
y_predictClass = thisFit.predict(mean)

_25per = {'volatile_acidity':[des.loc['25%']['volatile_acidity']],
        'chlorides':[des.loc['25%']['chlorides']],
        'density':[des.loc['25%']['density']],
        'alcohol':[des.loc['25%']['alcohol']]}
_25per = pandas.DataFrame(_25per)
print(_25per)
y_predictClass = thisFit.predict(_25per)

_75per = {'volatile_acidity':[des.loc['75%']['volatile_acidity']],
        'chlorides':[des.loc['75%']['chlorides']],
        'density':[des.loc['75%']['density']],
        'alcohol':[des.loc['75%']['alcohol']]}
_75per = pandas.DataFrame(_75per)
print(_75per)
y_predictClass = thisFit.predict(_75per)
