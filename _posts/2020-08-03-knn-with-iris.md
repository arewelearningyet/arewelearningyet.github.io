---
layout: post
title: "DIY" K Nearest Neighbors Classification of iris species
subtitle: in which we apply our aglorithmic savvy to a homespun KNN class
---


# "DIY" K Nearest Neighbors Classification of iris species
## ...in which we apply our aglorithmic savvy to a homespun KNN class to guess what kind of flower we're looking at!

 [This exploration is based on the classic Iris flower data set outlined by Ronald Fisher in his 1936 paper ["The use of multiple measurements in taxonomic problems"](http://digital.library.adelaide.edu.au/coll/special//fisher/138.pdf) as an example of [linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis).]

___
# The Closest Pair of Points Problem 

![nope](https://en.cursor.style/resources/cursors/thumb/5e712b819e959.png) | 👆                            👇
:--|:--
![yep](https://en.cursor.style/resources/pointers/thumb/5e712b819e963.png)| 👉👈  


<br>
>"The closest pair problem for points in the Euclidean plane[[1](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem#cite_note-sh-1)] was among the first geometric problems that were treated at the origins of the systematic study of the [computational complexity](https://en.wikipedia.org/wiki/Analysis_of_algorithms) of geometric algorithms. "  
>
>[https://en.wikipedia.org/wiki/Closest_pair_of_points_problem](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem)

The essence of the algorithm is to determine the closest pair of points in Euclidean space, or the "nearest neighbor" to some observation (an individual iris that we have measured, but have not yet identified). 

When we are able to identify the iris species of the unknown flowers' nearest neigbors, we are able to make an educated guess about the species (or "class") that the unknown iris belongs to, with some degree of accuracy, depending on the robustness of our algorithm, and the relative "noise" of the dataset we are working with.


With respect to the iris dataset, we are looking at three types of equally-represented species, setosa, versicolor, and virginica. 

This dataset is so well-known that it comes prepackaged with some machine learning and visualization libraries, such as scikit-learn, and summary statistics are readily available even before we begin any analysis, allowing us to clearly identify the most "predictive" features, (highest 'Class Correlation') being petal width and petal length...  
![yep](https://raw.githubusercontent.com/CSLSDS/KNNbuild/cs/assets/iris_summstats.png)

When we plot petal width, petal length, and the less misleading sepal measurement, length (sepal width has negative class correlation), we can see that setosa is very easily differentiated from the others... but that versicolor and virginica are more widely distributed, as well as more closely intertwined, feature-wise:
<iframe width="1200" height="800" frameborder="0" scrolling="no" src="//plotly.com/~cslsds/1.embed"></iframe>


# HOW-TO implement k-nearest neighbors classification

Have no fear! We can make guesses about our irises based on measurements of the relative 'distance' between the feature-points of our observations (unknown irises) by applying the k-nearest neighbors algorithm to the problem.

It will work easily with our entirely-numeric dataset, as the calculations will be very straightforward... 

Since we have three classes of irises, we will look at the 3 nearest neigbors, making our choice of "K" 3. These three nearest neighbors will then essentially cast their "vote" for which class it is likely to belong to, and tallying the neighbors' votes will give us our best guess as to which class, or perhaps we could say 'neighborhood', the unknown iris is likely to belong to. 

## Preparation...  
We can implement this with a few of the basic preparations that are standard for predictive modeling, mainly arranging our observations in rows, and our features or measurements into columns. 

This "dataframe" will then be divided into two sets, the larger of which we will use to 'train' our model to know what these flower species are like, and the other will be reserved to 'test' that training on unknown irises and see how well it performs! 

We can then check our homegrown algorithm against the trusted standard implementation offered in scikit-learn's libraries!

Fortunately, it will be pretty simple to work with this dataset seeing as how it is included in so many libraries. 
## Let's import it and get ready for analysis!

```python
from sklearn import datasets

# load a common datascience toy dataset, containing iris species observations
iris = datasets.load_iris()

# arrange data into attributes/feature matrix and a target/y matrix
x = iris.data

# this is the 1-dimensional array that contains the classes
#   that we are trying to predict
y = iris.target

# if you like, print the iris dataset information to add context
print(iris['DESCR'])
print()

# you could also explore the pandas library as a way of examining the dataframe
#import pandas as pd
#irisdf = pd.DataFrame(x)
#irisdf['class'] = y
irisdf = datasets.load_iris(as_frame=True)
irisdf = irisdf.frame
#print(f'"Shape"/dimensions of the dataframe: {irisdf.shape}')
print()
print(irisdf)
print()

# arrange x, y matrices into training (80% of data) & testing (20% remaining)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

## Next! 
### We'll import our DIY KNN algorithm to give our best shot at performing this task ourselves! [My implementation can be found [here](https://github.com/CSLSDS/KNNbuild/blob/master/knn.py). You should be running these py files in the same directory to make importing your homemade class as simple as possible.]  

```python
from knn import KNN

# instantiate knn model with 3 classes for our 3 iris species
knn = KNN(3)

# 'fit' the model to our 'known' training subsets
knn.fit(X_train, y_train)

# predict class for 'unseen' test subset matrix of features without classes
# while storing it for later comparison
our_prediction = knn.predict(X_test)
```

## Next! 
### We'll import and follow the same process, but with sklearn's implementation of the k-nearest neighbors classification model.

```python
from sklearn.neighbors import KNeighborsClassifier

# our step 1:
# knn = KNN(3)
sklearn = KNeighborsClassifier(3)

# our step 2:
# knn.fit(X_train, y_train)
sklearn.fit(X_train, y_train)

# our step 3:
# our_prediction = knn.predict(X_test)
sklearn_prediction = sklearn.predict(X_test)
```

## And now...! 
### ...for our final triumph, we can check and compare our accuracy!

```python
from sklearn.metrics import accuracy_score

print(f"accuracy of homebrewed knn: {accuracy_score(y_test, our_prediction)}\n")

print(f"accuracy of sklearn library results: {accuracy_score(y_test, sklearn_prediction)}\n")
```
![HOORAY!](https://raw.githubusercontent.com/CSLSDS/KNNbuild/master/assets/comparison.png)  
# Hey! 
### That is not too bad, even if this is a very simple classification problem.

#### While it may not perform as fast and as effectively as scikit-learn's implementation, it is very likely much more intelligible if you want to take a look "under the hood" and follow how this algorithm is really working.
  
<br>
  
### [See the repository README for some more context for this algorithm.](https://github.com/CSLSDS/KNNbuild)  

<br>  
  
### See below for the full text of our constructed KNN class.   
  
  
    
    
```python
import pandas as pd
import numpy as np
import statistics as stat

class KNN:
    X_train = None
    y_train = None
    def __init__(self, K):
        # K Nearest Neighbors requires a value for K,
        #   based on # of classes, or # of 'votes' later
        self.K = K

    def fit(self, X_train, y_train):
        # fit method requires data as input,
        #   arranged in features (X_train) and target (y_train) matrices
        self.X_train = X_train
        self.y_train = y_train

    def edist(self, rowA, rowB):
        # calculates Euclidean distance between points
        return np.linalg.norm(rowA-rowB)

    def predict(self, val_or_test):
        # takes as input a validation or test feature matrix (X_val/X_test)
        # Here, euclidean distances will be calculated,
        #   classifying points based on minimum distance (nearest neigbors)

        # initialize list to return predictions
        predictions = []
        for secondary_obs in val_or_test:
        # iterate over observations in secondary test/validation dataset
            # store in initialized list to sort/compare distances/neighbors
            distances = []
            for ix, train_obs in enumerate(self.X_train):
            # iterate over index, observations in original training set
                # calculate euclidean distance between train obs and test obs
                distance = self.edist(train_obs, secondary_obs)
                # [distance, index] appended to distances for comparison
                distances.append([distance, ix]) # neighbor entries
            # sort the neighbors by which are nearest, descending
            ranked_neighbors = sorted(distances)
            # subset ranked_neighbors to length/number defined in K
            k_nearest_neighbors = ranked_neighbors[:self.K]

            # assign the index matching target training observation
            #    for the top ranked neighbors (corresponding classes)
            predict = [self.y_train[ix[1]] for ix in k_nearest_neighbors]
            
            # return prediction with max count/"votes" as the 
            #   nearest neigbhor/most likely class
            result = max(predict, key = predict.count)
            
            predictions.append(result)
        
        return predictions
```
