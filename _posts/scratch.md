gh-repo: arewelearningyet/KNNbuild
gh-badge:
  - star
  - fork
  - follow
tags:
  - data
comments: true
published: true
date: '2020-07-30'
image: https://raw.githubusercontent.com/CSLSDS/KNNbuild/cs/assets/220px-KnnClassification.svg.png?raw=true

##      :octocat:  A homespun K-Nearest-Neighbors Classifier  :octocat:  

###            exploring machine learning algorithms
___
# The Closest Pair of Points Problem 
![nope](https://en.cursor.style/resources/cursors/thumb/5e712b819e959.png) | :point_up:                            :point_down:
:--|:--
![yep](https://en.cursor.style/resources/pointers/thumb/5e712b819e963.png)| :point_right::point_left:

>"The closest pair problem for points in the Euclidean plane[[1](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem#cite_note-sh-1)] was among the first geometric problems that were treated at the origins of the systematic study of the [computational complexity](https://en.wikipedia.org/wiki/Analysis_of_algorithms) of geometric algorithms. "
>
>A naive algorithm of finding distances between all pairs of points in a space of dimension d and selecting the minimum requires O(n2) time. It turns out that the problem may be solved in O(n log n) time in a [Euclidean space](https://en.wikipedia.org/wiki/Euclidean_space) or [Lp space](https://en.wikipedia.org/wiki/Lp_space) of fixed dimension d.[[2](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem#cite_note-2)] In the [algebraic decision tree](https://en.wikipedia.org/wiki/Algebraic_decision_tree) [model of computation](https://en.wikipedia.org/wiki/Model_of_computation), the O(n log n) algorithm is optimal, by a reduction from the [element uniqueness problem](https://en.wikipedia.org/wiki/Element_uniqueness_problem). In the computational model that assumes that the floor function is computable in constant time the problem can be solved in O(n log log n) time.[[3](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem#cite_note-fh-3)] If we allow randomization to be used together with the [floor function](https://en.wikipedia.org/wiki/Floor_function), the problem can be solved in O(n) time.[[4](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem#cite_note-km-4)][[5](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem#cite_note-rl-5)]
>
[https://en.wikipedia.org/wiki/Closest_pair_of_points_problem]

___

##  k-nearest neighbors algorithm  

>The naive version of the algorithm is easy to implement by computing the distances from the test example to all stored examples, but it is computationally intensive for large training sets. Using an approximate nearest neighbor search algorithm makes k-NN computationally tractable even for large data sets. Many nearest neighbor search algorithms have been proposed over the years; these generally seek to reduce the number of distance evaluations actually performed.   
...  
>Feature extraction
>
>When the input data to an algorithm is too large to be processed and it is suspected to be redundant (e.g. the same measurement in both feet and meters) then the input data will be transformed into a reduced representation set of features (also named features vector). Transforming the input data into the set of features is called feature extraction. If the features extracted are carefully chosen it is expected that the features set will extract the relevant information from the input data in order to perform the desired task using this reduced representation instead of the full size input. Feature extraction is performed on raw data prior to applying k-NN algorithm on the transformed data in feature space.
>
>An example of a typical computer vision computation pipeline for face recognition using k-NN including feature extraction and dimension reduction pre-processing steps (usually implemented with OpenCV):
>
>1. Haar face detection
>1. Mean-shift tracking analysis
>1. PCA or Fisher LDA projection into feature space, followed by k-NN classification
>
[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]  

___
# Documentation for official SciKit Learn class
[https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html]
