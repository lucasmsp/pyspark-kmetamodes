# Ensemble-based incremental distributed k-modes/k-metamodes clustering for PySpark

## Background

Ensemble-based incremental distributed k-modes clustering for PySpark (Python 3), similar to the algorithm proposed by Visalakshi and Arunprabha in "Ensemble based Distributed K-Modes Clustering" (IJERD, March 2015) to perform K-modes clustering in an ensemble-based way. In short, k-modes will be performed for each partition in order to identify a set of *modes* (of clusters) for each partition. Next, k-modes will be repeated to identify modes of a set of all modes from all partitions. These modes of modes are called *metamodes* here.

This package was originally based on the work of [`Marissa Saunders`](https://github.com/ThinkBigAnalytics/pyspark-distributed-kmodes). Then, it was refactored by [`Andrey Sapegin`](https://github.com/asapegin/pyspark-kmetamodes), to fixing some major issues and adding new distance functions. However, I decided to contribute to this project mainly by providing a Spark ML's interface, now one can use the algorithm as a Spark native algorithm, for example, using it inside a Pipeline ML. Other contributions are, a seed parameter was included to generate initial clusters and some performance improvements.

## Quickstart

This module has been developed and tested on Spark 2.4 and Python 3. 

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from kmetamodes import IncrementalPartitionedKMetaModes

...

df = spark.createDataFrame(data)

featuresCol = 'features'
max_dist_iter = 100
local_kmodes_iter = 10
n_modes = 3
predictionCol = 'cluster'

vectorizer = VectorAssembler(inputCols=['col1', 'col2', 'col3', 'col4'], outputCol=featuresCol)

kmodes = IncrementalPartitionedKMetaModes(n_clusters = n_modes, max_dist_iter = max_dist_iter, local_kmodes_iter = local_kmodes_iter, similarity = "frequency", metamodessimilarity = "meta", seed=None, featuresCol=features, predictionCol=predictionCol)

pipeline = Pipeline(stages=[vectorizer, kmodes])

model = pipeline.fit(df)  
output = model.transform(df)
clusters = model.clusterCenters()  # a python list of metamodes
```

A sample is also available in [example](./example) folder.

## Distance functions for k-modes: 

This module, as the original Andrey Sapegin [repository](https://github.com/asapegin/pyspark-kmetamodes), uses several different distance functions for k-modes:

1. Hamming distance (*"hamming"*). It can be used as similarity or metamodes similarity functions;
2. Frequency-based dissimilarity (*"frequency"*) proposed by He Z., Deng S., Xu X. in Improving K-Modes Algorithm Considering Frequencies of Attribute Values in Mode.  It can be used as similarity or metamodes similarity functions;
3. Andrey Sapegin's dissimilarity function (*"meta"*). This distance function keeps track of and takes into account all frequencies of all unique values of all attributes in the cluster, and NOT only most frequent values that became the attributes of the mode/metamode. This work is described at original respository. This function can only be used if similarity is *"frequency"*.

