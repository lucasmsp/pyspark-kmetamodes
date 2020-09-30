import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from kmetamodes import IncrementalPartitionedKMetaModes

from sklearn.metrics import silhouette_score


spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("file:///tmp/soybean.csv", header=False, inferSchema=True)

df.show(5)

n_clusters = 4
max_dist_iter = 1500
local_kmodes_iter = 10
seed = 42
input_cols = df.columns[0:-1]
featuresCol = 'Features'
predictionCol = 'Cluster'

vectorizer = VectorAssembler(inputCols=input_cols, outputCol=featuresCol)

kmodes = IncrementalPartitionedKMetaModes(n_clusters = n_clusters, max_dist_iter = max_dist_iter, 
    local_kmodes_iter = local_kmodes_iter, similarity = "hamming", metamodessimilarity = "hamming",
    seed=seed, featuresCol=featuresCol, predictionCol=predictionCol)

# or by:
# kmodes = IncrementalPartitionedKMetaModes()
# kmodes.setK(n_clusters)
# kmodes.setMetamodesSimilarity('hamming')
# kmodes.setSimilarity('hamming')
# kmodes.setLocalKmodesIter(local_kmodes_iter)
# kmodes.setMaxDistIter(max_dist_iter)
# kmodes.setSeed(seed)
# kmodes.setFeaturesCol(featuresCol)
# kmodes.setPredictionCol(predictionCol)

# Using Pipeline
pipeline = Pipeline(stages=[vectorizer, kmodes])
pipeline_model = pipeline.fit(df)  
output = pipeline_model.transform(df)
kmodes_model = pipeline_model.stages[-1]
clusters = kmodes_model.clusterCenters() 

### Or using the convencional way
# df = vectorizer.transform(df)
# model = kmodes.fit(df)
# clusters = model.clusterCenters()
# output  = model.transform(df)

output.show()

print("Clusters:")
print(clusters)

df_tmp = output.toPandas()

prediction = df_tmp[predictionCol].values.tolist()
X = df_tmp[input_cols].values

def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    s = 0
    for t1, t2 in zip(a, b):
        if t1 != t2:
            s += 1
    return s

print("Silhouette Score: ", silhouette_score(X, prediction, metric=matching_dissim))
