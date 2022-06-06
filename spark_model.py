# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 08:34:50 2022

@author: miolmos
"""
import pandas as pd
from pyspark import SparkContext
#Lunch UI
# Create a Spark Session
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext 
import pyspark.ml.feature
# Load Our Transformer & Extractor Packages
from pyspark.ml.feature import Tokenizer, StopWordsRemover 
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit, col, explode, array
from sklearn.metrics import confusion_matrix, classification_report


spark = SparkSession.builder.appName("TextClassificationwithPySpark")\
    .getOrCreate()
    
spark_clase_pos = spark.read.csv('proyectos_clase_positiva.csv',
                                 header = True, inferSchema = True,
                                 sep = ';')

print("datos del dataset spark_clase_pos")
spark_clase_pos.show(5)

### Observing general information of a dataframe
print("Hay 294 valores nulos en la columna codBPIN")
print("31% de los datos de esa columna son nulos")
print(spark_clase_pos.toPandas().info())
### I observe that there are 244 null values in codBPIN column
### 244/785 = 0.31 There is 31% of null values in codBPIN column





spark_clase_neg = spark.read.csv('proyectos_clase_negativa.csv',
                                 header = True, inferSchema = True,
                                 sep = ';')

print("datos del dataset spark_clase_neg")
spark_clase_neg.show(5)

### Observing general info of a dataframe
print("Hay 2924 valores nulos en la columna codBPIN")
print("45% de los datos de esa columna son nulos")
print(spark_clase_neg.toPandas().info())
### I observe that there are 2924 null values in codBPIN column
# 2924/6472 = 0.45 There is 45% of null values in codBPIN

### Omiting column codBPIN
print("Se omite la columna codBPIN en los dos dataset porque ")
print("es un índice con una cantidad considerable de datos nulos.")
print("Por esa razón se crea una columna en cada dataset llamada label ")
print("que representará a cada una de las clases.")
## As we have a text classification problem the important column is
## 'nomProy' and codBPIN just represent an index with a big percentage of
## null values. For that reason codBPIN is removed and it is created
## a new column called 'label' to represent each of the classes

spark_clase_pos = spark_clase_pos.withColumn("label", lit(0.0))
spark_clase_pos = spark_clase_pos.select('nomProy', 'label')

spark_clase_neg = spark_clase_neg.withColumn("label", lit(1.0))
spark_clase_neg = spark_clase_neg.select('nomProy', 'label')

### Counting rows of each dataset
print("\nCantidad de filas en el dataset spark_clase_pos: {}"\
      .format(spark_clase_pos.count()))
    
print("Cantidad de filas en el dataset spark_clase_neg: {}"\
      .format(spark_clase_neg.count()))


print("\nLos datos están desbalanceados.")
print("Sin embargo se continua en la construcción del modelo ")
print("para saber como se comporta el modelo en datos no balanceados")

print("\nUnión de los datasets: spark_clase_pos y spark_clase_neg.")
print("Últimas filas\n")
spark_clase_positiva_negativa = spark_clase_pos.union(
                                spark_clase_neg)
print(spark_clase_positiva_negativa.tail(5))



print("\nFeatures de la librería ml de pyspark\n")
print(dir(pyspark.ml.feature))
    

### Feature Extraction
### Build Features From Text
# * CountVEctorizer
# * TFIDF
# * WordEmbedding
# * HashingTF
# * etc

# Stages For the Pipeline

tokenizer = Tokenizer(inputCol = 'nomProy', outputCol = 'mytokens')
stopwords_remover = StopWordsRemover(inputCol = 'mytokens',
                                     outputCol='filtered_tokens')
vectorizer = CountVectorizer(inputCol = 'filtered_tokens', 
                             outputCol='rawFeatures')
idf = IDF(inputCol='rawFeatures', outputCol='vectorizedFeatures')

### split dataset
(trainDF, testDF) = spark_clase_positiva_negativa.randomSplit((0.8, 0.2),
                                                              seed = 42)
# print(trainDF.show())
print("\nCantidad de datos en el dataset trainDF: {}".format(trainDF.count()))
print("Cantidad de datos en el dataset testDF: {}\n".format(testDF.count()))

### Estimator
print("\nEstimador")
lr = LogisticRegression(featuresCol = 'vectorizedFeatures',
                        labelCol='label')
print(lr)

### Building a Pipeline
print("\nPipeline")
pipeline = Pipeline(stages = [tokenizer, stopwords_remover, vectorizer, 
                    idf, lr])
print(pipeline)
print(pipeline.stages)

### Building Model
print("\nEntrenando el modelo en el pipeline")
lr_model = pipeline.fit(trainDF)
print(lr_model)

### Predictions on our Test Dataset
print("\nPredicciones")
predictions = lr_model.transform(testDF)
predictions.show()


### Model Evaluation
# Accuracy
# F1score
print("evaluador")
evaluator = MulticlassClassificationEvaluator(labelCol = 'label',
                                             predictionCol = 'prediction')
print(evaluator) 
### Accuracy
print("\nMétricas")
print("accuracy: {}"\
      .format(evaluator.evaluate(predictions, 
                                 {evaluator.metricName: "accuracy"})))
### f1
print("f1score: {}".format(evaluator.evaluate(predictions, 
                         {evaluator.metricName: "f1"})))

### Confusion matrix
print("\nMatriz de confusión")
y_true = predictions.select('label')
y_true = y_true.toPandas()
y_pred = predictions.select('prediction')
y_pred = y_pred.toPandas()

###
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("\nCantidad de datos por clase en el dataset testDF")
cant_class_zero = testDF.groupBy('label').count()
print(cant_class_zero.show())
print("Falsos positivos: {}".format(cm[0][1]))
print("Falsos negativos: {}".format(cm[1][0]))
label_zero = cant_class_zero\
    .filter(cant_class_zero["label"] == 0.0)\
    .select("count").collect()[0]\
    .__getitem__('count')

label_one = cant_class_zero\
    .filter(cant_class_zero["label"] == 1.0)\
    .select("count").collect()[0]\
    .__getitem__('count')

falsos_positivos = cm[0][1]/label_zero
falsos_negativos = cm[1][0]/label_one

print("Proporción de falsos positivos: {}".format(falsos_positivos))
print("Proporción de falsos positivos: {}".format(falsos_negativos))

### Oversampling
print("\nOversampling\n")
minor_df = spark_clase_positiva_negativa.filter(col("label")==0)
major_df = spark_clase_positiva_negativa.filter(col("label")==1)

ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))

a = range(ratio)

### duplicates the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a])))\
    .drop("dummy")
    
### combine both oversampled minority rows and majority rows
spark_clase_positiva_negativa = major_df.unionAll(oversampled_df)
### Count proportion of classes
spark_clase_positiva_negativa.groupBy('label').count().show()
### Applying the pipeline to the balance dataset
## Train Test split dataset
### split dataset
(trainDF, testDF) = spark_clase_positiva_negativa.randomSplit((0.8, 0.2),
                                                              seed = 42)
print("Cantidad de datos en el dataset trainDF: {}".format(trainDF.count()))
# print(trainDF.show())
print("Cantidad de datos en el dataset testDF: {}".format(testDF.count()))

### Building Model
print("\nModelo")
lr_model = pipeline.fit(trainDF)
print(lr_model)
### Predictions on our Test Dataset
print("\nPredicciones")
predictions = lr_model.transform(testDF)
predictions.show()

### Model Evaluation
# Accuracy
# F1score
print("\nEvaluador")
evaluator = MulticlassClassificationEvaluator(labelCol = 'label',
                                              predictionCol = 'prediction')
print(evaluator)
### Accuracy
print("\nMétricas")
print("accuracy: {}".format(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
### f1
print("f1score: {}".format(evaluator.evaluate(predictions, 
                         {evaluator.metricName: "f1"})))


### Confusion matrix
print("\nMatriz de confusión")
y_true = predictions.select('label')
y_true = y_true.toPandas()
y_pred = predictions.select('prediction')
y_pred = y_pred.toPandas()

###
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("\nCantidad de datos por clase en el dataset testDF")
cant_class_zero = testDF.groupBy('label').count()
print(cant_class_zero.show())
print("Falsos positivos: {}".format(cm[0][1]))
print("Falsos negativos: {}".format(cm[1][0]))
label_zero = cant_class_zero\
    .filter(cant_class_zero["label"] == 0.0)\
    .select("count").collect()[0]\
    .__getitem__('count')

label_one = cant_class_zero\
    .filter(cant_class_zero["label"] == 1.0)\
    .select("count").collect()[0]\
    .__getitem__('count')

falsos_positivos = cm[0][1]/label_zero
falsos_negativos = cm[1][0]/label_one

print("Proporción de falsos positivos: {}".format(falsos_positivos))
print("Proporción de falsos positivos: {}".format(falsos_negativos))

