from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    # Fill missing values
    # Encode categorical variables    
    # One-hot encode indexed features
    # Assemble features into a single vector
    df = df.na.drop()

    categorical_cols = [field for (field, dtype) in df.dtypes if dtype == "string" and field != "Churn"]
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=col+"_ohe") for indexer, col in zip(indexers, categorical_cols)]

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

    numeric_cols = [field for (field, dtype) in df.dtypes if ((dtype == "double" or dtype == "int") and field != "label")]

    assembler_inputs = [col+"_ohe" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler])
    model = pipeline.fit(df)
    final_df = model.transform(df).select("label", "features")

    return final_df

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    # Split data into training and testing sets
    # Train logistic regression model
    # Predict and evaluate
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    print(f"Logistic Regression AUC: {auc:.4f}")


# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
    result = selector.fit(df).transform(df)
    print("Top 10 important features selected using Chi-Square Test.")
    return result  

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    # Split data
    # Define models
    # Define hyperparameter grids
    # Perform cross-validation for each model
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    evaluator = BinaryClassificationEvaluator()

    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBTClassifier": GBTClassifier()
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [3, 5, 10]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 50]).build(),
        "GBTClassifier": ParamGridBuilder().addGrid(models["GBTClassifier"].maxIter, [10, 20]).build()
    }

    for name in models:
        print(f"Training and tuning {name}...")
        cv = CrossValidator(estimator=models[name],
                            estimatorParamMaps=param_grids[name],
                            evaluator=evaluator,
                            numFolds=3)
        cv_model = cv.fit(train_df)
        predictions = cv_model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"{name} AUC: {auc:.4f}")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
