import pandas as pd
import tensorflow as tf
from IPython.display import clear_output

# Loading Train Data
dftrain = pd.read_csv("taxi_fare/train.csv")
y_train = dftrain.pop("surge_applied")

# Loading Test Data
dfeval = pd.read_csv("taxi_fare/test.csv")
y_eval = dfeval.pop("surge_applied")

# Dividing columns to create feature columns
CATEGORICAL_COLUMNS = []
NUMERICAL_COLUMNS = ["trip_duration", "distance_traveled",
                     "num_of_passengers", "fare", "tip", "miscellaneous_fees", "total_fare"]
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))

for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))


# Create input func


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, num_epochs=1)

linear_est = tf.estimator.LinearClassifier(
    feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(result["accuracy"])

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[4])
print(y_eval.loc[4])
print(result[4]["probabilities"][1])
