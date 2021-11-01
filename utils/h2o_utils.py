import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init(max_mem_size="16G")

def train(data, target_variable, id_column=None, max_runtime=30):
    # Split into train and validation
    h2o_df = h2o.H2OFrame(data)
    splits = h2o_df.split_frame(ratios=[0.8], seed=1)
    train, test = splits[0], splits[1]

    # Split into x and y
    y = target_variable
    x = h2o_df.columns
    x.remove(y)
    if id_column:
        x.remove(id_column)

    # Train model
    aml = H2OAutoML(max_runtime_secs=max_runtime, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    # Save model
    save_path = "models"
    model_path = h2o.save_model(model=aml.leader, path=save_path, force=True)

    # Return metrics
    lb = aml.leaderboard.as_data_frame(use_pandas=True)
    metrics = lb.to_dict(orient="records")[0]

    return model_path, metrics


def validate(data, model_path, id_column=None):
    print(model_path)
    model = h2o.load_model(model_path)

    if id_column:
        data.drop(id_column, axis=1, inplace=True)

    h2o_df = h2o.H2OFrame(data)

    # perf = model.model_performance(h2o_df)
    # print(perf)

    predictions = model.predict(h2o_df)

    return predictions.as_data_frame(use_pandas=True)