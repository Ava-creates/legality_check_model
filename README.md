# legality_check_model
This is the tiramisu legality check model. Setting it up is similar to the cost model. However, the utility files have been updated and will work differently. 

pikle.py generates a dataset from the actual data. 

generate_dataset.py generates data for the benchmark.

balance.py helps balance the legal and illegal schedules in the dataset to get better accuracy in the f1 score.

train_model.py for training and evaluate_model.py for evaluating. 

You can set the configuration in the config.yaml
