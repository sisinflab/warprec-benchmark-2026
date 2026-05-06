from elliot.run import run_experiment

run_experiment(f"config_files/lightgcn_netflix100m.yml", model_name = "LightGCN", dataset_name = "NetflixPrize100M")