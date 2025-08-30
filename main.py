if __name__ == "__main__":
    import yaml

    from train_model import train_model
    from preprocessing import preprocessing

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # # PROCESSING DATA
    # preprocessing(**config)

    # # MODEL TRAINING
    # train_model(**config)