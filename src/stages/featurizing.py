import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import joblib
import dvc.api

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger


def featurizing() -> None:
    """
    Load processed data. Apply PCA to reduce the dimensions of the dataset.
    Args:
        config_path {Text}: path to config
    """
    config = dvc.api.params_show()

    logger = get_logger("FEATURIZE", log_level=config["base"]["log_level"])
    logger.info("Load processed data")
    df = pd.read_csv(config["data"]["data_preprocessing"])
    logger.info("Scale data")
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(df)
    logger.info("Data is standardized")
    featurizer_name = config["featurize"]["featurizer_name"]
    logger.info(f"Featurizer: {featurizer_name}")
    
    if featurizer_name == "pca":
        n_components = config["featurize"]["n_components"]
        logger.info(f"Number of components: {n_components}")
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_standardized)
        logger.info("Save PCA model")
        joblib.dump(pca, config["featurize"]["featurizer_path"])
        logger.info("Save processed data")
        pd.DataFrame(data_pca).to_csv(Path(config["data"]["data_featurized"]), index=False)
    
    elif featurizer_name == "autoencoder":
        seed = config["base"]["random_state"]
        tf.random.set_seed(seed)
        np.random.seed(seed)
        # Define the number of features
        input_dim = data_standardized.shape[1]
        encoding_dim = config["featurize"]["parameters"]["encoding_dim"]
        # Define the input layer
        input_layer = Input(shape=(input_dim, ))
        # Define the encoder layer
        encoder_activation = config["featurize"]["hyperparameters"]["encoder_activation"]
        encoder_layer = Dense(encoding_dim, activation=encoder_activation, activity_regularizer=regularizers.l1(10e-5))(input_layer)
        # Define the decoder layer
        decoder_activation = config["featurize"]["hyperparameters"]["decoder_activation"]
        decoder_layer = Dense(input_dim, activation=decoder_activation)(encoder_layer)
        # Define the autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
        # Compile the autoencoder model
        optimizer = config["featurize"]["hyperparameters"]["optimizer"]
        loss = config["featurize"]["hyperparameters"]["loss"]
        logger.info(f"Optimizer: {optimizer}")
        logger.info(f"Loss: {loss}")
        autoencoder.compile(optimizer=optimizer, loss=loss)
        # Define the early stopping criteria
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min', restore_best_weights=True)
        # Train the autoencoder model
        epochs = config["featurize"]["hyperparameters"]["epochs"]
        batch_size = config["featurize"]["hyperparameters"]["batch_size"]
        autoencoder.fit(data_standardized, data_standardized, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, verbose=1, callbacks=[early_stopping])
        # Save the autoencoder model
        logger.info("Save autoencoder model")
        autoencoder.save(Path(config["featurize"]["featurizer_path"]))
        # Get the encoder model
        encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=1).output)
        # Get the reduced data using the encoder
        reduced_data = encoder_model.predict(data_standardized)
        # Convert the reduced data to a DataFrame
        reduced_df = pd.DataFrame(reduced_data, columns=[f"latent_{i+1}" for i in range(encoding_dim)])
        # Save the reduced dataset to a CSV file
        logger.info("Save processed data")
        reduced_df.to_csv(Path(config["data"]["data_featurized"]), index=False)

if __name__ == "__main__":
    featurizing()