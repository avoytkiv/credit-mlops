import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import joblib
import dvc.api

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger

def make_model(logger, model_path, data_standardized, encoding_dim, encoder_activation, decoder_activation, optimizer, loss, epochs, batch_size, seed):
    """
    Define the autoencoder model and train it.
    """

    logger.info("Define the random seed")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    logger.info("Define the number of features")
    input_dim = data_standardized.shape[1]
    
    logger.info(f"Input dimension: {input_dim}")
    input_layer = Input(shape=(input_dim,))

    logger.info("Define the encoder layer")
    x = Dense(7, activation='relu')(input_layer)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(x)
    encoder_layer = Dense(encoding_dim, activation='relu', kernel_initializer='glorot_uniform', name='encoder')(x)

    logger.info("Define the decoder layer")
    x = Dense(2000, activation='sigmoid', kernel_initializer='glorot_uniform')(encoder_layer)
    x = Dense(500, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    decoder_layer = Dense(17, kernel_initializer='glorot_uniform')(x)

    logger.info("Define the autoencoder model")
    autoencoder = Model(inputs=input_layer, outputs=decoder_layer)

    logger.info("Compile the autoencoder model")
    autoencoder.compile(optimizer=optimizer, loss=loss)

    logger.info("Define the early stopping criteria")
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min', restore_best_weights=True)

    logger.info("Train the autoencoder model")
    autoencoder.fit(data_standardized, data_standardized, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    logger.info("Plot the encoder model")
    plot_model(autoencoder, model_path / 'autoencoder.png', show_shapes=True, show_layer_names=True)

    logger.info("Save autoencoder model")
    save_model(autoencoder, model_path)

def save_model(autoencoder, model_path):
    """
    Save the autoencoder model
    """
    # Define the model path
    model_path = Path(model_path)
    # Save the model architecture and weights
    model_json = autoencoder.to_json()
    with open(model_path / 'architecture.json', "w") as json_file:
        json_file.write(model_json)
    autoencoder.save_weights(Path(model_path) / 'weights.h5')

def load_model(logger, model_path, optimizer, loss):
    """
    Load the autoencoder model architecture and weights.
    Args:
        model: folder name that contains the model architecture and weights
    """
    logger.info("Load autoencoder model")
    json_file = open(Path(model_path) / 'architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    autoencoder.load_weights(Path(model_path) / 'weights.h5')
    logger.info("Check if the model is compiled")
    if hasattr(autoencoder, 'compiled_metrics'):
        logger.info("Compile the autoencoder model")
        autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder

def apply_model(logger, model_path, autoencoder, data_standardized, encoding_dim, data_featurized_path):
    """
    Apply the autoencoder model to the data and save the reduced dataset.
    """
    logger.info("Define the encoder model")
    encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output) # autoencoder.get_layer('encoder_layer').output

    logger.info("Plot the encoder model")
    plot_model(encoder_model, model_path / 'encoder.png', show_shapes=True, show_layer_names=True)

    logger.info("Get the reduced data using the encoder")
    reduced_data = encoder_model.predict(data_standardized)
    
    logger.info(f"Convert the reduced data to a DataFrame with {encoding_dim} columns")
    reduced_df = pd.DataFrame(reduced_data, columns=[f"latent_{i+1}" for i in range(encoding_dim)])
    
    logger.info("Save the reduced dataset to a CSV file")
    reduced_df.to_csv(Path(data_featurized_path), index=False)


def featurizing() -> None:
    """
    Load processed data. Apply PCA to reduce the dimensions of the dataset.
    """
    config = dvc.api.params_show()
    

    logger = get_logger("FEATURIZE", log_level=config["base"]["log_level"])
    logger.info("Start featurizing")
    featurizer_path = config["featurize"]["featurizer_path"]
    data_featurized_path = config["data"]["data_featurized"]
    
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
        joblib.dump(pca, featurizer_path)
        logger.info("Save processed data")
        pd.DataFrame(data_pca).to_csv(Path(data_featurized_path), index=False)
    
    elif featurizer_name == "autoencoder":
        model_path = Path(config["featurize"]["model_path"]) / config["featurize"]["model_name"]
        logger.info(f"Model path: {model_path}")
        # Create the directory if it does not exist
        model_path.mkdir(parents=True, exist_ok=True)
        seed = config["base"]["random_state"]
        encoding_dim = config["featurize"]["parameters"]["encoding_dim"]
        logger.info(f"Encoding dimension: {encoding_dim}")
        encoder_activation = config["featurize"]["hyperparameters"]["encoder_activation"]
        decoder_activation = config["featurize"]["hyperparameters"]["decoder_activation"]
        optimizer = config["featurize"]["hyperparameters"]["optimizer"]
        loss = config["featurize"]["hyperparameters"]["loss"]
        epochs = config["featurize"]["hyperparameters"]["epochs"]
        batch_size = config["featurize"]["hyperparameters"]["batch_size"]

        make_model(logger, model_path, data_standardized, encoding_dim, encoder_activation, decoder_activation, optimizer, loss, epochs, batch_size, seed)
        autoencoder = load_model(logger, model_path, optimizer, loss)
        autoencoder.save(featurizer_path)
        apply_model(logger, model_path, autoencoder, data_standardized, encoding_dim, data_featurized_path)


if __name__ == "__main__":
    featurizing()