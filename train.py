import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from config import Config
from Load_Dataset import load_train_test_data
from Models.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default="DataSet1_Q",
                        help="Path to the training data pickle file")
    parser.add_argument('--test_data', type=str, default="DataSet2_Q",
                        help="Path to the testing data pickle file")
    parser.add_argument('--learning_rate', type=float,
                        default=Config.model_params["learning_rate"], help="Learning rate")
    parser.add_argument(
        '--epochs', type=int, default=Config.train_params["epochs"], help="Number of epochs")
    parser.add_argument('--batch_size', type=int,
                        default=Config.train_params["batch_size"], help="Batch size")
    parser.add_argument('--validation_split', type=float,
                        default=Config.train_params["validation_split"], help="Validation split ratio")
    parser.add_argument('--kernel_regularizer', type=float,
                        default=Config.model_params["kernel_regularizer"], help="Kernel regularizer value")
    parser.add_argument('--loss_function', type=str,
                        default=Config.model_params["loss_function"], help="Loss function")
    parser.add_argument('--hidden_units', nargs='+', type=int,
                        default=Config.model_params["hidden_units"], help="Number of units in each hidden layer")
    parser.add_argument('--weights', type=str, default=None,
                        help="Path to the model weights file (hdf5 format)")
    parser.add_argument('--model_load_dir', type=str, default=None,
                        help="Path to directory to load the model file (hdf5 format)")
    parser.add_argument('--model_save_dir', type=str, default=None,
                        help="Path to the directory to save the model file (hdf5 format)")
    parser.add_argument('--save_image_dir', type=str, default=None,
                        help="Path to the directory to save images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_train, y_train, X_test, y_test = load_train_test_data(
        args.train_data, args.test_data)

    model_params = {
        "hidden_units": args.hidden_units,
        "kernel_regularizer": args.kernel_regularizer,
        "learning_rate": args.learning_rate,
        "loss_function": args.loss_function
    }

    model = build_model(input_shape=(X_train.shape[1],), **model_params)

    if args.weights:
        model.load_weights(args.weights)
        print('Weights loaded')
    if args.model_load_dir:
        model = tf.keras.models.load_model(args.model_load_dir)
        print('Model Loaded')

    callbacks = [eval(callback_str) for callback_str in Config.callbacks]

    model_hist = model.fit(X_train, y_train, epochs=args.epochs, validation_split=args.validation_split,
                           callbacks=callbacks, batch_size=args.batch_size, verbose=1)

    if args.model_save_dir:
        model.save(args.model_save_dir)
    else:
        model.save('Models/my_model.h5')

    if args.save_image_dir:
        test_predictions = model.predict(X_test).flatten()
        error = test_predictions - y_test
        plt.hist(error, bins=25, rwidth=0.8)
        plt.xlabel('Prediction Error [Alpha]')
        plt.ylabel('Count')
        error_histogram_path = f'{args.save_image_dir}/error_histogram.png'
        plt.savefig(error_histogram_path)
        print(f"Histogram plot saved to {error_histogram_path}")

        loss_vs_epoch_path = f'{args.save_image_dir}/loss_vs_epoch.png'
        plt.figure()
        plt.plot(model_hist.history['loss'], label='Train Loss')
        plt.plot(model_hist.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(loss_vs_epoch_path)
        print(f"Loss vs. Epoch plot saved to {loss_vs_epoch_path}")
    else:
        error_histogram_path = f'Output/error_histogram.png'
        test_predictions = model.predict(X_test).flatten()
        error = test_predictions - y_test
        plt.hist(error, bins=25, rwidth=0.8)
        plt.xlabel('Prediction Error [Alpha]')
        plt.ylabel('Count')
        plt.savefig(error_histogram_path)
        
        loss_vs_epoch_path = f'Output/loss_vs_epoch.png'
        plt.figure()
        plt.plot(model_hist.history['loss'], label='Train Loss')
        plt.plot(model_hist.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(loss_vs_epoch_path)
        print(f"Loss vs. Epoch plot saved to {loss_vs_epoch_path}")
