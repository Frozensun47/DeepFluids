import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from config import Config
from Load_Dataset import load_train_test_data
from Models.model import build_model
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default="DataSet2_Q",
                        help="Path to the testing data pickle file")
    parser.add_argument('--load_model', type=str, default=None,
                        help="Path to the model file (hdf5 format) to load")
    parser.add_argument('--load_weights', type=str, default=None,
                        help="Path to the model weights file (hdf5 format) to load")
    parser.add_argument('--save_image_dir', type=str, default=None,
                        help="Directory to save the histogram plot image")
    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, np.round(y_pred), average='binary')
    return rmse, precision, recall, f1


if __name__ == "__main__":
    args = parse_args()

    _, _, X_test, y_test = load_train_test_data(args.test_data, args.test_data)

    model = build_model(input_shape=(X_test.shape[1],), **Config.model_params)

    if args.load_model:
        model = tf.keras.models.load_model(args.load_model)
    elif args.load_weights:
        model.load_weights(args.load_weights)

    test_loss = model.evaluate(X_test, y_test, verbose=2)
    print("Test loss:", test_loss)

    test_predictions = model.predict(X_test).flatten()

    rmse, precision, recall, f1 = calculate_metrics(y_test, test_predictions)
    print("RMSE:", rmse)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    error = test_predictions - y_test
    plt.hist(error, bins=25, rwidth=0.8)
    plt.xlabel('Prediction Error [Alpha]')
    plt.ylabel('Count')

    if args.save_image_dir:
        image_path = args.save_image_dir + '/histogram.png'
        plt.savefig(image_path)
        print(f"Histogram plot saved to {image_path}")

        real_vs_pred_path = args.save_image_dir + '/real_vs_pred.png'
        plt.figure()
        plt.scatter(y_test, test_predictions)
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        plt.savefig(real_vs_pred_path)
        print(f"Real vs. Predicted plot saved to {real_vs_pred_path}")

    else:
        image_path = 'Output/test_output_histogram.png'
        plt.savefig(image_path)
        print(f"Histogram plot saved to {image_path}")

    metrics_text = f"RMSE: {rmse}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}"
    if args.save_image_dir:
        metrics_path = args.save_image_dir + '/testmetrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(metrics_text)
            print(f"Metrics saved to {metrics_path}")
    else:
        metrics_path = 'Output/testmetrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(metrics_text)
            print(f"Metrics saved to {metrics_path}")
