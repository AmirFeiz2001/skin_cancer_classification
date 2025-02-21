import argparse
from src.preprocess import load_and_merge_data, clean_data, load_images, prepare_data
from src.train import train_model
from src.utils import visualize_image, predict_new_data
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Classification with CNN and Dense Model")
    parser.add_argument('--labels_file', type=str, required=True, help='Path to ISIC_2019_Training_GroundTruth.csv')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to ISIC_2019_Training_Metadata.csv')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing ISIC_2019_Training_Input images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data for validation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs per trial')
    parser.add_argument('--max_trials', type=int, default=5, help='Number of tuner trials')
    args = parser.parse_args()

    # Load and preprocess data
    data = load_and_merge_data(args.labels_file, args.metadata_file)
    if data is None:
        return
    features, target = clean_data(data)
    images = load_images(args.image_dir)
    result = prepare_data(features, target, images, args.train_size, args.val_size, args.output_dir)
    if result is None:
        return
    train_images, val_images, test_images, x_train, x_val, x_test, y_train, y_val, y_test, scaler = result

    # Visualize sample data
    visualize_image(train_images[2], y_train[2], "Train Sample", args.output_dir + "/plots")
    visualize_image(test_images[0], y_test[0], "Test Sample", args.output_dir + "/plots")

    # Train model
    num_features = x_train.shape[1]
    best_model = train_model(train_images, x_train, y_train, val_images, x_val, y_val, num_features,
                             args.output_dir, args.max_trials, args.epochs)

    # Evaluate
    loss, accuracy = best_model.evaluate([test_images, x_test], y_test, verbose=0)
    logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Example prediction (optional, requires a sample image and metadata)
    # with open(os.path.join(args.output_dir, 'scaler.pkl'), 'rb') as f:
    #     scaler = pickle.load(f)
    # predict_new_data(best_model, "path/to/new_image.jpg", [30.0, 0, 1], scaler, args.output_dir)

if __name__ == "__main__":
    main()
