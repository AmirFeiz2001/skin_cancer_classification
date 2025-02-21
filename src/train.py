from kerastuner.tuners import RandomSearch
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(train_images, x_train, y_train, val_images, x_val, y_val, num_features, output_dir="output", max_trials=5, epochs=5):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tuner = RandomSearch(
        lambda hp: build_model(hp, num_features),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=3,
        directory=os.path.join(output_dir, 'tuner'),
        project_name='skin_cancer'
    )

    logging.info("Starting hyperparameter search...")
    tuner.search(
        [train_images, x_train], y_train,
        batch_size=32,
        epochs=epochs,
        verbose=1,
        validation_data=([val_images, x_val], y_val)
    )

    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(os.path.join(output_dir, 'best_model.h5'))
    logging.info(f"Best model saved to {output_dir}/best_model.h5")
    return best_model
