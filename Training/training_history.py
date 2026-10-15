import matplotlib.pyplot as plt

def read_training_results(filename):
    results = []
    with open(filename, 'r') as file:
        # Skip the header line
        next(file)
        # Iterate over each line in the file
        for line in file:
            # Split the line by comma and extract the first three values
            lstm_units, batch_size, weight, *_ = line.split(',')
            # Convert the values to integers and add them as a set to the results list
            results.append((int(lstm_units), int(batch_size), int(weight)))
    return results # Expected output : [{128, 32, 1}, {128, 32, 100}, {128, 32, 200}, {128, 32, 300}]


# Function to plot and save the training history
def plot_and_save_history(history, unit_lstm, batch_size, loss_weight):
    # Overall training history
    plt.figure()
    plt.plot(history.history['loss'], label='Overall loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and validation loss (LSTM_units={unit_lstm}, batch_size={batch_size}, weight={loss_weight})')
    plt.legend()
    plt.savefig(f'training_history/{unit_lstm}_{batch_size}_{loss_weight}.png')
    plt.close()

    # bbox_output training history
    plt.figure()
    plt.plot(history.history['bbox_output_loss'], label='bbox_output loss')
    plt.plot(history.history['val_bbox_output_loss'], label='Validation bbox_output loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'bbox_output training and validation loss (LSTM_units={unit_lstm}, batch_size={batch_size}, weight={loss_weight})')
    plt.legend()
    plt.savefig(f'training_history/{unit_lstm}_{batch_size}_{loss_weight}_bbox.png')
    plt.close()

    # pedestrian_classification training history
    plt.figure()
    plt.plot(history.history['pedestrian_classification_loss'], label='pedestrian_classification loss')
    plt.plot(history.history['val_pedestrian_classification_loss'], label='Validation pedestrian_classification loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'pedestrian_classification training and validation loss (LSTM_units={unit_lstm}, batch_size={batch_size}, weight={loss_weight})')
    plt.legend()
    plt.savefig(f'training_history/{unit_lstm}_{batch_size}_{loss_weight}_classification.png')
    plt.close()
