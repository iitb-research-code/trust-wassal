import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_and_calculate_accuracy(wassal_path):
    # Initialize a dictionary to store accuracy data for each budget, AL round, experiment, and hypothesized class
    accuracy_data = {}

    # Create the viz_images folder if it doesn't exist
    viz_images_path = os.path.join(wassal_path, 'viz_images')
    os.makedirs(viz_images_path, exist_ok=True)

    # Iterate through each budget folder
    for budget_folder in os.listdir(wassal_path):
        budget_path = os.path.join(wassal_path, budget_folder)
        if os.path.isdir(budget_path):
            budget = int(budget_folder.split('_')[-1])

            # Initialize a dictionary for this budget
            accuracy_data[budget] = {}

            # Iterate through each experiment folder
            for exp_folder in os.listdir(budget_path):
                exp_path = os.path.join(budget_path, exp_folder, 'simplex_viz')
                if os.path.isdir(exp_path) and exp_folder.startswith('exp'):
                    exp_id = exp_folder.split('_')[-1]

                    # Initialize a dictionary for this experiment
                    accuracy_data[budget][exp_id] = {}

                    # Iterate through each AL round folder
                    for al_round_folder in os.listdir(exp_path):
                        if not os.path.isdir(os.path.join(exp_path, al_round_folder)):
                            continue
                        
                        al_round_path = os.path.join(exp_path, al_round_folder, 'simplex')
                        for al_number in os.listdir(al_round_path):
                            al_round = int(al_number)

                            # Initialize a list for this AL round
                            accuracy_data[budget][exp_id][al_round] = []

                            # Read the CSV file and calculate accuracies
                            csv_file = os.path.join(al_round_path, al_number, 'simplex_data.csv')
                            df = pd.read_csv(csv_file, header=None)

                            # Calculate accuracies for each hypothesized class
                            total_classes = df.shape[1] - 1
                            for cls in range(total_classes):
                                actual_class_rows = df[df[0] == cls]
                                correct_predictions = actual_class_rows.apply(
                                    lambda row: row.iloc[1:].idxmax() == cls + 1, axis=1
                                ).sum()
                                accuracy = correct_predictions / len(actual_class_rows)
                                accuracy_data[budget][exp_id][al_round].append(accuracy)

                    # Plot the accuracies for each class across all AL rounds for this experiment
                    plt.figure(figsize=(10, 6))
                    for cls in range(len(accuracy_data[budget][exp_id][1])):
                        accuracies = [accuracy_data[budget][exp_id][al_round][cls] for al_round in sorted(accuracy_data[budget][exp_id].keys())]
                        plt.plot(sorted(accuracy_data[budget][exp_id].keys()), accuracies, label=f'Class {cls}')
                    plt.xlabel('AL Round')
                    plt.ylabel('Accuracy')
                    plt.title(f'Accuracy for Budget {budget}, Experiment {exp_id}')
                    plt.legend()
                    # Save the accuracy plot in the viz_images folder
                    accuracy_plot_filename = f'simplex_data_accuracy_budget_{budget}_exp_{exp_id}.png'
                    plt.savefig(os.path.join(viz_images_path, accuracy_plot_filename))
                    plt.close()
                    print(f'Saved accuracy plot in {os.path.join(viz_images_path, accuracy_plot_filename)}')

                    # Plot and save histogram for each class showing total number and correct predictions
                    plt.figure(figsize=(10, 6))
                    bar_width = 0.35
                    indices = np.arange(total_classes)
                    total_counts = [df[df[0] == cls].shape[0] for cls in range(total_classes)]
                    correct_counts = [df[(df[0] == cls) & (df.iloc[:, 1:].idxmax(axis=1) == cls + 1)].shape[0] for cls in range(total_classes)]

                    plt.bar(indices, total_counts, bar_width, label='Total', alpha=0.6)
                    plt.bar(indices + bar_width, correct_counts, bar_width, label='Correct', alpha=0.6)

                    plt.xlabel('Class')
                    plt.ylabel('Count')
                    plt.title('Total vs Correct Predictions for Each Class')
                    plt.xticks(indices + bar_width / 2, [f'Class {i}' for i in range(total_classes)])
                    plt.legend()

                    # Save the histogram in the viz_images folder
                    histogram_filename = f'simplex_data_histogram_budget_{budget}_exp_{exp_id}.png'
                    plt.savefig(os.path.join(viz_images_path, histogram_filename))

def extract_max_hypothesized(wassal_path):
    # Iterate through each budget folder
    for budget_folder in os.listdir(wassal_path):
        budget_path = os.path.join(wassal_path, budget_folder)
        if os.path.isdir(budget_path):

            # Iterate through each experiment folder
            for exp_folder in os.listdir(budget_path):
                exp_path = os.path.join(budget_path, exp_folder, 'simplex_viz')
                if os.path.isdir(exp_path) and exp_folder.startswith('exp'):

                    # Iterate through each AL round folder
                    for al_round_folder in os.listdir(exp_path):
                        if not os.path.isdir(os.path.join(exp_path, al_round_folder)):
                            continue
                        
                        al_round_path = os.path.join(exp_path, al_round_folder, 'simplex')
                        for al_number in os.listdir(al_round_path):
                            # Read the CSV file
                            csv_file = os.path.join(al_round_path, al_number, 'simplex_data.csv')
                            df = pd.read_csv(csv_file, header=None)

                            # Get the max value and its corresponding hypothesized label for each row
                            df['max_value'] = df.iloc[:, 1:].max(axis=1)
                            df['hypothesized_label'] = df.iloc[:, 1:].idxmax(axis=1) - 1

                            # Store the actual label, hypothesized label, and max value in a new CSV file
                            df[['max_value', 'hypothesized_label']] = df[['max_value', 'hypothesized_label']].astype(float)
                            df[['actual_label', 'hypothesized_label', 'max_value']] = df[[0, 'hypothesized_label', 'max_value']]
                            new_csv_file = os.path.join(al_round_path, al_number, 'simplex_max_hypothesized.csv')
                            df[['actual_label', 'hypothesized_label', 'max_value']].to_csv(new_csv_file, index=False)
                            print(f'Saved {new_csv_file}')
# Example usage
wassal_path = '/home/venkatapathy/trust-wassal/tutorials/results/onlywassal/cifar10/classimb/rounds10/WASSAL'
extract_max_hypothesized(wassal_path)