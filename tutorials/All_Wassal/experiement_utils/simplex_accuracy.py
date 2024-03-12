import os
import pandas as pd

def read_csv_and_calculate_accuracy(wassal_path):
    # Initialize a list to store accuracy data for each budget, AL round, and hypothesized class
    accuracy_data = []

    # Iterate through each budget folder
    for budget_folder in os.listdir(wassal_path):
        budget_path = os.path.join(wassal_path, budget_folder)
        if os.path.isdir(budget_path):
            budget = int(budget_folder)

            # Iterate through each experiment folder
            for exp_folder in os.listdir(budget_path):
                exp_path = os.path.join(budget_path, exp_folder, 'simplex_viz','al_round_', 'simplex')
                if os.path.isdir(exp_path):

                    # Iterate through each AL round folder
                    for al_round_folder in os.listdir(exp_path):
                        al_round_path = os.path.join(exp_path, al_round_folder)
                        if os.path.isdir(al_round_path):
                            al_round = int(al_round_folder)
                            csv_file = os.path.join(al_round_path, 'simplex_data.csv')

                            # Read the CSV file and calculate accuracies
                            df = pd.read_csv(csv_file)
                            actual_labels = df['Real Class'].apply(lambda x: int(x.split('(')[1].split(')')[0])).to_list()
                            df = df.drop(columns=['Real Class'])
                            df.columns = range(df.shape[1])
                            for hypothesized_class in df.columns:
                                correct_predictions = sum(actual_labels[i] == hypothesized_class for i in range(len(actual_labels)))
                                accuracy = correct_predictions / len(actual_labels)
                                accuracy_data.append([budget, al_round, hypothesized_class, accuracy])

    # Create a DataFrame from the accuracy data
    accuracy_df = pd.DataFrame(accuracy_data, columns=['Budget', 'AL Round', 'Hypothesized Class', 'Accuracy'])

    return accuracy_df

# Example usage
wassal_path = '/home/venkatapathy/trust-wassal/tutorials/results/onlywassal/cifar10/classimb/rounds10/WASSAL'
accuracy_df = read_csv_and_calculate_accuracy(wassal_path)
print(accuracy_df)
