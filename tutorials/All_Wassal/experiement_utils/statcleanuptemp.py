import os
import pandas as pd
import numpy as np

import os
import pandas as pd
import numpy as np

def process_and_update_stats_recursive(parent_folder):
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file == 'simplex_data.csv':
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)

                total_elements = len(data)
                non_zero_data = data[data['Simplex Value'] != 0]
                total_non_zero = len(non_zero_data)
                percent_non_zero = (total_non_zero / total_elements) * 100

                # Save all non-zero data
                non_zero_data.to_csv(os.path.join(root, 'simplex_data_non_zero.csv'), index=False)

                total_zero = total_elements - total_non_zero
                percent_zero = (total_zero / total_elements) * 100

                avg_non_zero = non_zero_data['Simplex Value'].mean()
                sd_non_zero = non_zero_data['Simplex Value'].std()

                correct_non_zero = non_zero_data[non_zero_data['Real Class'] == non_zero_data['Hypothesized Class']]
                wrong_non_zero = non_zero_data[non_zero_data['Real Class'] != non_zero_data['Hypothesized Class']]
                total_correct_non_zero = len(correct_non_zero)
                total_wrong_non_zero = len(wrong_non_zero)

                # Write statistics to file
                with open(os.path.join(root, 'simplex_data_stat.txt'), 'w') as f:
                    f.write(f"Total number of elements: {total_elements}\n")
                    f.write(f"Total elements with non-zero simplex value: {total_non_zero}\n")
                    f.write(f"Percentage of elements with non-zero simplex value against total elements: {percent_non_zero}%\n")
                    f.write(f"Total elements with simplex value 0: {total_zero}\n")
                    f.write(f"Percentage of elements with zero simplex value against total elements: {percent_zero}%\n")
                    f.write(f"Average of the non-zero simplex value elements: {avg_non_zero}\n")
                    f.write(f"SD of the non-zero simplex value elements: {sd_non_zero}\n")
                    f.write(f"Total Correct hypothesized labels (non-zero): {total_correct_non_zero}\n")
                    f.write(f"Total Wrong hypothesized labels (non-zero): {total_wrong_non_zero}\n")

                # Delete the original simplex_data.csv file
                os.remove(file_path)

# Example usage
# process_and_update_stats_recursive('/path/to/your/parent/folder')


# Call the function
process_and_update_stats_recursive('/home/wassal/trust-wassal/tutorials/results/softloss0.3/pneumoniamnist/classimb/rounds10')
