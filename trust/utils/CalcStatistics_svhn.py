import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
#for cifar
#base_dir = "/home/wassal/trust-wassal/tutorials/results/svhn/classimb"
#budgets=['50', '100', '150', '200']

#budgets=['5']
#filename = "output_statistics_cifar_classimb_withAL_"
rounds=10

#for pneumonia
base_dir = "/home/venkat/trust-wassal/tutorials/results/svhn/classimb/rounds"+str(rounds)
#budgets=['5', '10', '15', '20', '25']
budgets=[100]
filename = "output_statistics_svhn_vanilla"

#strategies = ["WASSAL", "WASSAL_P", "fl1mi", "fl2mi", "gcmi", "logdetmi", "random","badge","us","glister","coreset","glister","gradmatch-tss","leastconf","logdetcmi","flcmi","margin"]
#strategy_group="allstrategies"
#strategies = ["WASSAL", "WASSAL_P","random","badge","us","glister","coreset","glister","gradmatch-tss","leastconf","margin"]
#strategy_group="AL"
#strategies = ["WASSAL_P","random","logdetcmi","flcmi"]
#strategy_group="withprivate"
#strategies = ["WASSAL",  "fl1mi", "fl2mi", "gcmi", "logdetmi","fl1mi_soft", "fl2mi_soft", "gcmi_soft", "logdetmi_soft", "random","WASSAL_P","logdetcmi","flcmi","logdetcmi_soft","flcmi_soft"]
#strategy_group="WASSAL_SOFT"
#strategies = ["random","badge","us","glister","coreset","glister","gradmatch-tss","leastconf","margin","badge_soft","us_soft","glister_soft","coreset_soft","glister_soft","gradmatch-tss_soft","leastconf_soft","margin_soft"]
strategies = ['WASSAL_WITHSOFT','glister','glister_withsoft','gradmatch-tss','gradmatch-tss_withsoft','us','us_withsoft','coreset','coreset_withsoft','leastconf','leastconf_withsoft','margin','margin_withsoft','random']
strategy_group="AL_WITH_SOFT"

#experiments=['exp1','exp2','exp3','exp4','exp5']
experiments=['exp1']

# Prepare the CSV file for saving stats
output_path = os.path.join(base_dir, filename+"_group_"+strategy_group+"_rounds_"+str(rounds))


def compute_stats(gains):
    mean_gain = sum(gains) / len(gains)
    variance = sum([(gain - mean_gain) ** 2 for gain in gains]) / len(gains)
    sd_gain = variance ** 0.5
    return mean_gain, variance, sd_gain

colors = [
    '#FF0000',  # Red
    '#00FF00',  # Green
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#00FFFF',  # Aqua
    '#FF00FF',  # Magenta
    '#FF4500',  # OrangeRed
    '#8A2BE2',  # BlueViolet
    '#A52A2A',  # Brown
    '#DEB887',  # BurlyWood
    '#5F9EA0',  # CadetBlue
    '#7FFF00',  # Chartreuse
    '#D2691E',  # Chocolate
    '#FF7F50',  # Coral
    '#6495ED',  # CornflowerBlue
    '#DC143C',  # Crimson
    '#00CED1',  # DarkTurquoise
    '#9400D3',  # DarkViolet
    '#FF1493',  # DeepPink
    '#00BFFF',  # DeepSkyBlue
]


 #mean gain of all classes
with open(output_path+"_allclasses.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Strategy", "Budget", "Mean Gain", "Variance", "Standard Deviation"])

    # Store data in dictionaries
    data = {}
    
    #for each budget
    for budget in budgets:
         
        #for each strategy
        for strategy in strategies:
           # Reset these lists for each strategy
            means = []
            sds = []
            budgets_list = []  # renamed to avoid conflict with 'budgets'
            
            #continue if a path does not exist with strategy and budget
            if not os.path.exists(os.path.join(base_dir, strategy,str(budget))):
                continue

            #calculate mean and sd of strategy across all runs
            gains = []
            for experiment in experiments:
                path = os.path.join(base_dir, strategy,str(budget),experiment)
                if not os.path.exists(path):
                    continue
                for csv_file in os.listdir(path):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(path, csv_file)
                    else:
                        continue
                    df = pd.read_csv(csv_path,header=None)
                    #cumlative gains from each row
                    #for i in range(0,5):
                    #gains=
                    y1 = df.iloc[0, 10]
                    y2 = df.iloc[rounds-1, 10]
                    gain1 = y2 - y1
                    y1 = df.iloc[0, 10]
                    y2 = df.iloc[rounds-2, 10]
                    gain2 = y2 - y1

                    gain=gain1 if gain1>gain2 else gain2
                    gains.append(gain)
                if not gains:
                    continue
            
             # Compute stats after processing all experiments for the current strategy and budget
            mean_gain, variance, sd_gain = compute_stats(gains)
            #round off mean_gain, variance, sd_gain
            mean_gain=round(mean_gain,2)
            variance=round(variance,2)
            sd_gain=round(sd_gain,2)
            writer.writerow([strategy, budget, mean_gain, variance, sd_gain])
            print(f"Strategy: {strategy}, Budget: {budget}")
            print("Mean Gain:", mean_gain)
            print("Variance:", variance)
            print("Standard Deviation of Gain:", sd_gain)
            print("----------------------------------------------------")

            # Save data to the dictionary
            if gains:
                
                if strategy not in data:
                    data[strategy] = {'means': [], 'sds': [], 'budgets': []}
                data[strategy]['means'].append(mean_gain)
                data[strategy]['sds'].append(sd_gain)
                data[strategy]['budgets'].append(budget)
    print(f"Statistics saved to {output_path}")



# Plot data
plt.figure(figsize=(10, 6))
color_index = 0

# Loop through the data dictionary to extract the values
for strategy, values in data.items():
    plt.plot(values['budgets'], values['means'], label=strategy,color=colors[color_index])
    # Add the strategy name to the end of the line using plt.text()
    x_pos = values['budgets'][-1]  # x-coordinate of the last point on the line
    y_pos = values['means'][-1]    # y-coordinate of the last point on the line
    plt.text(x_pos, y_pos, strategy, fontsize=12, color=colors[color_index])
    color_index += 1
    #plt.errorbar(values['budgets'], values['means'],values['sds'], label=strategy)

plt.xlabel('Budget')
plt.ylabel('Mean Gain for all classes')
plt.title('Mean Gain for all classes for '+str(rounds)+'AL rounds')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(output_path+"_allclasses.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

#write to a latex format
import pandas as pd

def generate_latex_table(data):
    budgets = sorted(data['Budget'].unique())
    strategies = data['Strategy'].unique()

    # Dictionary to hold the max value for each budget
    max_values = {}
    for budget in budgets:
        max_values[budget] = data[data['Budget'] == budget]['Mean Gain'].max()

    table = "\\begin{table}[h!]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{|l|*{%d}{c|}}\n" % len(budgets)
    table += "\\hline\n"
    table += "Strategy & " + " & ".join(map(str, budgets)) + " \\\\\n"
    table += "\\hline\n"
    table += "\\hline\n"

    for strategy in strategies:
        row_data = [strategy.replace("_", "\\_")]
        for budget in budgets:
            subset = data[(data['Strategy'] == strategy) & (data['Budget'] == budget)]
            if not subset.empty:
                mean_gain = subset['Mean Gain'].values[0]
                # Check if this is the max value for this budget
                if mean_gain == max_values[budget]:
                    row_data.append("\\textbf{" + str(mean_gain) + "}")
                else:
                    row_data.append(str(mean_gain))
            else:
                row_data.append('-')
        table += " & ".join(row_data) + " \\\\\n"
        table += "\\hline\n"

    table += "\\end{tabular}\n"
    table += "\\caption{Mean Gain for various strategies across budgets}\n"
    table += "\\label{tab:my_label}\n"
    table += "\\end{table}\n"

    return table


df = pd.read_csv(output_path+"_allclasses.csv")
latex_table = generate_latex_table(df)
print(latex_table)
#save to a file
with open(output_path+"_allclasses.tex", "w") as text_file:
    text_file.write(latex_table)