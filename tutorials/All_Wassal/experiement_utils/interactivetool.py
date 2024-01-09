import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ipywidgets import widgets
from IPython.display import display
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

import os
import pandas as pd

def parse_stats_file(file_path):
    stats = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(": ")
            stats[key] = float(value.replace('%', ''))  # Convert to float, remove '%' if present
    return stats

def collect_data(base_path):
    data_records = []
    for root, _, files in os.walk(base_path):
        if 'simplex_data_stat.txt' in files:
            # Split the path and extract relevant parts
            parts = root.split(os.sep)
            try:
                al_strategy = parts[-7]  # e.g., 'leastconf_withsoft'
                budget = parts[-6]       # e.g., '20'
                experiment = parts[-5]   # e.g., 'exp3'
                round_num = parts[-2]    # e.g., '2'

                # Parse the stats file
                stats = parse_stats_file(os.path.join(root, 'simplex_data_stat.txt'))
                for stat_name, stat_value in stats.items():
                    record = {
                        'AL Strategy': al_strategy, 
                        'Budget': budget, 
                        'Experiment': experiment, 
                        'Round': round_num,
                        'Statistic': stat_name, 
                        'Value': stat_value
                    }
                    data_records.append(record)

            except IndexError:
                # Skip if the path format is unexpected
                continue

    return pd.DataFrame(data_records)

# Example usage
# df = collect_data('/home/wassal/trust-wassal/tutorials/results/softloss0.3/pneumoniamnist/classimb/rounds10')





import plotly.graph_objs as go
from ipywidgets import widgets, HBox, VBox
from IPython.display import display

def interactive_visualization(df):
    # Get unique values for dropdowns
    experiments = df['Experiment'].unique()
    stats = df['Statistic'].unique()

    # Create dropdown widgets
    experiment_dropdown = widgets.Dropdown(options=experiments, value=experiments[0], description='Experiment:')
    stat_dropdown = widgets.Dropdown(options=stats, value=stats[0], description='Statistic:')

    # Plotting function
    def update_plot(change):
        experiment = experiment_dropdown.value
        stat = stat_dropdown.value
        filtered_df = df[(df['Experiment'] == experiment) & (df['Statistic'] == stat)]

        # Create figure
        fig = go.Figure()
        for al_strategy in filtered_df['AL Strategy'].unique():
            strategy_df = filtered_df[filtered_df['AL Strategy'] == al_strategy]
            fig.add_trace(go.Scatter(x=strategy_df['Round'], y=strategy_df['Value'], mode='lines+markers', 
                                     name=al_strategy))

        # Update layout
        fig.update_layout(title=f'{stat} for {experiment}',
                          xaxis_title='Round',
                          yaxis_title=stat,
                          legend_title='AL Strategy')
        fig.show()

    # Display widgets and initial plot
    experiment_dropdown.observe(update_plot, names='value')
    stat_dropdown.observe(update_plot, names='value')
    display(HBox([experiment_dropdown, stat_dropdown]))
    update_plot(None)

# df = collect_data('path_to_round10_folder')
# interactive_visualization(df)

# Example usage after collecting data
# df = collect_data('path_to_round10_folder')
# interactive_visualization(df)

# Example usage after collecting data
df = collect_data('/home/wassal/trust-wassal/tutorials/results/softloss0.3/pneumoniamnist/classimb/rounds10')
interactive_visualization(df)
