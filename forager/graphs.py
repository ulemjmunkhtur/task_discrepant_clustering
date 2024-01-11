import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def calculate_standard_error(data, column_name):
    """Calculate the standard error of the mean for a specified column."""
    sample_std = np.nanstd(data[column_name], ddof=1)
    sample_size = data[column_name].count()
    return sample_std / np.sqrt(sample_size)

def calculate_mean_methods(data):
    """
    calculate the mean number of switches and the mean cluster size for each switch method within the data.

    Parameters:
    - data: A DF containing the 'Switch_Method', 'Number_of_Switches', and 'Cluster_Size_mean' columns.

    Returns:
    - switches: A dictionary mapping switch methods to their corresponding mean number of switches
    - clusters: A dictionary mapping switch methods to their corresponding mean cluster size 
    """
    switch_methods = data['Switch_Method'].unique()
    switches = {}
    clusters = {}
    
    for method in switch_methods:
        filtered_data = data[data['Switch_Method'] == method]
        mean_switches = filtered_data['Number_of_Switches'].mean()
        mean_clusters = filtered_data['Cluster_Size_mean'].mean()
        switches[method] = mean_switches
        clusters[method] = mean_clusters
    
    return switches, clusters

def calculate_nll(data):
    """
    calculate the sum of negative log likelihood for each delta method in the dynamic delta model

    Parameters
    - data: A DF containing the 'Model' and 'Negative_Log_Likelihood_Optimized' columns.

    Returns:
    - nll: A dictionary with each model mapped to their corresponding sum of negative log likelihood 
    """
    models = data['Model'].unique()
    nll = {}
    for model in models:
        filtered_data = data[data['Model'] == model]
        sum_nll = filtered_data['Negative_Log_Likelihood_Optimized'].sum()
        nll[model] = sum_nll
    return nll
def create_bar_graph(type_name, values, standard_errors):
    """
    Create and save a bar graph with error bars.

    Parameters: 
    - type_name: Title of the graph.
    - values: A list of mean values for each category.
    - standard_errors: A list of standard errors corresponding to the mean values.
    """
    categories = ['letter_f', 'letter_a', 'letter_s', 'animals']
    colors = sns.color_palette("Greens", len(categories))

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors, yerr=standard_errors, capsize=5)
    plt.ylabel('Mean')
    plt.title(type_name)

    for i, value in enumerate(values):
        plt.annotate(f'{value:.4f}', (categories[i], value), ha='center', va='bottom', color='black', fontsize=9)

    output_folder = "output/graphs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filepath = os.path.join(output_folder, f"{type_name.replace(' ', '_')}_bar_graph.png")
    plt.savefig(filepath, bbox_inches='tight')

    print(f"Bar graph for {type_name} saved as {filepath}")

    plt.show()
    plt.close()

def create_line_graph(df, name):
    """
    Create and display a line graph for the specified dataset and metric.

    Args:
    - df: A DataFrame containing the data to plot.
    - name: A string representing the name of the metric to be plotted (e.g., 'Mean Switches').

    Returns:
    - None: This function displays the generated line graph.
    """
    sns.set_style("darkgrid", {'axes.spines.right': False, 'axes.spines.top': False})
    plt.rcParams['font.family'] = 'Georgia'
    palette = sns.color_palette("Dark2", n_colors=len(df['Switch_Method'].unique()))

  
    plt.figure(figsize=(16, 10))
    
    ax = plt.gca()  

    for i, switch_method in enumerate(df['Switch_Method'].unique()):
        subset = df[df['Switch_Method'] == switch_method]
        line = sns.lineplot(
            data=subset, 
            x='Dataset', 
            y=f'{name}', 
            label=switch_method, 
            linewidth=3.5 if switch_method == 'simdrop' else 1.5,
            color=palette[i],
            marker='o' if switch_method == 'simdrop' else None
        )

        #the numerical x-coordinate for the last data point
        last_point_x = ax.get_xticks()[-1]
        last_point_y = subset[name].values[-1]  #get the last y value

        #offset the label (of the switch method) slightly to the right of the last data point
        plt.text(last_point_x + 0.1, last_point_y, switch_method, 
                 horizontalalignment='left', size='small', color=palette[i], transform=ax.transData)

    plt.legend(title='Switch Method', bbox_to_anchor=(-0.05, 1), loc='upper right')
    plt.ylabel('Mean', fontsize=14)
    plt.title(f'{name}', fontsize=16)
    plt.tight_layout()

    output_folder = "output/graphs"  
    filepath = os.path.join(output_folder, f"{name.replace(' ','_')}.png")
    plt.savefig(filepath)

    print(f"Line graph for {name} saved as {filepath}")
    plt.show()



def compare_lexical_results():
    data_f = pd.read_csv("output/letter_f/indiv_stats_results_f.csv")
    data_a = pd.read_csv("output/letter_a/indiv_stats_results_a.csv")
    data_s = pd.read_csv("output/letter_s/indiv_stats_results_s.csv")
    data_animals = pd.read_csv("output/letter_animals/indiv_stats_results_animals.csv")
    
    #calculate the mean of the Semantic_Similarity_mean column
    semantic_f = data_f['Semantic_Similarity_mean'].mean()
    semantic_a = data_a['Semantic_Similarity_mean'].mean()
    semantic_s = data_s['Semantic_Similarity_mean'].mean()
    semantic_animals = data_animals['Semantic_Similarity_mean'].mean()

   # calculate standard errors for Semantic Similarity
    sem_f = calculate_standard_error(data_f, 'Semantic_Similarity_mean')
    sem_a = calculate_standard_error(data_a, 'Semantic_Similarity_mean')
    sem_s = calculate_standard_error(data_s, 'Semantic_Similarity_mean')
    sem_animals = calculate_standard_error(data_animals, 'Semantic_Similarity_mean')

    #calculate the mean frequency
    frequency_f = data_f['Frequency_Value_mean'].mean()
    frequency_a = data_a['Frequency_Value_mean'].mean()
    frequency_s = data_s['Frequency_Value_mean'].mean()
    frequency_animals = data_animals['Frequency_Value_mean'].mean()

     # Calculate standard errors for Frequency
    freq_sem_f = calculate_standard_error(data_f, 'Frequency_Value_mean')
    freq_sem_a = calculate_standard_error(data_a, 'Frequency_Value_mean')
    freq_sem_s = calculate_standard_error(data_s, 'Frequency_Value_mean')
    freq_sem_animals = calculate_standard_error(data_animals, 'Frequency_Value_mean')

    #calculate the mean phonological similarity 
    phon_f = data_f['Phonological_Similarity_mean'].mean()
    phon_a = data_a['Phonological_Similarity_mean'].mean()
    phon_s = data_s['Phonological_Similarity_mean'].mean()
    phon_animals = data_animals['Phonological_Similarity_mean'].mean()

    # Calculate standard errors for Phonological Similarity
    phon_sem_f = calculate_standard_error(data_f, 'Phonological_Similarity_mean')
    phon_sem_a = calculate_standard_error(data_a, 'Phonological_Similarity_mean')
    phon_sem_s = calculate_standard_error(data_s, 'Phonological_Similarity_mean')
    phon_sem_animals = calculate_standard_error(data_animals, 'Phonological_Similarity_mean')


    #create bar graphs with error bars
    create_bar_graph("Mean Semantic Similarity", [semantic_f, semantic_a, semantic_s, semantic_animals], [sem_f, sem_a, sem_s, sem_animals])
    create_bar_graph("Mean Phonological Similarity", [phon_f, phon_a, phon_s, phon_animals], [phon_sem_f, phon_sem_a, phon_sem_s, phon_sem_animals])
    create_bar_graph("Mean Frequency", [frequency_f, frequency_a, frequency_s, frequency_animals], [freq_sem_f, freq_sem_a, freq_sem_s, freq_sem_animals])


    #calculates the mean number of switches and the mean cluster size for each switch method 
    mean_switches_f, mean_clusters_f = calculate_mean_methods(data_f)
    mean_switches_a, mean_clusters_a = calculate_mean_methods(data_a)
    mean_switches_s, mean_clusters_s = calculate_mean_methods(data_s)
    mean_switches_animals, mean_clusters_animals = calculate_mean_methods(data_animals)

    #creating a df summarizing results by switch method 
    data = []
    for method in mean_switches_f.keys():
        data.append(['F', method, mean_switches_f[method], mean_clusters_f[method]])
        data.append(['A', method, mean_switches_a[method], mean_clusters_a[method]])
        data.append(['S', method, mean_switches_s[method], mean_clusters_s[method]])
        data.append(['Animals', method, mean_switches_animals[method], mean_clusters_animals[method]])

    switch_df = pd.DataFrame(data, columns=['Dataset', 'Switch_Method', 'Mean Switches', 'Mean Clusters'])
 
    create_line_graph(switch_df, "Mean Switches")
    create_line_graph(switch_df, "Mean Clusters")

def anova_test(means, std_devs, sizes):
    """
    given lists of means, standard deviations, and sizes
    """
    #sum of squares within groups
    ss_within = sum((size - 1) * std_dev**2 for size, std_dev in zip(sizes, std_devs))

    #grand mean
    grand_mean = sum(mean * size for mean, size in zip(means, sizes)) / sum(sizes)

    #compute sum of squares between groups
    ss_between = sum(size * (mean - grand_mean)**2 for size, mean in zip(sizes, means))

    #compute degrees of freedom
    df_within = sum(sizes) - len(sizes)
    df_between = len(sizes) - 1

    #compute mean square values
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    #compute F statistic
    f_statistic = ms_between / ms_within

    #compute p-value
    p_value = stats.f.sf(f_statistic, df_between, df_within)

    return p_value

def tukey_hsd_test(data, group_col, value_col):
    """
    Perform Tukey's HSD (Honest Significant Difference) test for pairwise comparisons.
    """
    if data[group_col].isnull().any() or data[value_col].isnull().any():
        print(" Missing values found.")
        data = data.dropna(subset=[group_col, value_col])

    # Perform Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=data[value_col], groups=data[group_col], alpha=0.05)
    return tukey_result

def statistical_significance():
    data_f = pd.read_csv("output/letter_f/indiv_stats_results_f.csv").assign(Group='F')
    data_a = pd.read_csv("output/letter_a/indiv_stats_results_a.csv").assign(Group='A')
    data_s = pd.read_csv("output/letter_s/indiv_stats_results_s.csv").assign(Group='S')
    data_animals = pd.read_csv("output/letter_animals/indiv_stats_results_animals.csv").assign(Group='Animals')

    combined_data = pd.concat([data_f, data_a, data_s, data_animals])

    #anova
    lexical_measures = ['Semantic_Similarity_mean', 'Phonological_Similarity_mean', 'Frequency_Value_mean']

    #doing it for each of these
    for measure in lexical_measures:
        p_value = anova_test(
            means=combined_data.groupby('Group')[measure].mean(),
            std_devs=combined_data.groupby('Group')[measure].std(),
            sizes=combined_data.groupby('Group')[measure].count()
        )
        print(f"P-value for {measure} ANOVA test:", p_value)

        # Tukey HSD test
        tukey_result = tukey_hsd_test(combined_data, 'Group', measure)
        print(f"Tukey HSD test results for {measure}:\n", tukey_result)


   
statistical_significance()

def model_results(letter):
    """
    compares the negative log likelihood results from different foraging models for a given letter.
    
    This function reads the results from static, dynamic simdrop, and dynamic delta models, calculates the
    sum of negative log likelihoods for each (lowest between the delta methods)
    
    Parameters:
    - letter: str, the letter to analyze
    
    Returns:
    - None: prints out the df with the NLL 
    """
    static = pd.read_csv(f"output/letter_{letter}/static_forager_{letter}.csv")
    dynamic_simdrop = pd.read_csv(f"output/letter_{letter}/dynamic_simdrop_forager_{letter}.csv")
    dynamic_delta = pd.read_csv(f"output/letter_{letter}/dynamic_delta_forager_{letter}.csv")
    pstatic = pd.read_csv(f"output/letter_{letter}/pstatic_forager_{letter}.csv")
    pdynamic_simdrop = pd.read_csv(f"output/letter_{letter}/pdynamic_simdrop_forager_{letter}.csv")
    pdynamic_multimodal = pd.read_csv(f"output/letter_{letter}/pdynamic_multimodal_forager_{letter}.csv")
    static_nll = static['Negative_Log_Likelihood_Optimized'].sum()
    dynamic_simdrop_nll = dynamic_simdrop['Negative_Log_Likelihood_Optimized'].sum()

    delta_nll = calculate_nll(dynamic_delta)
    lowest_delta_nll_key = min(delta_nll, key=delta_nll.get)
    lowest_delta_nll_value = delta_nll[lowest_delta_nll_key]
    
    pstatic_nll = pstatic['Negative_Log_Likelihood_Optimized'].sum()
    pdynamic_simdrop_nll = pdynamic_simdrop["Negative_Log_Likelihood_Optimized"].sum()
    multimodal_nll = calculate_nll(pdynamic_multimodal)
    lowest_multimodal_nll_key = min(multimodal_nll, key=multimodal_nll.get)
    lowest_multimodal_nll_value = multimodal_nll[lowest_multimodal_nll_key]

    nll_df = pd.DataFrame({
        'Model': ['Static', 'Dynamic Simdrop', 'Dynamic Delta', "Pstatic","Pdynamic Simdrop","Pdynamic Multimodal"],
        'Negative Log Likelihood': [static_nll, dynamic_simdrop_nll, lowest_delta_nll_value, pstatic_nll, pdynamic_simdrop_nll, lowest_multimodal_nll_value ]
    })
    nll_df.to_csv(f'output/letter_{letter}/nll_{letter}.csv', index=False)
    nll_df_string = nll_df.to_string(index=False)
    

    print(f"Letter {letter.upper()} model results was saved as output/letter_{letter}/nll_{letter}.csv")
    print(nll_df_string)
    print(f"The lowest dynamic delta NLL method is {lowest_delta_nll_key}")
    print(f"The lowest pdynamic multimodal NLL method is {lowest_multimodal_nll_value}")
    print("\n")
    
    return nll_df, lowest_delta_nll_key, lowest_multimodal_nll_key