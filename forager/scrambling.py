import math
import pandas as pd
import random
import forager.run_foraging_phon
import forager.run_foraging_phon as run_foraging_norms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import forager.forager.utils
import matplotlib.patches as mpatches
from scipy import stats


def scramble(letter, type):
    """
    scrambles participant response data 1000 times (for each subject, the same words but reordered randomly)

    After shuffling participant responses 1000 times, it obtains the lexical results for each iteration. 
    From the lexical results, it calculates the mean semantic similarity and saves all the means in a csv file. 
  
    Parameters:
    letter (str): which letter data to analyze

    Outputs:
    A CSV file containing mean semantic similarities for each of the 1000 shuffled data. 
    """
    # #participant data
    # original_data = pd.read_csv(f'transformed_data/{letter}_transformed.csv')

    # similarity_matrix = np.loadtxt(f"/Users/ulemjmunkhtur/Desktop/GitHub/task-discrepant-clustering/forager/data/lexical_data/letter_{letter}/USE_semantic_matrix_{letter}.csv", delimiter=',')
    # frequency_list = np.array(pd.read_csv(f"/Users/ulemjmunkhtur/Desktop/GitHub/task-discrepant-clustering/forager/data/lexical_data/letter_{letter}/USE_frequencies.csv",header=None,encoding="unicode-escape")[1])
    # labels = pd.read_csv(f"/Users/ulemjmunkhtur/Desktop/GitHub/task-discrepant-clustering/forager/data/lexical_data/letter_{letter}/USE_frequencies.csv",header=None)[0].values.tolist()
    # phon_matrix = np.loadtxt(f"/Users/ulemjmunkhtur/Desktop/GitHub/task-discrepant-clustering/forager/data/lexical_data/letter_{letter}/USE_phon_matrix_{letter}.csv",delimiter=',')
    # # labels = [label.lower().strip() for label in labels]
    #loading data 
    similarity_matrix = np.loadtxt(f"forager/data/lexical_data/letter_{letter}/USE_semantic_matrix_{letter}.csv",delimiter=',')
    frequency_list = np.array(pd.read_csv(f"forager/data/lexical_data/letter_{letter}/USE_frequencies.csv",header=None,encoding="unicode-escape")[1])
    labels = pd.read_csv(f"forager/data/lexical_data/letter_{letter}/USE_frequencies.csv",header=None)[0].values.tolist()
    phon_matrix = np.loadtxt(f"forager/data/lexical_data/letter_{letter}/USE_phon_matrix_{letter}.csv",delimiter=',')

    participant_df = pd.read_csv(f"forager/data/fluency_lists/transformed_data/{letter}_transformed.csv")

    
    all_means = []
    for i in range(1000):
    #shuffle the responses
        shuffled_data = []
        for participant, group in participant_df.groupby('Participant'):
            #responses to lower case
            responses = group['Response'].str.lower().tolist()
            shuffled_responses = random.sample(responses, len(responses))
            shuffled_data.append((participant, shuffled_responses))


        #correct format for responses
        data_tuples = []
        for participant, responses in shuffled_data:
            data_tuples.append((participant, responses))

        #get lexical results through running lexical
        lexical_results = forager.run_foraging_phon.run_lexical(data_tuples, similarity_matrix, phon_matrix, frequency_list, labels)
        mean_semantic_similarity = lexical_results[type].mean()
        print(f"Iteration {i}: Mean {type} Calculated - {mean_semantic_similarity}")
        all_means.append(mean_semantic_similarity)

    similarity_df = pd.DataFrame({f'Mean {type}': all_means})
    similarity_df.to_csv(f"forager/scrambled_means/{letter}_{type}.csv", index=False)


def distribution_plot(letter, type, actual_mean):
    """
    generates and saves a distribution plot for mean semantic similarities of scrambled data.

    reads the computed mean semantic similarities from a CSV file and creates a histogram
    plot using Seaborn. The plot highlights the actual mean semantic similarity

    Parameters:
    letter (str): which letter data to analyze
    actual_mean_semantic_similarity (float): The actual mean semantic similarity value 

    Outputs:
    A PNG of the distribution plot.
    """
    
    data = pd.read_csv(f"forager/scrambled_means/{letter}_{type}.csv")

    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'Georgia'
    plt.figure(figsize=(15, 6))
    ax = sns.histplot(data[f'Mean {type}'], kde=True, color='blue', bins=30)
    ax.set_xlim(left=0, right=0.5)


    plt.axvline(actual_mean, color='red', linestyle='dashed', linewidth=2)



    plt.title(f"Distribution of the Mean {type} of 1000 Shuffled {letter.upper()} Data")
    plt.xlabel(f'Mean {type}')
    plt.ylabel('Frequency')
    plt.text(actual_mean, plt.ylim()[1] * 0.95, f'Actual Mean: {actual_mean:.4f}  ', color='red', horizontalalignment='right')
 
    
    output_folder = "output/graphs"
    filepath = os.path.join(output_folder, f"{letter}_{type}_shuffled_distribution.png")
    plt.savefig(filepath)
    print(f"Shuffled data of {letter} of type {type} is saved as {filepath}")
    plt.show()


def combined_distribution_plots(dfs, actual_means, letters, type):
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.figure(figsize=(15, 6))

    custom_colors = ["red", "green", "blue", "yellow"]
    colors = sns.color_palette(custom_colors, n_colors=len(dfs))
    legend_handles = []  

    for df, actual_mean, letter, color in zip(dfs, actual_means, letters, colors):
        sns.histplot(df[f'Mean {type}'], kde=True, bins=30, color=color, label=f'{letter.upper()} Data')
        plt.axvline(actual_mean, color=color, linestyle='dashed', linewidth=2)
        
        legend_handles.append(mpatches.Patch(color=color, label=f'{letter.upper()} Data'))
        legend_handles.append(mpatches.Patch(color=color, label=f'{letter.upper()} Actual Mean {type}: {actual_mean:.4f}', linestyle='dashed'))

    plt.title(f'Combined Distribution of Mean {type} of Shuffled Data')
    plt.xlabel(f'Mean {type}')
    plt.ylabel('Frequency')

    # Use the custom legend handles here
    plt.legend(handles=legend_handles, title=f'Distributions and Actual Means')
    plt.xlim(0, 0.5)
    output_folder = "output/graphs"
    filepath = os.path.join(output_folder, f"combined_{type}_shuffled_distribution.png")
    plt.savefig(filepath)

    print(f"Combined Shuffled data of type {type} is saved as {filepath}")
    plt.show()


def test_significance(letter, csv_file_path, column_name, actual_mean, significance_level=0.05):
    """
    Perform a one-sample t-test to compare the mean of scrambled data with the actual mean for a specified letter.

    Parameters:
    letter (str): The letter corresponding to the dataset (e.g., 'a', 'f', 's', 'animals').
    csv_file_path (str): Path to the CSV file containing scrambled data means.
    column_name (str): Name of the column in the CSV file that contains the means.
    actual_mean (float): The actual mean to compare against.
    significance_level (float): The significance level for the test (default is 0.05).

    Returns:
    None: Prints the results of the t-test.
    """

    #load scrambled data means from the CSV file
    scrambled_means = pd.read_csv(csv_file_path)[column_name]

    #perform the t-test
    t_statistic, p_value = stats.ttest_1samp(scrambled_means, actual_mean)

    print(letter.upper(),column_name)
    print(f"T-statistic for {letter.upper()}: {t_statistic}")
    print(f"P-value for {letter.upper()}: {p_value}")

    #interpret the results
    if p_value < significance_level:
        print(f"There is a significant difference between the actual mean and the mean of the scrambled data for letter {letter.upper()}.")
    else:
        print(f"There is no significant difference between the actual mean and the mean of the scrambled data for letter {letter.upper()}.")



def plot_distributions():
    #semantic similarities of each 
    distribution_plot("f", "Semantic_Similarity", 0.2802965447238993)
    distribution_plot("a", "Semantic_Similarity",0.24923782797487523)
    distribution_plot("s", "Semantic_Similarity",0.27174260919705256)
    distribution_plot("animals", "Semantic_Similarity", 0.43188669017027626)

    #combined semantic similarity 

    a_data_sem = pd.read_csv(f"forager/scrambled_means/a_Semantic_Similarity.csv")
    f_data_sem = pd.read_csv(f"forager/scrambled_means/f_Semantic_Similarity.csv")
    s_data_sem = pd.read_csv(f"forager/scrambled_means/s_Semantic_similarity.csv")
    animals_data_sem = pd.read_csv(f"forager/scrambled_means/animals_Semantic_Similarity.csv")

    dfs = [a_data_sem, f_data_sem, s_data_sem, animals_data_sem]
    actual_means_sem = [ 0.24923782797487523, 0.2802965447238993, 0.27174260919705256, 0.43188669017027626]
    letters = ["a", "f", "s", "animals"]
    combined_distribution_plots(dfs, actual_means_sem, letters, "Semantic_Similarity")

   


    #phonological similarity of each 
    distribution_plot("f", "Phonological_Similarity", 0.2865961763502669)
    distribution_plot("a", "Phonological_Similarity", 0.25896070346527433)
    distribution_plot("s", "Phonological_Similarity", 0.2865961763502669)
    distribution_plot("animals", "Phonological_Similarity", 0.2173854922778406)

    #combined phonological similarity 
    a_data_phon = pd.read_csv(f"forager/scrambled_means/a_Phonological_Similarity.csv")
    f_data_phon = pd.read_csv(f"forager/scrambled_means/f_Phonological_Similarity.csv")
    s_data_phon = pd.read_csv(f"forager/scrambled_means/s_Phonological_Similarity.csv")
    animals_data_phon = pd.read_csv(f"forager/scrambled_means/animals_Phonological_Similarity.csv")

    dfs_phon = [a_data_phon, f_data_phon, s_data_phon, animals_data_phon]
    actual_means_phon = [ 0.25896070346527433, 0.2865961763502669, 0.2933515691035733, 0.2173854922778406]

    combined_distribution_plots(dfs_phon, actual_means_phon, letters, "Phonological_Similarity")




def run_test_significance():

    #semantic similarity 
    test_significance("a", "forager/scrambled_means/a_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.24923782797487523)
    test_significance("f", "forager/scrambled_means/f_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.2802965447238993)
    test_significance("s", "forager/scrambled_means/s_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.27174260919705256)
    test_significance("animals", "forager/scrambled_means/animals_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.43188669017027626)
    # test_significance("f", "path_to_your_scrambled_data_f.csv", "Mean Semantic_Similarity", 0.2802965447238993)

    #phonological similarity 
    test_significance("a", "forager/scrambled_means/a_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.25896070346527433)
    test_significance("f", "forager/scrambled_means/f_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.2865961763502669)
    test_significance("s", "forager/scrambled_means/s_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.2933515691035733)
    test_significance("animals", "forager/scrambled_means/animals_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.2173854922778406)

    


#### SAMPLE RUN CODE ####
# scramble("a", "Phonological_Similarity")
# distribution_plot("a", "Semantic_Similarity",0.24923782797487523)
# test_significance("a", "forager/scrambled_data/a_semantic_similarity.csv", "Mean Semantic_Similarity", 0.24923782797487523)




