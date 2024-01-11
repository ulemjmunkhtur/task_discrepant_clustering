import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import graphs

def create_bar_graph_group(type_name, group1, group2, group3, letter):
    """
    Creates and saves a bar graph comparing a specific lexical data type across different groups.
    """
    categories = ['G1: Young Adult', 'G2: Older Adult', 'G3: MCI Patient']
    values = [group1, group2, group3]
    colors = sns.color_palette("Greens", 3)

    #create bar graph
    bars = plt.bar(categories, values, color=colors)
    plt.ylabel('Mean')
    plt.title(f"Comparison between the {type_name} of {letter.upper()}'s groups")


    plt.annotate(f'{group1:.4f}', (categories[0], group1 * 0.8), ha='center', va='top', color='black', fontsize=9)
    plt.annotate(f'{group2:.4f}', (categories[1], group2 * 0.8), ha='center', va='top', color='black', fontsize=9)
    plt.annotate(f'{group3:.4f}', (categories[2], group3 * 0.8), ha='center', va='top', color='black', fontsize=9)


    output_folder = "output/graphs/groups"
    filepath = os.path.join(output_folder, f"{letter}_Group_{type_name}_bar_graph.png")
    plt.savefig(filepath)

    print(f"Graph for the groups of {type_name} of letter {letter} saved as {filepath}")

    plt.show()

    plt.close() 

def create_combined_bar_graph(lexical_data, semantic_means, phonological_means, frequency_means, letter):
    """
    Creates and saves a combined bar graph comparing multiple lexical data types across different groups.
    Lexical Data: Semantic Similarity, Phonological Similarity, Frequency
    """
    categories = ['G1: Young Adult', 'G2: Older Adult', 'G3: MCI Patient']

    num_groups = len(lexical_data)

    bar_width = 0.2
    indices = np.arange(num_groups)

    plt.figure(figsize=(10, 6))

    #plotting each group's bars 
    for i, category in enumerate(categories):
        bars = plt.bar(indices + i * bar_width, 
                       [semantic_means[i], phonological_means[i], frequency_means[i]], 
                       width=bar_width, 
                       label=category)

        #adding annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', 
                     ha='center', va='bottom')

    plt.xlabel('Lexical Data Type', fontweight='bold')
    plt.xticks(indices + bar_width, lexical_data)
    plt.ylabel('Mean Value')
    plt.title(f'Comparison of Lexical Data Types across Groups for Letter {letter.upper()}')
    plt.legend(title="Groups")

    output_folder = "output/graphs/groups"
    filepath_combined = os.path.join(output_folder, f"{letter}_combined_bar_graph.png")
    plt.savefig(filepath_combined)
    print(f"Combined graph saved as {filepath_combined}")

    plt.show()
    plt.close()


def groups(letter):
    """
    Analyzes and creates bar graphs for different lexical data across participant groups.

    Parameters:
    letter (str): letter to analyze

    - Processes group data for participants.
    - Calculates mean values for semantic, phonological, and frequency data for each group.
    - Generates individual and combined bar graphs to compare these means across groups.

    Outputs:
    - Bar graphs saved as PNG files showing the comparisons of lexical data types 
      (semantic, phonological, frequency) across different participant groups.
    """
    df_results = pd.read_csv(f"output/letter_{letter}/indiv_stats_results_{letter}.csv")
    df_groups = pd.read_csv(f"forager/data/fluency_lists/raw_data/{letter}.csv")

    #rename the 'Subject' column to 'Participant' in df_results
    df_results.rename(columns={'Subject':'Participant'}, inplace=True)

    #filter out non-numeric values from the 'Participant' column of df_groups
    df_groups = df_groups[df_groups['Participant'].apply(lambda x: str(x).isnumeric())]

    #convert the 'Participant' columns in both dataframes to int for consistency
    df_results['Participant'] = df_results['Participant'].astype(int)
    df_groups['Participant'] = df_groups['Participant'].astype(int)


    #map the 'Group' column from df_groups to df_results based on 'Participant'
    df_results['Group'] = df_results['Participant'].map(df_groups.set_index('Participant')['Group'])

    # print(df_results.columns)
    # Filter the dataframe for rows where the groups are 
    group1_df = df_results.loc[df_results['Group'] == 1.0]
    group2_df = df_results.loc[df_results['Group'] == 2.0]
    group3_df = df_results.loc[df_results['Group'] == 3.0]

    # Calculate means for group 1
    group1_semantic_mean = group1_df['Semantic_Similarity_mean'].mean()
    group1_phonological_mean = group1_df['Phonological_Similarity_mean'].mean()
    group1_frequency_mean = group1_df['Frequency_Value_mean'].mean()

    #Calculate means for group 2 
    group2_semantic_mean = group2_df['Semantic_Similarity_mean'].mean()
    group2_phonological_mean = group2_df['Phonological_Similarity_mean'].mean()
    group2_frequency_mean = group2_df['Frequency_Value_mean'].mean()

    #Calculate means for group 2 
    group3_semantic_mean = group3_df['Semantic_Similarity_mean'].mean()
    group3_phonological_mean = group3_df['Phonological_Similarity_mean'].mean()
    group3_frequency_mean = group3_df['Frequency_Value_mean'].mean()

    create_bar_graph_group("Mean Semantic Similarity", group1_semantic_mean, group2_semantic_mean, group3_semantic_mean, letter)
    create_bar_graph_group("Mean Phonological Similarity", group1_phonological_mean, group2_phonological_mean, group3_phonological_mean, letter)
    create_bar_graph_group("Mean Frequency", group1_frequency_mean, group2_frequency_mean, group3_frequency_mean, letter)


    #creating the combined bar graphs 
    lexical_data = ["Semantic Similarity", "Phonological Similarity", "Frequency"]
    semantic_means = [group1_semantic_mean, group2_semantic_mean, group3_semantic_mean]
    phonological_means = [group1_phonological_mean, group2_phonological_mean, group3_phonological_mean]
    frequency_means = [group1_frequency_mean, group2_frequency_mean, group3_frequency_mean]


    create_combined_bar_graph(lexical_data, semantic_means, phonological_means, frequency_means, letter)

def significance_tests(letter):

    df_results = pd.read_csv(f"output/letter_{letter}/indiv_stats_results_{letter}.csv")
    df_groups = pd.read_csv(f"forager/data/fluency_lists/raw_data/{letter}.csv")
    df_results.rename(columns={'Subject':'Participant'}, inplace=True)
    df_groups = df_groups[df_groups['Participant'].apply(lambda x: str(x).isnumeric())]
    df_results['Participant'] = df_results['Participant'].astype(int)
    df_groups['Participant'] = df_groups['Participant'].astype(int)
    df_results['Group'] = df_results['Participant'].map(df_groups.set_index('Participant')['Group'])

    #calculate means for each group
    group1_df = df_results.loc[df_results['Group'] == 1.0]
    group2_df = df_results.loc[df_results['Group'] == 2.0]
    group3_df = df_results.loc[df_results['Group'] == 3.0]
    group1_semantic_mean = group1_df['Semantic_Similarity_mean'].mean()
    group1_phonological_mean = group1_df['Phonological_Similarity_mean'].mean()
    group1_frequency_mean = group1_df['Frequency_Value_mean'].mean()
    group2_semantic_mean = group2_df['Semantic_Similarity_mean'].mean()
    group2_phonological_mean = group2_df['Phonological_Similarity_mean'].mean()
    group2_frequency_mean = group2_df['Frequency_Value_mean'].mean()
    group3_semantic_mean = group3_df['Semantic_Similarity_mean'].mean()
    group3_phonological_mean = group3_df['Phonological_Similarity_mean'].mean()
    group3_frequency_mean = group3_df['Frequency_Value_mean'].mean()

    #create bar graphs
    create_bar_graph_group("Mean Semantic Similarity", group1_semantic_mean, group2_semantic_mean, group3_semantic_mean, letter)
    create_bar_graph_group("Mean Phonological Similarity", group1_phonological_mean, group2_phonological_mean, group3_phonological_mean, letter)
    create_bar_graph_group("Mean Frequency", group1_frequency_mean, group2_frequency_mean, group3_frequency_mean, letter)


    combined_data = pd.concat([
        group1_df.assign(Group='G1'),
        group2_df.assign(Group='G2'),
        group3_df.assign(Group='G3')
    ])

    for measure in ['Semantic_Similarity_mean', 'Phonological_Similarity_mean', 'Frequency_Value_mean']:
        p_value = graphs.anova_test(
            means=combined_data.groupby('Group')[measure].mean(),
            std_devs=combined_data.groupby('Group')[measure].std(),
            sizes=combined_data.groupby('Group')[measure].count()
        )
        print(f"P-value for {measure} ANOVA test:", p_value)

        # Tukey HSD test
        tukey_result = graphs.tukey_hsd_test(combined_data, 'Group', measure)
        print(f"Tukey HSD test results for {measure}:\n", tukey_result)








## SAMPLE CODE TEST ##
# groups("a")
# significance_tests('a')
