# task-discrepant-clustering

Task-Discrepant Clustering extends the functionality of the forager python package to help with the analysis of verbal fluency data. This is specifically to analyze how much semantic influence there is on phonological tasks and vice versa. 

Fluency List Creation: uses 3 corpa to generate lists of words beginning with a specific letter. selects the words with the highest frequency (f, a, and s data all have around 2000-2500). Found in forager.forager.words.py

#1 must come before all else. #2 must come next. 
#3, #4, #5 are interchangable. #6 & #7 depend on #5
#8 only depends on #2. 

Workflow: 
# 1) Data Transformation: 
Converts raw fluency data into a structured format for analysis. Can be found in forager.forager.transform_data.py
        transform.transform_data('path/to/raw_data.csv')
# 2) Lexical and Model Data Attainment: 
Run lexical and model analyses for each dataset to extract lexical metrics and model performances. Uses all switch methods and models except those associated with Troyer norms. Can be found in forager.run_foraging_phon.py 
        forager.run_foraging_phon.run_letter("letter")
# 3)Lexical Results Analysis/Visualization
use lexical results to compute 3 bar graphs (Mean Frequency, Mean Phonological Similarity, Mean Semantic Similarity). also obtains Mean Switches and Cluster Size line graphs that show the difference amongst all switch methods for F, A, S and animals. The primary methods can be found in forager.graphs
        forager.graphs.compare_lexical_results()
# 4) Model Results Analysis/Visualization 
model results to compute the mean negative log likelihoods for each model (static, dynamic simdrop, dynamic delta, pstatic, pdynamic simdrop, pdynamic multimodal). A lower number can suggest it's a better model to assess the data with. This can be found in forager.graphs
# 5) Data Scrambling 
Checks the robustness of the lexical results (mean semantic similarity and mean phonological similarity) through the scrambling of participant data. create 1000 scrambled participant data (words that each participant says are shuffled) and obtain the mean semantic similarity & mean phonological similarity 
# 6 & 7) Assesss Significance Through Distribution Plots & T-testing
test for significance between the 1000 shuffled data means vs the actual means using t-tests 
        scrambling.plot_distributions()
        scrambling.test_significance("a", "forager/scrambled_means/a_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.24923782797487523)
# 8) Group Based Analysis 
the fluency list data for F A S are from Taler V, Johns BT, Sheppard C, Young K, Jones MN. (2013). They conducted the VFT on 3 groups of people: G1: Young Adult, G2: Older Adult, G3: MCI Patient. For all the steps previously we're not distinguishing between the groups. But forager.groups includes functionality to compare mean semantic similarity, mean phonological similarity, and frequency across groups. 
        groups.groups("a")


