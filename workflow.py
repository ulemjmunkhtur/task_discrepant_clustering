import forager.forager.transform_data as transform
import forager
import forager.graphs as graphs
import forager.scrambling as scrambling 
import forager.groups as groups

# 1) transforming raw data into a format needed to run foraging 
transform.transform_data('forager/data/fluency_lists/raw_data/a.csv')
transform.transform_data('forager/data/fluency_lists/raw_data/f.csv')
transform.transform_data('forager/data/fluency_lists/raw_data/s.csv')
transform.transform_data('forager/data/fluency_lists/raw_data/animals.csv')

"""
do next steps assuming you have the following files for each fluency list
1. frequencies
2. embeddings
3. phonological similarity matrix
4. semantic similarity matrix
"""
# 2) get lexical and model results for each letter 
forager.run_foraging_phon.run_letter("a")
forager.run_foraging_phon.run_letter("s")
forager.run_foraging_phon.run_letter("f")

# 3) use lexical results to compute 3 bar graphs (Mean Frequency, Mean Phonological Similarity, Mean Semantic Similarity)
# also obtains Mean Switches and Cluster Size line graphs that show the difference amongst all switch methods for F, A, S and animals 
forager.graphs.compare_lexical_results()

# 4) use model results to compute the mean negative log likelihoods for each model (static, dynamic simdrop, dynamic delta, pstatic, pdynamic simdrop, pdynamic multimodal)
graphs.model_results("f")
graphs.model_results("a")
graphs.model_results("s")

# 5) create 1000 scrambled participant data (words that each participant says are shuffled) and obtain the mean semantic similarity & mean phonological similarity 
scrambling.scramble("a", "Phonological_Similarity")
scrambling.scramble("s", "Phonological_Similarity")
scrambling.scramble("f", "Phonological_Similarity")
scrambling.scramble("animals", "Phonological_Similarity")

scrambling.scramble("a", "Semantic_Similarity")
scrambling.scramble("s", "Semantic_Similarity")
scrambling.scramble("f", "Semantic_Similarity")
scrambling.scramble("animals", "Semantic_Similarity")

# 6) plot these 1000 means against the actual mean in a distribution plot 
scrambling.plot_distributions()

# 7) test for significance between the 1000 shuffled data means vs the actual means using t-tests 
 #semantic similarity 
scrambling.test_significance("a", "forager/scrambled_means/a_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.24923782797487523)
scrambling.test_significance("f", "forager/scrambled_means/f_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.2802965447238993)
scrambling.test_significance("s", "forager/scrambled_means/s_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.27174260919705256)
scrambling.test_significance("animals", "forager/scrambled_means/animals_Semantic_Similarity.csv", "Mean Semantic_Similarity", 0.43188669017027626)

# #phonological similarity 
scrambling.test_significance("a", "forager/scrambled_means/a_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.25896070346527433)
scrambling.test_significance("f", "forager/scrambled_means/f_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.2865961763502669)
scrambling.test_significance("s", "forager/scrambled_means/s_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.2933515691035733)
scrambling.test_significance("animals", "forager/scrambled_means/animals_Phonological_Similarity.csv", "Mean Phonological_Similarity", 0.2173854922778406)

# 8) see if there's a difference between the lexical results of the groups (G1: Young Adult, G2: Older Adult, G3: MCI Patient): 
groups.groups("a")
groups.groups("f")
groups.groups("s")