import argparse
from scipy.optimize import fmin
from forager.forager.foraging import forage
from forager.forager.switch import switch_delta, switch_multimodal, switch_simdrop, switch_norms_associative, switch_norms_categorical
from forager.forager.cues import create_history_variables
from forager.forager.utils import prepareData
import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize
import os, sys
from tqdm import tqdm
import warnings 
import zipfile

warnings.simplefilter('ignore')

"""
Workflow: 
1. Evaluate data
    a. "Prepare Data" 
        - takes path of fluency list ; 

2. Select use case (lexical, switches, models) via --pipeline
    a. Lexical: returns similarity & frequency values for each word in fluency list
    b. Switches: returns switch values for each word in fluency list + lexical values
    c. Models: returns model outputs for each word in fluency list + lexical values + switch values
"""
# Global Path Variables
fp = "/".join(sys.argv[0].split('/')[:-1])

if sys.argv[0] != 'run_foraging.py': # check the reference path
    # normspath = os.path.join(fp,'data/norms/animals_snafu_scheme_vocab.csv')
    similaritypath =  os.path.join(fp,'data/lexical_data/USE_semantic_matrix.csv')
    frequencypath =  os.path.join(fp,'data/lexical_data/USE_frequencies.csv')
    phonpath = os.path.join(fp,'data/lexical_data/USE_phonological_matrix.csv')
    vocabpath = os.path.join(fp,'data/lexical_data/vocab.csv')
else:
    # normspath = 'data/norms/animals_snafu_scheme_vocab.csv'
    similaritypath =  'data/lexical_data/USE_semantic_matrix.csv'
    frequencypath =  'data/lexical_data/USE_frequencies.csv'
    phonpath = 'data/lexical_data/USE_phonological_matrix.csv'
    vocabpath = 'data/lexical_data/vocab.csv'

# Global Variables
models = ['static','dynamic','pstatic','pdynamic','all']
switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta','all']

#Methods
def retrieve_data(path,fp):
    """
    1. Verify that data path exists

    """
    if os.path.exists(path) == False:
        ex_str = "Provided path to data \"{path}\" does not exist. Please specify a proper path".format(path=path)
        raise Exception(ex_str)
    data = prepareData(path,fp)
    return data

def get_lexical_data():
    # norms = pd.read_csv(normspath, encoding="unicode-escape")
    similarity_matrix = np.loadtxt(similaritypath,delimiter=',')
    frequency_list = np.array(pd.read_csv(frequencypath,header=None,encoding="unicode-escape")[1])
    phon_matrix = np.loadtxt(phonpath,delimiter=',')
    labels = pd.read_csv(frequencypath,header=None)[0].values.tolist()
    return norms, similarity_matrix, phon_matrix, frequency_list,labels

def calculate_model(model, history_vars, switch_names, switch_vecs):
    """
    1. Check if specified model is valid
    2. Return a set of model functions to pass
    """
    model_name = []
    model_results = []
    if model not in models:
        ex_str = "Specified model is invalid. Model must be one of the following: {models}".format(models=models)
        raise Exception(ex_str)
    if model == models[0] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()

        v = minimize(forage.model_static, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1])).x
        beta_df = float(v[0]) # Optimized weight for frequency cue
        beta_ds = float(v[1]) # Optimized weight for similarity cue
        
        nll, nll_vec = forage.model_static_report([beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1])
        model_name.append('forage_static')
        model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[1] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            r1 = np.random.rand()
            r2 = np.random.rand()

            v = minimize(forage.model_dynamic, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], switch_vec)).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            
            nll, nll_vec = forage.model_dynamic_report([beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1],switch_vec)
            model_name.append('forage_dynamic_' + switch_names[i])
            model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[2] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        v = minimize(forage.model_static_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4],history_vars[5])).x
        beta_df = float(v[0]) # Optimized weight for frequency cue
        beta_ds = float(v[1]) # Optimized weight for similarity cue
        beta_dp = float(v[2]) # Optimized weight for phonological cue

        nll, nll_vec = forage.model_static_phon_report([beta_df, beta_ds, beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5])
        model_name.append('forage_phonologicalstatic')
        model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    if model == models[3] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            # Global Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'global')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'global')
            model_name.append('forage_phonologicaldynamicglobal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
            # Local Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'local')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'local')
            model_name.append('forage_phonologicaldynamiclocal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
            # Switch Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'switch')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'switch')
            model_name.append('forage_phonologicaldynamicswitch_' + switch_names[i])

            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
    # Unoptimized Model
    model_name.append('forage_random_baseline')
    nll_baseline, nll_baseline_vec = forage.model_static_report(beta = [0,0], freql = history_vars[2], freqh = history_vars[3], siml = history_vars[0], simh = history_vars[1])
    model_results.append((0, 0, nll_baseline, nll_baseline_vec))
    return model_name, model_results

def calculate_switch(switch, fluency_list, semantic_similarity, phon_similarity, alpha = np.arange(0, 1.1, 0.1), rise = np.arange(0, 1.25, 0.25), fall = np.arange(0, 1.25, 0.25)):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required

    switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta','all']
    '''
    switch_names = []
    switch_vecs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(switch=switch_methods)
        raise Exception(ex_str)

    if switch == switch_methods[0] or switch == switch_methods[5]:
        switch_names.append(switch_methods[0])
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == switch_methods[1] or switch == switch_methods[5]:
        for i, a in enumerate(alpha):
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(fluency_list, semantic_similarity, phon_similarity, a))

    if switch == switch_methods[4] or switch == switch_methods[5]:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                switch_names.append("delta_rise={rise}_fall={fall}".format(rise=r,fall=f))
                switch_vecs.append(switch_delta(fluency_list, semantic_similarity, r, f))

    return switch_names, switch_vecs

def run_model(data, model_type, switch_type, similarity_matrix, phon_matrix, frequency_list, labels):
    # Get Lexical Data needed for executing methods
    forager_results = []
    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        print("\nRunning Model for Subject {subj}".format(subj=subj))
        import time
        start_time = time.time()
        # Get History Variables 
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        
        # Calculate Switch Vector(s)
        switch_names, switch_vecs = calculate_switch(switch_type, fl_list, history_vars[0],   history_vars[4])

        #Execute Individual Model(s) and get result(s)
        model_names, model_results = calculate_model(model_type,history_vars, switch_names, switch_vecs)

        #Create Model Output Results DataFrame
        for i, model in enumerate(model_names):
            model_dict = dict()
            model_dict['Subject'] = subj
            model_dict['Model'] = model
            model_dict['Beta_Frequency'] = model_results[i][0]
            model_dict['Beta_Semantic'] = model_results[i][1]
            # print(results[i])
            # sys.exit()
            if len(model_results[i]) == 4:
                model_dict['Beta_Phonological'] = None
                model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][2]
            if len(model_results[i]) == 5:
                model_dict['Beta_Phonological'] = model_results[i][2]
                model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][3]
            forager_results.append(model_dict)
    forager_results = pd.DataFrame(forager_results)
        
    return forager_results
    
def run_lexical(data, similarity_matrix, phon_matrix, frequency_list, labels):
    # Get Lexical Data needed for executing methods
    lexical_results = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        lexical_df = pd.DataFrame()
        lexical_df['Subject'] = len(fl_list) * [subj]
        lexical_df['Fluency_Item'] = fl_list
        lexical_df['Semantic_Similarity'] = history_vars[0]
        lexical_df['Frequency_Value'] = history_vars[2]
        lexical_df['Phonological_Similarity'] = history_vars[4]
        lexical_results.append(lexical_df)
    lexical_results = pd.concat(lexical_results,ignore_index=True)
    return lexical_results

def run_switches(data,switch_type, similarity_matrix, phon_matrix, frequency_list, labels):
    switch_results = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        switch_names, switch_vecs = calculate_switch(switch_type, fl_list, history_vars[0], history_vars[4])
    
        switch_df = []
        for j, switch in enumerate(switch_vecs):
            df = pd.DataFrame()
            df['Subject'] = len(switch) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch
            df['Switch_Method'] = switch_names[j]
            switch_df.append(df)
    
        switch_df = pd.concat(switch_df, ignore_index=True)
        switch_results.append(switch_df)
    switch_results = pd.concat(switch_results, ignore_index=True)
    return switch_results


def indiv_desc_stats(lexical_results, switch_results = None):
    metrics = lexical_results[['Subject', 'Semantic_Similarity', 'Frequency_Value', 'Phonological_Similarity']]
    metrics.replace(.0001, np.nan, inplace=True)
    grouped = metrics.groupby('Subject').agg(['mean', 'std'])
    grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
    grouped.reset_index(inplace=True)
    num_items = lexical_results.groupby('Subject')['Fluency_Item'].size()
    grouped['#_of_Items'] = num_items[grouped['Subject']].values
    # create column for each switch method per subject and get number of switches, mean cluster size, and sd of cluster size for each switch method
    if switch_results is not None:
        # count the number of unique values in the Switch_Method column of the switch_results DataFrame
        n_rows = len(switch_results['Switch_Method'].unique())
        new_df = pd.DataFrame(np.nan, index=np.arange(len(grouped) * (n_rows)), columns=grouped.columns)

        # Insert the original DataFrame into the new DataFrame but repeat the value in 'Subject' column n_rows-1 times
        new_df.iloc[(slice(None, None, n_rows)), :] = grouped
        new_df['Subject'] = new_df['Subject'].ffill()

        switch_methods = []
        num_switches_arr = []
        cluster_size_mean = []
        cluster_size_sd = []
        for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
            switch_method = sub[1]
            cluster_lengths = []
            num_switches = 0
            ct = 0
            for x in fl_list['Switch_Value'].values:
                ct += 1
                if x == 1:
                    num_switches += 1
                    cluster_lengths.append(ct)
                    ct = 0
            if ct != 0:
                cluster_lengths.append(ct)
            avg = sum(cluster_lengths) / len(cluster_lengths)
            sd = np.std(cluster_lengths)
            switch_methods.append(switch_method)
            num_switches_arr.append(num_switches)
            cluster_size_mean.append(avg)
            cluster_size_sd.append(sd)

        new_df['Switch_Method'] = switch_methods
        new_df['Number_of_Switches'] = num_switches_arr
        new_df['Cluster_Size_mean'] = cluster_size_mean
        new_df['Cluster_Size_std'] = cluster_size_sd
        grouped = new_df
        
    return grouped

def agg_desc_stats(switch_results, model_results=None):
    agg_df = pd.DataFrame()
    # get number of switches per subject for each switch method
    switches_per_method = {}
    for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
        method = sub[1]
        if method not in switches_per_method:
            switches_per_method[method] = []
        if 1 in fl_list['Switch_Value'].values:
            switches_per_method[method].append(fl_list['Switch_Value'].value_counts()[1])
        else: 
            switches_per_method[method].append(0)
    agg_df['Switch_Method'] = switches_per_method.keys()
    agg_df['Switches_per_Subj_mean'] = [np.average(switches_per_method[k]) for k in switches_per_method.keys()]
    agg_df['Switches_per_Subj_SD'] = [np.std(switches_per_method[k]) for k in switches_per_method.keys()]
    
    if model_results is not None:
        betas = model_results.drop(columns=['Subject', 'Negative_Log_Likelihood_Optimized'])
        betas.drop(betas[betas['Model'] == 'forage_random_baseline'].index, inplace=True)
        grouped = betas.groupby('Model').agg(['mean', 'std'])
        grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
        grouped.reset_index(inplace=True)

        # add a column to the grouped dataframe that contains the switch method used for each model
        grouped.loc[grouped['Model'].str.contains('static'), 'Model'] += ' none'
        # if the model name starts with 'forage_dynamic_', ''forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', or 'forage_phonologicaldynamicswitch_', replace the second underscore with a space
        switch_models = ['forage_dynamic_', 'forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', 'forage_phonologicaldynamicswitch_']
        for model in switch_models:
            # replace only the second underscore with a space
            grouped.loc[grouped['Model'].str.contains(model), 'Model'] = grouped.loc[grouped['Model'].str.contains(model), 'Model'].str.replace('_', ' ', 2)
            grouped.loc[grouped['Model'].str.contains("forage "), 'Model'] = grouped.loc[grouped['Model'].str.contains("forage "), 'Model'].str.replace(' ', '_', 1)
        
        # split the Model column on the space
        grouped[['Model', 'Switch_Method']] = grouped['Model'].str.rsplit(' ', n=1, expand=True)

        # merge the two dataframes on the Switch_Method column 
        agg_df = pd.merge(agg_df, grouped, how='outer', on='Switch_Method')


    return agg_df



def run_letter(letter):
    """
    Processes and analyze participant response data for a specified letter.
    Runs:
    -run_switches()
    -run_lexical()
    -indiv_desc_stats()
    -run_model() for static, dynamic simdrop, dynamic delta, pstatic, pdynamic simdrop, pdynamic multimodal
    """
    #loading data 
    similarity_matrix = np.loadtxt(f"forager/data/lexical_data/letter_{letter}/USE_semantic_matrix_{letter}.csv",delimiter=',')
    frequency_list = np.array(pd.read_csv(f"forager/data/lexical_data/letter_{letter}/USE_frequencies.csv",header=None,encoding="unicode-escape")[1])
    labels = pd.read_csv(f"forager/data/lexical_data/letter_{letter}/USE_frequencies.csv",header=None)[0].values.tolist()
    phon_matrix = np.loadtxt(f"forager/data/lexical_data/letter_{letter}/USE_phon_matrix_{letter}.csv",delimiter=',')

    participant_df = pd.read_csv(f"forager/data/fluency_lists/transformed_data/{letter}_transformed.csv")
    data = []
    labels = [label.lower().strip() for label in labels]
    
    data = []
    for subj, group in participant_df.groupby('Participant'):
        fl_list = group['Response'].str.lower().str.strip().tolist()
        fl_list = [item for item in fl_list if item in labels]
        data.append((subj, fl_list))


    #acquiring switch and lexical results
    switch_results = run_switches(data,"all", similarity_matrix, phon_matrix, frequency_list, labels)
    lexical_results = run_lexical(data, similarity_matrix, phon_matrix, frequency_list, labels)
    grouped = indiv_desc_stats(lexical_results, switch_results)

    #saving into csvs 
    switch_results.to_csv(f"output/letter_{letter}/switch_results_{letter}.csv", index=False)
    lexical_results.to_csv(f"output/letter_{letter}/lexical_results_{letter}.csv", index=False)
    grouped.to_csv(f"output/letter_{letter}/indiv_stats_results_{letter}.csv", index=False)
  
    #acquiring model results 
    static_forager_results = run_model(data, "static", "simdrop", similarity_matrix, phon_matrix, frequency_list, labels)

    dynamic_simdrop_forager_results = run_model(data, "dynamic", "simdrop", similarity_matrix, phon_matrix, frequency_list, labels)
    dynamic_delta_forager_results = run_model(data, "dynamic", "delta", similarity_matrix, phon_matrix, frequency_list, labels)

    pstatic_forager_results = run_model(data, "pstatic", "simdrop", similarity_matrix, phon_matrix, frequency_list, labels)
    pdynamic_simdrop_forager_results = run_model(data, "pdynamic", "simdrop", similarity_matrix, phon_matrix, frequency_list, labels)
    pdynamic_multimodal_forager_results = run_model(data, "pdynamic", "multimodal", similarity_matrix, phon_matrix, frequency_list, labels)
    
    #saving into csvs
    static_forager_results.to_csv(f"output/letter_{letter}/static_forager_{letter}.csv", index=False)
    dynamic_simdrop_forager_results.to_csv(f"output/letter_{letter}/dynamic_simdrop_forager_{letter}.csv", index=False)
    dynamic_delta_forager_results.to_csv(f"output/letter_{letter}/dynamic_delta_forager_{letter}.csv", index=False)

    pstatic_forager_results.to_csv(f"output/letter_{letter}/pstatic_forager_{letter}.csv", index=False)
    pdynamic_simdrop_forager_results.to_csv(f"output/letter_{letter}/pdynamic_simdrop_forager_{letter}.csv", index=False)
    pdynamic_multimodal_forager_results.to_csv(f"output/letter_{letter}/pdynamic_multimodal_forager_{letter}.csv", index=False)

