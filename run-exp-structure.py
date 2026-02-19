# *********************** DEFINE DATA ***********************
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from pyscm import SetCoveringMachineClassifier
from sklearn.tree import DecisionTreeClassifier

from icscm import InvariantCausalSCM
from icpscm import InvariantCausalPredictionSetCoveringMachine
from icpdt import InvariantCausalPredictionDecisionTree

random.seed(7)

def compute_y(causal_variables_list, noise_on_y, structure_type):
    n_values = len(set(causal_variables_list))
    binarized_causal_variables_list = [int(a > 0.5) for a in causal_variables_list]
    #print('causal_variables_list          ', causal_variables_list)
    #print('binarized_causal_variables_list', binarized_causal_variables_list)
    causal_variables_list = binarized_causal_variables_list
    if structure_type == 'conjunction':
        y_theory = 1
        for a in causal_variables_list:
            y_theory *= a
    elif structure_type == 'tree':
        if len(causal_variables_list) > 3:
            first_conj = 1
            for a in causal_variables_list[:3]:
                first_conj *= a
        else:
            first_conj = 1
        if first_conj == 0:
            y_theory = 0
        else:
            after_separation_node = causal_variables_list[-3]
            if after_separation_node == 0:
                y_theory = causal_variables_list[-2]
            else:
                y_theory = causal_variables_list[-1]
    elif structure_type == 'treecomplete':
        outcome_first_node = causal_variables_list[0]
        y_theory = outcome_first_node
        if len(causal_variables_list) > 1:
            outcome_second_node = causal_variables_list[1+outcome_first_node]
            y_theory = outcome_second_node
        if len(causal_variables_list) > 3:
            outcome_third_node = causal_variables_list[3+outcome_first_node*2+outcome_second_node]
            y_theory = outcome_third_node
        if len(causal_variables_list) > 7:
            outcome_fourth_node = causal_variables_list[5+outcome_first_node*2+outcome_second_node]
            y_theory = outcome_fourth_node
    elif structure_type == 'disjunction':
        y_theory = max(causal_variables_list)
    #y_theory = int((a1 == 1) or (a2 == 1)) #disjunction
    r_value = random.random()
    if (r_value < noise_on_y):
        random_y = random.random()
        y = int(random_y < 0.5)
    else:
        y = y_theory
    return y

def compute_Xc_v1(y, e, noise_on_Xc):
    Xc_theory = int(y)
    r_value = random.random()
    if (r_value < noise_on_Xc):
        if e == 0:
            Xc = 1
        else:
            Xc = 0
    else:
        Xc = Xc_theory
    return Xc

def compute_Xc_v2(y, e, noise_on_Xc):
    Xc_theory = int(y)
    r_value = random.random()
    if (r_value < noise_on_Xc[e]):
        random_Xc = random.random()
        Xc = int(random_Xc < 0.5)
    else:
        Xc = Xc_theory
    return Xc

def compute_dataset(structure, n_samples_per_env, n_random_variables, noise_on_y, noise_on_Xc, gap_E0_E1, proportion_of_ones, random_seed=11):
    random.seed(random_seed)
    data = []
    id_first_int = 0
    char_are_str = True
    while char_are_str:
        try:
            int(structure[id_first_int])
            char_are_str = False
        except:
            id_first_int += 1
    if id_first_int == 0:
        raise ValueError(f'structure {structure} should not start with a number')
    structure_type = structure[:id_first_int]
    variables_type = 'binary' # default value
    if structure_type.endswith('continuous'):
        variables_type = 'continuous'
        structure_type = structure_type[:-len('continuous')]
    elif structure_type.endswith('binary'):
        variables_type = 'binary'
        structure_type = structure_type[:-len('binary')]
    n_causal_vars = int(structure[id_first_int:])
    #print(f'{structure} is understood as a {structure_type} of size {n_causal_vars}')
    for e in [0,1]:
        proba_of_a_being_one = proportion_of_ones - e * gap_E0_E1
        for i in range(n_samples_per_env):
            causal_variables_list, causal_variables_list_names = [], []
            for j in range(1, n_causal_vars + 1):
                name = 'Xa' + str(j)
                a_rand = random.random()
                if variables_type == 'continuous':
                    another_random = random.random()
                    if a_rand < proba_of_a_being_one:
                        a = 0.5 + another_random * 0.5
                    else:
                        a = another_random * 0.5
                    a = round(a, 2)
                else:
                    a = 1 if a_rand < proba_of_a_being_one else 0
                causal_variables_list.append(a)
                causal_variables_list_names.append(name)
            y = compute_y(causal_variables_list, noise_on_y, structure_type)
            Xc = compute_Xc_v1(y, e, noise_on_Xc)
            random_variables_list, random_variables_list_names = [], []
            for j in range(n_random_variables):
                b0_rand = random.random()
                if variables_type == 'continuous':
                    b0 = round(b0_rand, 2)
                elif variables_type == 'binary':
                    b0 = 0 if b0_rand < 0.5 else 1
                random_variables_list.append(b0)
                random_variables_list_names.append('Xb' + str(j))
            data.append([e, y] + random_variables_list + causal_variables_list + [Xc])
    df1 = pd.DataFrame(data, columns=['E', 'Y'] + random_variables_list_names + causal_variables_list_names + ['Xc'])
    return df1

def init_model(algo):
    if algo == 'SCM':
        model = SetCoveringMachineClassifier(random_state=11)
    elif algo == 'DT':
        model = DecisionTreeClassifier(random_state=11)
    elif algo == 'ICSCM':
        model = InvariantCausalSCM(threshold=0.05, pruning=True, random_state=11)
    elif algo == 'ICSCMconjunction':
        model = InvariantCausalSCM(model_type='conjunction', threshold=0.05, pruning=True, random_state=11)
    elif algo == 'ICSCMdisjunction':
        model = InvariantCausalSCM(model_type='disjunction', threshold=0.05, pruning=True, random_state=11)
    elif algo == 'ICP+SCM':
        model = InvariantCausalPredictionSetCoveringMachine(threshold=0.05, random_state=11)
    elif algo == 'ICP+DT':
        model = InvariantCausalPredictionDecisionTree(threshold=0.05, random_state=11)
    else:
        raise ValueError('unknown algo', algo)
    return model


param_grids = {
    'ICSCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction', 'disjunction'], 'stopping_method': ['independance_y_e']},
    'ICSCMconjunction': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['independance_y_e']},
    'ICSCMdisjunction': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['disjunction'], 'stopping_method': ['independance_y_e']},
    'ICSCMnopruning': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction', 'disjunction'], 'stopping_method': ['independance_y_e']},
    'SCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction']}, 
    'DT':
        {
            'min_samples_split' : [2, 0.01, 0.05, 0.1, 0.3],
            'max_depth' : [1, 2, 3, 4, 5, 10],
        },
}


def compute_features_usage_df(algo, data_df, repetition, do_grid_search=False, param_grids=None, results_dir='.', structure=None, gap_E0_E1=None, proportion_of_ones=None):
    perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type', 'split'])
    row_i = 0
    print('  repetition', repetition)
    df2 = data_df.copy()
    y = df2['Y'].values
    del df2['Y']
    X = df2.values
    features_names = list(df2)
    true_causal_features = [f for f in features_names if f.startswith('Xa')]
    print('features_names', features_names)
    print('true_causal_features', true_causal_features)
    true_causal_features_vector = [int(f in true_causal_features) for f in features_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11+repetition)
    print(algo)
    model = init_model(algo)
    # grid search for best parameters
    if do_grid_search and (algo in param_grids):
        hyperparameters = param_grids[algo]
        grid = GridSearchCV(
            estimator=model,
            param_grid=hyperparameters,
            verbose=1,
            n_jobs=1,
        )
        grid_result = grid.fit(X_train, y_train)
        tuned_hyperparameters = grid_result.best_params_
        print('tuned_hyperparameters', tuned_hyperparameters)
        model.set_params(**tuned_hyperparameters)  # set best params
        model.set_params(random_state=11)  # set random state
    # time 1 :
    t1 = datetime.now()
    model.fit(X_train, y_train)
    # time after fited :
    t2 = datetime.now()
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    perf_df.loc[row_i] = [algo, accuracy_score(y_test, test_pred), 'accuracy', 'test', repetition]
    row_i += 1
    perf_df.loc[row_i] = [algo, accuracy_score(y_train, train_pred), 'accuracy', 'train', repetition]
    row_i += 1
    if algo in ['SCM', 'ICSCM', 'ICSCMnopruning', 'ICSCMconjunction', 'ICSCMdisjunction']:
        features_used = [0]*len(features_names)
        if hasattr(model, 'rule_importances'):
            for i in range(len(model.rule_importances)):
                feat_name = features_names[model.model_.rules[i].feature_idx]
                if model.rule_importances[i] > 0:
                    features_used[model.model_.rules[i].feature_idx] = model.rule_importances[i]              
    elif algo in ['DT', 'ICP+DT', 'ICP+DT2', 'ICP+SCM', 'UI+DT', 'UI+SCM', 'UIR+DT', 'UIR+SCM']:
        features_used = model.feature_importances_
    else:
        raise Exception('algo not implemented')
    features_used_binary = [1 if f > 0 else 0 for f in features_used]
    causal_score = int(features_used_binary == true_causal_features_vector)
    print('features_used_binary       ', features_used_binary)
    print('true_causal_features_vector', true_causal_features_vector)
    if causal_score == 0 and hasattr(model, 'stream'):
        stream = model.stream
        print('log_stream.getvalue(): ', stream.getvalue())
    perf_df.loc[row_i] = [algo, causal_score, '01 loss', 'causal', repetition]
    row_i += 1
    perf_df.loc[row_i] = [algo, (t2 - t1).total_seconds(), 't2-t1', 'fit time', repetition]
    row_i += 1
    n_random_var = sum([f.startswith('Xb') for f in features_names])
    print('n_random_var', n_random_var)
    perf_df['n_var'] = [n_random_var]*perf_df.shape[0]
    n_samples_per_env = df2[df2['E'] == 0].shape[0]
    perf_df['n_samples'] = [n_samples_per_env]*perf_df.shape[0]
    idx_features_used_binary_code = sum([2**i*f for i, f in enumerate(features_used_binary)])
    perf_df['idx_features_used_binary_code'] = [idx_features_used_binary_code]*perf_df.shape[0]
    print(idx_features_used_binary_code)
    perf_df['structure'] = [structure]*perf_df.shape[0]
    perf_df['gap_E0_E1'] = [gap_E0_E1]*perf_df.shape[0]
    perf_df['proportion_of_ones'] = [proportion_of_ones]*perf_df.shape[0]
    # saving perf_df
    gap_E0_E1_str = str(gap_E0_E1).replace('.', '')
    proportion_of_ones_str = str(proportion_of_ones).replace('.', '')
    print('structure', structure)
    save_perf_path = os.path.join(results_dir, f'perf_df_algo_{algo}_nsamplesperenv_{n_samples_per_env}_nrandomvar_{n_random_var}_structure_{structure}_repetition_{repetition}_gapE0E1_{gap_E0_E1_str}_proportionofones_{proportion_of_ones_str}.csv')
    perf_df.to_csv(save_perf_path, index=False)

algos_to_run = []
algos_to_run.append('SCM')
algos_to_run.append('DT')
algos_to_run.append('ICSCM')
algos_to_run.append('ICP+DT')

noise_on_y = 0.05
noise_on_Xc = 0.05

perf_df_list = []
n_samples_per_env_list = [10000]
random_vars_list = [3]
structure_list = [
                  'conjunction1',
                  'conjunction2', 
                  'conjunction3',
                  'conjunction4',
                  'disjunction1',
                  'disjunction2', 
                  'disjunction3',
                  'disjunction4',
                  'treecomplete1',
                  'treecomplete3', 
                  'treecomplete7',
                  ]

gap_E0_E1_dict = {
                  'conjunction1': [0.25],
                  'conjunction2': [0.25],
                  'conjunction3': [0.25],
                  'conjunction4': [0.25],
                  'disjunction1': [0.25],
                  'disjunction2': [0.25],
                  'disjunction3': [0.25],
                  'disjunction4': [0.25],
                  'treecomplete1': [0.25],
                  'treecomplete3': [0.25],
                  'treecomplete7': [0.25],
                  'conjunctioncontinuous1': [0.25],
                  'conjunctioncontinuous2': [0.25], 
                  'conjunctioncontinuous3': [0.25],
                  'disjunctioncontinuous1': [0.25],
                  'disjunctioncontinuous2': [0.25], 
                  'disjunctioncontinuous3': [0.25],
                  'treecompletecontinuous1': [0.25],
                  'treecompletecontinuous3': [0.25], 
                  'treecompletecontinuous7': [0.25],
                }

proportion_of_ones_dict = {
                  'conjunction1': [0.9],
                  'conjunction2': [0.9],
                  'conjunction3': [0.9],
                  'conjunction4': [0.9],
                  'disjunction1': [0.3],
                  'disjunction2': [0.3],
                  'disjunction3': [0.3],
                  'disjunction4': [0.3],
                  'treecomplete1': [0.6],
                  'treecomplete3': [0.6],
                  'treecomplete7': [0.6],
                  'conjunctioncontinuous1': [0.9],
                  'conjunctioncontinuous2': [0.9], 
                  'conjunctioncontinuous3': [0.9],
                  'disjunctioncontinuous1': [0.3],
                  'disjunctioncontinuous2': [0.3], 
                  'disjunctioncontinuous3': [0.3],
                  'treecompletecontinuous1': [0.6],
                  'treecompletecontinuous3': [0.6], 
                  'treecompletecontinuous7': [0.6],
                }

repetitions_range = list(range(10)) # 10 repetitions

df_results_list = []
results_dir = 'results-structure'

#list files in directory:
for file in os.listdir(results_dir):
    df_loc = pd.read_csv(os.path.join(results_dir, file))
    df_results_list.append(df_loc)
if len(df_results_list) > 0:
    big_perf_df = pd.concat(df_results_list)
else:
    big_perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type', 'split', 'n_var', 'n_samples', 'structure', 'gap_E0_E1', 'proportion_of_ones'])

for n_samples_per_env in n_samples_per_env_list:
    big_perf_df_n_samples = big_perf_df[big_perf_df['n_samples'] == n_samples_per_env]
    for n_random_var in random_vars_list:
        big_perf_df_n_samples_n_random_var = big_perf_df_n_samples[big_perf_df_n_samples['n_var'] == n_random_var]
        for structure in structure_list:
            big_perf_df_n_samples_n_random_var_structure = big_perf_df_n_samples_n_random_var[big_perf_df_n_samples_n_random_var['structure'] == structure]
            gap_E0_E1_list = gap_E0_E1_dict[structure]
            proportion_of_ones_list = proportion_of_ones_dict[structure]
            print('structure', structure)
            print('gap_E0_E1_list', gap_E0_E1_list)
            print('proportion_of_ones_list', proportion_of_ones_list)
            if 'continuous' in structure:
                do_grid_search = False
            else:
                do_grid_search = True
            for algo in algos_to_run:
                big_perf_df_n_samples_n_random_var_structure_algo = big_perf_df_n_samples_n_random_var_structure[big_perf_df_n_samples_n_random_var_structure['algo'] == algo]
                for gap_E0_E1 in gap_E0_E1_list:
                    big_perf_df_n_samples_n_random_var_structure_algo_gap_E0_E1 = big_perf_df_n_samples_n_random_var_structure_algo[big_perf_df_n_samples_n_random_var_structure_algo['gap_E0_E1'] == gap_E0_E1]
                    for proportion_of_ones in proportion_of_ones_list:
                        big_perf_df_n_samples_n_random_var_structure_algo_gap_E0_E1_proportion_of_ones = big_perf_df_n_samples_n_random_var_structure_algo_gap_E0_E1[big_perf_df_n_samples_n_random_var_structure_algo_gap_E0_E1['proportion_of_ones'] == proportion_of_ones]
                        print('algo=', algo)
                        print('n_random_var=', n_random_var)
                        print('n_samples_per_env=', n_samples_per_env)
                        print('structure=', structure)
                        print('gap_E0_E1=', gap_E0_E1)
                        print('proportion_of_ones=', proportion_of_ones)
                        done_repetition_range = big_perf_df_n_samples_n_random_var_structure_algo['split'].unique()
                        print('already done repetitions_range=', done_repetition_range)
                        print('expected repetitions_range=', repetitions_range)
                        to_be_done_repetition_range = [r for r in repetitions_range if r not in done_repetition_range]
                        print('to_be_done_repetition_range=', to_be_done_repetition_range)
                        if len(to_be_done_repetition_range) > 0:
                            exec_time_1 = datetime.now()
                            generated_df_dict = {}
                            for repetition in to_be_done_repetition_range:
                                data_df = compute_dataset(structure=structure, n_samples_per_env=n_samples_per_env, n_random_variables=n_random_var, noise_on_y=noise_on_y, noise_on_Xc=noise_on_Xc, gap_E0_E1=gap_E0_E1, proportion_of_ones=proportion_of_ones, random_seed=repetition)
                                generated_df_dict[repetition] = data_df
                            Parallel(n_jobs=10, verbose=5)(delayed(compute_features_usage_df)(algo, generated_df_dict[repetition], repetition=repetition, do_grid_search=do_grid_search, param_grids=param_grids, results_dir=results_dir, structure=structure, gap_E0_E1=gap_E0_E1, proportion_of_ones=proportion_of_ones) for repetition in to_be_done_repetition_range)
                            exec_time_2 = datetime.now()
                            print('execution time=', (exec_time_2 - exec_time_1).total_seconds(), 'seconds')
                            print(exec_time_2 - exec_time_1)
