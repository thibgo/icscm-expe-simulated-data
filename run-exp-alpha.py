import os
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from icscm import InvariantCausalSCM
from icpscm import InvariantCausalPredictionSetCoveringMachine
from icpdt import InvariantCausalPredictionDecisionTree

#                                E=0  E=1
probas_table_for_a1 = np.array([[0.9, 0.5],  # Xa1=0
                                [0.1, 0.5]]) # Xa1=1

#                                E=0  E=1
probas_table_for_a2 = np.array([[0.5, 0.7],  # Xa2=0
                                [0.5, 0.3]]) # Xa2=1

#                                E=0  E=1
probas_table_for_a3 = np.array([[0.6, 0.4],  # Xa=0
                                [0.4, 0.6]]) # Xa=1

#                                E=0  E=1
probas_table_for_a4 = np.array([[0.4, 0.1],  # Xa=0
                                [0.6, 0.9]]) # Xa=1

#                                E=0  E=1
probas_table_for_a5 = np.array([[0.1, 0.4],  # Xa=0
                                [0.9, 0.6]]) # Xa=1

proba_table_dict = {'Xa1': probas_table_for_a1, 'Xa2': probas_table_for_a2, 'Xa3': probas_table_for_a3, 'Xa4': probas_table_for_a4, 'Xa5': probas_table_for_a5}


def compute_y(causal_variables_list, noise_on_y, structure_type):
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

def compute_dataset(structure, n_samples_per_env, n_random_variables, noise_on_y, noise_on_Xc, random_seed=11):
    random.seed(random_seed)
    data = []
    if structure.startswith('conjunction'):
        structure_type = 'conjunction'
    elif structure.startswith('tree'):
        structure_type = 'tree'
    else:
        raise ValueError('unknown structure', structure)
    n_causal_vars = int(structure[-1])
    for e in [0,1]:
        for i in range(n_samples_per_env):
            causal_variables_list, causal_variables_list_names = [], []
            for j in range(1, n_causal_vars + 1):
                name = 'Xa' + str(j)
                a_rand = random.random()
                proba_table = proba_table_dict[name]
                a = 0 if a_rand < proba_table[0,e] else 1
                causal_variables_list.append(a)
                causal_variables_list_names.append(name)
            y = compute_y(causal_variables_list, noise_on_y, structure_type)
            Xc = compute_Xc_v1(y, e, noise_on_Xc)
            random_variables_list, random_variables_list_names = [], []
            for j in range(n_random_variables):
                b0_rand = random.random()
                b0 = 0 if b0_rand < 0.5 else 1
                random_variables_list.append(b0)
                random_variables_list_names.append('Xb' + str(j))
            data.append([e, y] + random_variables_list + causal_variables_list + [ Xc])
    df1 = pd.DataFrame(data, columns=['E', 'Y'] + random_variables_list_names + causal_variables_list_names + ['Xc'])
    return df1

def init_model(algo, alpha_threshold):
    if algo == 'ICSCM':
        model = InvariantCausalSCM(threshold=alpha_threshold, random_state=11, pruning=True)
    elif algo == 'ICSCMnopruning':
        model = InvariantCausalSCM(threshold=alpha_threshold, random_state=11, pruning=False)
    elif algo == 'ICP+SCM':
        model = InvariantCausalPredictionSetCoveringMachine(threshold=alpha_threshold, random_state=11)
    elif algo == 'ICP+DT':
        model = InvariantCausalPredictionDecisionTree(threshold=alpha_threshold, random_state=11)
    else:
        raise ValueError('unknown algo', algo)
    return model

def compute_features_usage_df(algo, data_df, repetition, alpha_threshold, results_dir='.', structure=None):
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
    model = init_model(algo, alpha_threshold)
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
    if algo in ['SCM', 'ICSCM', 'ICSCMnopruning']:
        features_used = [0]*len(features_names)
        if hasattr(model, 'rule_importances'):
            for i in range(len(model.rule_importances)):
                feat_name = features_names[model.model_.rules[i].feature_idx]
                if model.rule_importances[i] > 0:
                    features_used[model.model_.rules[i].feature_idx] = model.rule_importances[i]              
    elif algo in ['DT', 'ICP+DT', 'ICP+DT2', 'ICP+SCM']:
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
    perf_df['alpha_threshold'] = [alpha_threshold]*perf_df.shape[0]
    # saving perf_df :
    alpha_str = str(alpha_threshold).replace('.', '')
    save_perf_path = os.path.join(results_dir, f'perf_df_algo_{algo}_nsamplesperenv_{n_samples_per_env}_nrandomvar_{n_random_var}_structure_{structure}_alphathreshold_{alpha_threshold}_repetition_{repetition}')
    perf_df.to_csv(save_perf_path, index=False)

random.seed(7)

algos_to_run = []
algos_to_run.append('ICSCM')
#algos_to_run.append('ICP+DT')

noise_on_y = 0.05
noise_on_Xc = 0.05

perf_df_list = []
n_samples_per_env_list = [10000]
random_vars_list = [1,2,3,4,5,6,7]
structure_list = ['conjunction2']

alpha_values_list = [0.001,
                     0.002,
                     0.005,
                     0.01,
                     0.025,
                     0.05,
                     0.10,
                     0.20,
                     0.50,
                     0.75,
                     1.0,
                     ]
                  
repetitions_range = list(range(100))

df_results_list = []
results_dir = 'results-alpha'

#list files in directory:
for file in os.listdir(results_dir):
    df_loc = pd.read_csv(os.path.join(results_dir, file))
    df_results_list.append(df_loc)
if len(df_results_list) > 0:
    big_perf_df = pd.concat(df_results_list)
else:
    big_perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type', 'split', 'n_var', 'n_samples', 'structure', 'alpha_threshold'])

for n_samples_per_env in n_samples_per_env_list:
    big_perf_df_n_samples = big_perf_df[big_perf_df['n_samples'] == n_samples_per_env]
    for n_random_var in random_vars_list:
        big_perf_df_n_samples_n_random_var = big_perf_df_n_samples[big_perf_df_n_samples['n_var'] == n_random_var]
        for structure in structure_list:
            big_perf_df_n_samples_n_random_var_structure = big_perf_df_n_samples_n_random_var[big_perf_df_n_samples_n_random_var['structure'] == structure]
            for algo in algos_to_run:
                big_perf_df_n_samples_n_random_var_structure_algo = big_perf_df_n_samples_n_random_var_structure[big_perf_df_n_samples_n_random_var_structure['algo'] == algo]
                for alpha_threshold in alpha_values_list:
                    big_perf_df_n_samples_n_random_var_structure_algo_alpha = big_perf_df_n_samples_n_random_var_structure_algo[big_perf_df_n_samples_n_random_var_structure_algo['alpha_threshold'] == alpha_threshold]
                    print('big_perf_df_n_samples_n_random_var_structure_algo_alpha.shape', big_perf_df_n_samples_n_random_var_structure_algo_alpha.shape)
                    print('algo=', algo)
                    print('n_random_var=', n_random_var)
                    print('n_samples_per_env=', n_samples_per_env)
                    print('structure=', structure)
                    print('alpha=', alpha_threshold)
                    done_repetition_range = big_perf_df_n_samples_n_random_var_structure_algo_alpha['split'].unique()
                    print('already done repetitions_range=', done_repetition_range)
                    print('expected repetitions_range=', repetitions_range)
                    to_be_done_repetition_range = [r for r in repetitions_range if r not in done_repetition_range]
                    print('to_be_done_repetition_range=', to_be_done_repetition_range)
                    if len(to_be_done_repetition_range) > 0:
                        exec_time_1 = datetime.now()
                        generated_df_dict = {}
                        for repetition in to_be_done_repetition_range:
                            data_df = compute_dataset(structure=structure, n_samples_per_env=n_samples_per_env, n_random_variables=n_random_var, noise_on_y=noise_on_y, noise_on_Xc=noise_on_Xc, random_seed=repetition)
                            generated_df_dict[repetition] = data_df
                        Parallel(n_jobs=5, verbose=5)(delayed(compute_features_usage_df)(algo, generated_df_dict[repetition], repetition=repetition, alpha_threshold=alpha_threshold, results_dir=results_dir, structure=structure) for repetition in to_be_done_repetition_range)
                        exec_time_2 = datetime.now()
                        print('execution time=', (exec_time_2 - exec_time_1).total_seconds(), 'seconds')
                        print(exec_time_2 - exec_time_1)
