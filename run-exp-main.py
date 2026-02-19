# *********************** DEFINE DATA ***********************
import os
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

from pyscm import SetCoveringMachineClassifier
from sklearn.tree import DecisionTreeClassifier

from icscm import InvariantCausalSCM
from icpdt import InvariantCausalPredictionDecisionTree

N_JOBS = 1
random.seed(7) # 7 in the paper
result_dir = 'results-main'

noise_on_y = 0.05
noise_on_Xc = 0.05

perf_df_list = []
n_samples_per_env_list = [10000]
algos = [
        'SCM',
        'DT',
        #'ICP+DT',
        'ICSCM',
        'ICSCMnopruning',
        ]
random_vars_list = [1,2,3,4,5,6,7]
repetitions_range = list(range(100))

#                                E=0  E=1
probas_table_for_a1 = np.array([[0.9, 0.5],  # Xa1=0
                                [0.1, 0.5]]) # Xa1=1

#                                E=0  E=1
probas_table_for_a2 = np.array([[0.5, 0.7],  # Xa2=0
                                [0.5, 0.3]]) # Xa2=1


param_grids = {
    'ICSCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['independance_y_e']},
    'ICSCMnopruning': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['independance_y_e']},
    'SCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction']}, 
    'DT':
        {
            'min_samples_split' : [2, 0.01, 0.05, 0.1, 0.3],
            'max_depth' : [1, 2, 3, 4, 5, 10],
        },
}

def compute_y(a1, a2, noise_on_y):
    y_theory = int((a1 == 1) and (a2 == 1))
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

def compute_dataset(n_samples_per_env, n_random_variables, noise_on_y, noise_on_Xc, random_seed=11):
    random.seed(random_seed)
    data = []
    for e in [0,1]:
        for i in range(n_samples_per_env):
            a1_rand = random.random()
            a2_rand = random.random()
            a1 = 0 if a1_rand < probas_table_for_a1[0,e] else 1
            a2 = 0 if a2_rand < probas_table_for_a2[0,e] else 1
            y = compute_y(a1, a2, noise_on_y)
            Xc = compute_Xc_v1(y, e, noise_on_Xc)
            random_variables_list, random_variables_list_names = [], []
            for j in range(n_random_variables):
                b0_rand = random.random()
                b0 = 0 if b0_rand < 0.5 else 1
                random_variables_list.append(b0)
                random_variables_list_names.append('Xb' + str(j))
            data.append([e, y] + random_variables_list + [a1, a2, Xc])
    df1 = pd.DataFrame(data, columns=['E', 'Y'] + random_variables_list_names + ['Xa1', 'Xa2', 'Xc'])
    return df1

def init_model(algo):
    if algo == 'SCM':
        model = SetCoveringMachineClassifier(random_state=11)
    elif algo == 'DT':
        model = DecisionTreeClassifier(random_state=11)
    elif algo == 'ICSCM':
        model = InvariantCausalSCM(threshold=0.05, pruning=True, random_state=11)
    elif algo == 'ICSCMnopruning':
        model = InvariantCausalSCM(threshold=0.05, pruning=False, random_state=11)
    elif algo == 'ICP+SCM':
        model = InvariantCausalPredictionSetCoveringMachine(threshold=0.05, random_state=11)
    elif algo == 'ICP+DT':
        model = InvariantCausalPredictionDecisionTree(threshold=0.05, random_state=11)
    else:
        raise ValueError('unknown algo', algo)
    return model

def compute_features_usage_df(algo, data_df, repetition, do_grid_search=False, param_grids=None, result_dir='results-runtime'):
    perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type', 'split'])
    row_i = 0
    print('  repetition', repetition)
    df2 = data_df.copy()
    y = df2['Y'].values
    del df2['Y']
    X = df2.values
    features_names = list(df2)
    true_causal_features = ['Xa1', 'Xa2']
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
    # activate logging of following line :
    model.fit(X_train, y_train)
    # time after fited :
    t2 = datetime.now()
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    perf_df.loc[row_i] = [algo, accuracy_score(y_test, test_pred), 'accuracy', 'test', repetition]
    row_i += 1
    perf_df.loc[row_i] = [algo, accuracy_score(y_train, train_pred), 'accuracy', 'train', repetition]
    row_i += 1
    # add precision and recall for test and train:
    if algo in ['SCM', 'ICSCM', 'ICSCMnopruning']:
        features_used = [0]*len(features_names)
        if hasattr(model, 'rule_importances'):
            for i in range(len(model.rule_importances)):
                feat_name = features_names[model.model_.rules[i].feature_idx]
                if model.rule_importances[i] > 0:
                    features_used[model.model_.rules[i].feature_idx] = model.rule_importances[i]              
    elif algo in ['DT', 'ICP+DT', 'ICP+SCM']:
        features_used = model.feature_importances_
    else:
        raise Exception('algo not implemented')
    features_used_binary = [1 if f > 0 else 0 for f in features_used]
    #features_usage_df[algo] = features_used_binary
    causal_score = int(features_used_binary == true_causal_features_vector)
    causal_score_precision = precision_score(true_causal_features_vector, features_used_binary)
    causal_score_recall = recall_score(true_causal_features_vector, features_used_binary)
    print('features_used_binary       ', features_used_binary)
    print('true_causal_features_vector', true_causal_features_vector)
    if causal_score == 0 and hasattr(model, 'stream'):
        stream = model.stream
        print('log_stream.getvalue(): ', stream.getvalue())
    perf_df.loc[row_i] = [algo, causal_score, '01 loss', 'causal', repetition]
    row_i += 1
    perf_df.loc[row_i] = [algo, causal_score_precision, 'precision', 'causal', repetition]
    row_i += 1
    perf_df.loc[row_i] = [algo, causal_score_recall, 'recall', 'causal', repetition]
    row_i += 1
    perf_df.loc[row_i] = [algo, (t2 - t1).total_seconds(), 't2-t1', 'fit time', repetition]
    row_i += 1
    n_random_var = sum([f.startswith('Xb') for f in features_names])
    print('n_random_var', n_random_var)
    perf_df['n_var'] = [n_random_var]*perf_df.shape[0]
    n_samples_per_env = df2[df2['E'] == 0].shape[0]
    perf_df['n_samples'] = [n_samples_per_env]*perf_df.shape[0]
    save_perf_path = os.path.join(result_dir, f'perf_df_{algo}_{n_samples_per_env}_{n_random_var}_{repetition}')
    perf_df.to_csv(save_perf_path, index=False)

df_results_list = []
#list files in directory:
for file in os.listdir(result_dir):
    df_loc = pd.read_csv(os.path.join(result_dir, file))
    df_results_list.append(df_loc)
if len(df_results_list) > 0:
    big_perf_df = pd.concat(df_results_list)
else:
    big_perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type', 'split', 'n_var', 'n_samples'])

for n_samples_per_env in n_samples_per_env_list:
    big_perf_df_n_samples = big_perf_df[big_perf_df['n_samples'] == n_samples_per_env]
    for n_random_var in random_vars_list:
        big_perf_df_n_samples_n_random_var = big_perf_df_n_samples[big_perf_df_n_samples['n_var'] == n_random_var]
        for algo in algos:
            big_perf_df_n_samples_n_random_var_algo = big_perf_df_n_samples_n_random_var[big_perf_df_n_samples_n_random_var['algo'] == algo]
            print('algo=', algo)
            print('n_random_var=', n_random_var)
            print('n_samples_per_env=', n_samples_per_env)
            done_repetition_range = big_perf_df_n_samples_n_random_var_algo['split'].unique()
            print('already done repetitions_range=', done_repetition_range)
            print('expected repetitions_range=', repetitions_range)
            to_be_done_repetition_range = [r for r in repetitions_range if r not in done_repetition_range]
            print('to_be_done_repetition_range=', to_be_done_repetition_range)
            if len(to_be_done_repetition_range) > 0:
                exec_time_1 = datetime.now()
                generated_df_dict = {}
                for repetition in to_be_done_repetition_range:
                    data_df = compute_dataset(n_samples_per_env=n_samples_per_env, n_random_variables=n_random_var, noise_on_y=noise_on_y, noise_on_Xc=noise_on_Xc, random_seed=repetition)
                    generated_df_dict[repetition] = data_df
                Parallel(n_jobs=N_JOBS, verbose=5)(delayed(compute_features_usage_df)(algo, generated_df_dict[repetition], repetition=repetition, do_grid_search=False, param_grids=param_grids, result_dir=result_dir) for repetition in to_be_done_repetition_range)
                exec_time_2 = datetime.now()
                print('execution time=', (exec_time_2 - exec_time_1).total_seconds(), 'seconds')
                print(exec_time_2 - exec_time_1)


