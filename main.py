# -*- coding: utf-8 -*-
"""
@author:        ZhengQiguang
@project:       rweChemScreener
@file name:     main
"""

import os
from itertools import combinations, chain
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import zscore
from scipy.stats import spearmanr
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, KFold, cross_val_score
import shap
import xgboost as xgb

from econml.dml import LinearDML
from tqdm import tqdm


xgb_model = None
best_params = {'max_depth': 5, 'learning_rate': 0.013779015433576913, 'subsample': 0.6224726191768079,
                   'colsample_bytree': 0.8935781847648362, 'reg_alpha': 9.608322494172442e-05,
                   'reg_lambda': 0.6, 'seed': 42, 'objective': 'reg:squarederror',
                   'tree_method': 'auto',
                   'eval_metric': 'rmse', 'device': 'cpu'}
avg_best_iter = 300

output_root = 'data'


def calculate_slope(sequence):
    last_ten = sequence[-10:]
    if len(last_ten) < 2:
        return 1
    x = np.arange(len(last_ten))
    y = np.array(last_ten)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope

def trans_new_M(m, M_size):
    def m_to_vector(m, M_size):
        from decimal import Decimal, InvalidOperation
        if M_size < 1:
            raise ValueError("M_size must be at least 1")

        sign = -1 if m < 0 else 1
        m = abs(m)

        try:
            d = Decimal(str(m)).normalize()
        except InvalidOperation:
            raise ValueError(f"Invalid number format: {m}")

        str_d = format(d, 'f') if d.as_tuple().exponent < 0 else str(d).replace('.', '')
        if d < 0:
            str_d = str_d[1:]

        if 'e' not in str(m).lower() and '.' not in str_d and 'e' not in str_d:
            str_d = format(d, 'f')

        if '.' in str_d:
            integer_part, fraction_part = str_d.split('.')
            digits = list(integer_part) + list(fraction_part)
        else:
            digits = list(str_d)

        required_length = M_size - 1
        processed_digits = digits[:required_length]

        if len(processed_digits) < required_length:
            remaining = required_length - len(processed_digits)
            processed_digits += [str(np.random.randint(0, 9)) for _ in range(remaining)]

        processed_digits = [int(d) for d in processed_digits]

        vector = [sign] + processed_digits
        return vector

    M = np.array([m_to_vector(mi, M_size) for mi in zscore(m)])

    return M


def simulate_mediation_casual(model, x_df, t, m_df, y, patience_max=10):
    sample_size = len(x_df)
    M = m_df.values

    pcaX = PCA(n_components=0.8)
    x_pca = pcaX.fit_transform(x_df)

    tx_df = x_df.copy()
    tx_df['t'] = t
    pcaTX = PCA(n_components=0.8)
    tx_pca = pcaTX.fit_transform(tx_df)

    z = np.random.normal(t, 0.5, size=sample_size)

    total_effect_model = LinearDML(model_t='linear', model_y='linear', random_state=42)
    total_effect_causal = total_effect_model.fit(Y=y, T=t, X=x_pca, W=x_pca)
    print(total_effect_causal.ate_inference(X=x_pca).mean_point)
    total_effect = total_effect_causal.ate_inference(X=x_pca).mean_point

    res_list = []
    gamma_list = []
    patience = patience_max
    iterations = 50
    for i in tqdm(range(0, iterations), disable=False):

        if np.corrcoef(z, y)[0, 1] < 0:
            z = z * (-1)

        if any(np.isnan(zscore(z))):
            z = z
        else:
            z = zscore(z)

        xt = np.concatenate([x_pca, t.reshape(-1, 1)], axis=1)
        lasso_reg_m = linear_model.LinearRegression()
        lasso_reg_m.fit(X=xt, y=z)
        alph = lasso_reg_m.coef_[-1]
        lasso_reg_m.score(X=xt, y=z)

        bet_causal_model = LinearDML(model_t='linear', model_y='linear', random_state=42)
        bet_inference = bet_causal_model.fit(Y=np.array(y), T=z, X=tx_pca, W=tx_pca).ate_inference(X=tx_pca)
        bet_causal = bet_inference.mean_point

        bet = bet_causal
        gam = total_effect - alph * bet

        print(f'iter{i} Alpha: {alph} Beta: {bet} Gamma: {gam}')
        res_list.append((i, alph, bet, gam))
        y_reg2 = linear_model.LinearRegression()

        y_reg2.fit(X=x_df.values, y=y.values - gam * t - bet * z)

        h = lasso_reg_m.predict(xt)
        e = y.values - gam * t - y_reg2.predict(X=x_df.values)
        d = (((bet * e) + h) / ((bet ** 2) + 1))

        gamma_list.append(gam)
        slope = calculate_slope(gamma_list)
        if np.abs(slope) < 1e-4 and i > 30:
            break

        xgb_model, best_params, avg_best_iter = model
        dtrain_all = xgb.DMatrix(M, label=d)
        final_model = xgb.train(
            best_params,
            dtrain_all,
            num_boost_round=avg_best_iter
        )

        z = final_model.predict(dtrain_all)

    return res_list, final_model


def run_rwechemscreener(x_df, t, m_df, y, pres_name, n_bootstraps=10):
    def jaccard_similarity(set_a, set_b):
        set_a = set(set_a)
        set_b = set(set_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union != 0 else 0

    N = len(y)
    M_names = m_df.columns

    effect_res_list = []

    res_list = []
    selected_importance_list = []
    importance_rate_list = []
    feature_values_all_list = []
    shap_values_all_list = []
    for run_i in tqdm(range(n_bootstraps), total=n_bootstraps):
        print('run', run_i)
        idx = np.random.choice(N, size=N, replace=True)

        model = (xgb_model, best_params, avg_best_iter)
        run_res, best_model = simulate_mediation_casual(model=model, x_df=x_df.iloc[idx],
                                                                         t=t.iloc[idx].values,
                                                                         m_df=m_df.iloc[idx], y=y.iloc[idx],
                                                                         patience_max=20)

        run_res = [[run_i] + list(run_res_i) for run_res_i in run_res]

        alpha, beta, gamma = run_res[-1][-3:]
        effect_res_list.append([run_i, alpha, beta, gamma])

        explainer = shap.TreeExplainer(best_model, feature_names=M_names)
        explanation = explainer(m_df.values)
        feature_importance = explanation.abs.values.mean(0)
        selected_index = feature_importance.argsort()[::-1][:10]
        selected_names = [explanation.feature_names[selected_index_i] for selected_index_i in selected_index]
        selected_importance = {
            explanation.feature_names[selected_index_i]: feature_importance[selected_index_i] * abs(beta) for
            selected_index_i in selected_index}
        feature_values_all_list.append(explanation.data)
        shap_values_all_list.append(explanation.values * beta)
        importance_rate_list.append(explanation.abs.values.sum(axis=0) / explanation.abs.values.sum())
        selected_importance_list.append(selected_importance)
        res_list.append(selected_names)

    ingred_counter = Counter(chain(*res_list))
    selected_sum = list(
        {k: v for k, v in dict(ingred_counter.most_common(10)).items() if v / n_bootstraps >= 0.6}.keys())
    similarities = [jaccard_similarity(dict(ingred_counter.most_common(10)).keys(), selected_i) for selected_i in
                    res_list]
    bsf = {k: v/ n_bootstraps for k, v in dict(ingred_counter.most_common(10)).items() if v / n_bootstraps >= 0.6}

    feature_values_all = np.concatenate(feature_values_all_list, axis=0)
    shap_values_all = np.concatenate(shap_values_all_list, axis=0)
    explanation_all = explanation
    explanation_all.data = feature_values_all
    explanation_all.values = shap_values_all

    cmap = sns.color_palette("vlag", as_cmap=True)
    shap.plots.beeswarm(explanation_all[:, selected_sum], show=False, color=cmap, alpha=0.2, plot_size=(8, 8))
    plt.xlabel('SHAP value (impact on m)', fontsize=18)
    plt.ylabel('Ingredient name', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{output_root}/{pres_name}_SHAPswarm_HF.tif', dpi=300, bbox_inches='tight')
    plt.close()

    feat_shap_spearman = {m_i: spearmanr(explanation_all[:, m_i].data, explanation_all[:, m_i].values).statistic for m_i in selected_sum}

    importance_rate_df = pd.DataFrame(importance_rate_list, columns=M_names) * 100
    selected_importance_rate_df_des = importance_rate_df.loc[:, selected_sum].describe(
        percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]).transpose().sort_values('mean', ascending=False)

    plot_df = importance_rate_df.loc[:, selected_importance_rate_df_des.index.tolist()].melt(var_name='Ingredient Name',
                                                                                             value_name='Importance Rate (%)')
    plt.figure(figsize=(8, 8))
    sns.set_context("paper", font_scale=1.5)
    ax = sns.boxenplot(data=plot_df, x='Importance Rate (%)', y='Ingredient Name')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.xlabel('Importance Rate (%)', fontsize=18)
    plt.ylabel('Ingredient name', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{output_root}/{pres_name}_SHAPboxen_HF.tif', dpi=300, bbox_inches='tight')
    plt.close()

    return effect_res_list, bsf, feat_shap_spearman, plot_df


def run_semi():
    demo_df = pd.read_csv('data/demoData.csv')
    alpha, beta, gamma = 0.4, 5, 3
    x_df = demo_df.loc[:, [col for col in demo_df.columns if col[0] == 'X']]
    t_df = demo_df['T']
    N=1000
    Xmat = x_df.values
    Trt=t_df.values
    dimx = Xmat.shape[-1]
    coef_m = np.random.normal(0, 0.1, size=dimx)
    a0 = np.dot(Xmat, coef_m)
    e0 = np.random.normal(0, 0.001, N)
    m = a0 * 1 + alpha * Trt + e0 * 1
    coef_y = np.random.normal(0, 0.1, size=dimx + 1)
    b0 = np.dot(np.c_[np.ones(N), Xmat], coef_y)
    error = np.random.normal(0, 0.001, N)
    Y = b0 + gamma * Trt + beta * m + error
    M = trans_new_M(m, 50)
    noise = np.random.normal(0, 0.2, size=M.shape)
    M = M.astype(float) + noise
    m_df = pd.DataFrame(M, columns=[f'M{i}' for i in range(50)])
    y = pd.Series(Y)

    result = run_rwechemscreener(x_df=x_df, t=t_df, m_df=m_df, y=y, n_bootstraps=10, pres_name='semi')




if __name__ == '__main__':
    pass
