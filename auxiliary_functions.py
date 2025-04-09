import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from sklearn.model_selection import cross_validate
import math
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from lime import lime_tabular
import os


def violin_plotter(df: pd.DataFrame, features: list, split_column : str =None, palette : dict =None, ax=None) -> None:
    """
    Creates violin plots for the given features, standardized, optionally grouped by a categorical column.
    Supports subplotting for side-by-side comparisons.

    Parameters:
    - df (DataFrame): The dataset containing numerical features.
    - features (list): List of feature names to plot.
    - split_column (str, optional): Column used to split the violin plots. If None, plots without classes.
    - palette (dict, optional): Dictionary defining colors for categories in the split_column.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib subplot axis for side-by-side plotting.

    Returns:
    - None (Displays the violin plot inside the given subplot).
    """

    df = df.copy()

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    if split_column and split_column in df.columns:
        df_scaled[split_column] = df_scaled[split_column].astype('category') 
        df_melted = df_scaled.melt(id_vars=[split_column], value_vars=features, 
                                   var_name="Feature", value_name="Standardized Value")
    else:
        df_melted = df_scaled.melt(value_vars=features, var_name="Feature", value_name="Standardized Value")
        split_column = None  

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6)) 
    if split_column:
        sns.violinplot(x="Feature", y="Standardized Value", hue=split_column, data=df_melted, 
                       split=True, inner="quartile", palette=palette, ax=ax)
        ax.legend(title=split_column)
    else:
        sns.violinplot(x="Feature", y="Standardized Value", data=df_melted, inner="quartile", color="skyblue", ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Standardized Distribution of Features" + (f" by {split_column}" if split_column else ""))
    return None


def swarm_plotter(df: pd.DataFrame, features: list, split_column : str =None, ax=None, palette : dict =None) -> None:
    """
    Creates swarm plots for the given features, standardized, optionally grouped by a categorical column.
    Supports subplotting for side-by-side comparisons.

    Parameters:
    - df (DataFrame): The dataset containing numerical features.
    - features (list): List of feature names to plot.
    - split_column (str, optional): Column used to split the violin plots. If None, plots without classes.
    - palette (dict, optional): Dictionary defining colors for categories in the split_column.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib subplot axis for side-by-side plotting.

    Returns:
    - None (Displays the violin plot inside the given subplot).
    """

    df = df.copy()

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    if split_column and split_column in df.columns:
        df_scaled[split_column] = df_scaled[split_column].astype('category')
        df_melted = df_scaled.melt(id_vars=[split_column], value_vars=features, 
                                   var_name="Feature", value_name="Standardized Value")
    else:
        df_melted = df_scaled.melt(value_vars=features, var_name="Feature", value_name="Standardized Value")
        split_column = None  # Ensure it's set to None for plotting

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6)) 

    if split_column:
        sns.swarmplot(x="Feature", y="Standardized Value", hue=split_column, data=df_melted, ax=ax, palette=palette)
        ax.legend(title=split_column)

    else:
        sns.swarmplot(x="Feature", y="Standardized Value", data=df_melted, color="skyblue", ax=ax, palette=palette)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Standardized Distribution of Features" + (f" by {split_column}" if split_column else ""))
    return None

def get_feature_importances(features :pd.DataFrame, target: pd.DataFrame, n_trees : int, n_runs: int =1, ax=None) -> Tuple[pd.DataFrame, plt.figure, plt.axes]:

    feature_importances = np.zeros((n_runs, features.shape[1]))

    for i in range(n_runs):
        rnf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1)
        rnf.fit(features, target)
        feature_importances[i, :] = rnf.feature_importances_

    avg_importances = feature_importances.mean(axis=0)
    std_importances = feature_importances.std(axis=0)
  
    feat_importances = pd.Series(avg_importances, index=features.columns).sort_values(ascending=True)
    std_sorted = pd.Series(std_importances, index=features.columns).loc[feat_importances.index]

    feature_importance_df = pd.DataFrame({
    'Feature': feat_importances.index,
    'Importance': feat_importances.values,
    'Std Dev': std_sorted.values})
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_figure = True
    else:
        fig = ax.figure
        created_figure = False


    ax.barh(feature_importance_df['Feature'], 
            feature_importance_df['Importance'], 
            xerr=feature_importance_df['Std Dev'], 
            capsize=5, 
            color=plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df))))

    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance with Error Bars (Random Forest)")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    if created_figure:
        plt.close(fig)

    return feature_importance_df, fig, ax




def evaluate_models(model, datasets :dict, scoring : dict='accuracy') -> pd.DataFrame:
    """
    Evaluates a model on multiple datasets using cross-validation. Returns a dataframe with the results.

    Parameters:
    - model: an instance of a model.
    - datasets: a dictionary containing (possibly several) pandas dataframes.
    - scoring: a dictionary containing (possibly several) evaluation metrics.

    Returns:
    - pd.Dataframe: A dataframe containing the results of the models for the different metrics in each dataset.
    """
    results = []

    for label, (X, y) in datasets.items():
        cv_results = cross_validate(model, X, y, cv=10, scoring=scoring, return_train_score=False)

        result = {"Dataset": label}

        if isinstance(scoring, dict):
            metric_names = scoring.keys()
        elif isinstance(scoring, list):
            metric_names = scoring
        else:
            metric_names = [scoring]

        for metric in metric_names:
            score_key = f'test_{metric}' if isinstance(scoring, (list, dict)) else 'test_score'
            scores = cv_results[score_key]
            result[f"{metric} Mean"] = np.mean(scores)
            result[f"{metric} Std"] = np.std(scores)

        results.append(result)

    return pd.DataFrame(results)



def evaluate_multiple_models(models :dict , datasets: dict, scoring: dict ='accuracy', groupby: str='dataset', ordering :str =None)-> pd.DataFrame:
    """
    Evaluates multiple models on multiple datasets using cross-validation.
    Returns a DataFrame with the evaluation results, without standard deviation.
    Supports grouping by model or dataset and ordered display by metric.
    
    Args:
        models (dict): A dictionary where keys are model names and values are model instances.
        datasets (dict): A dictionary where keys are dataset labels and values are tuples (X, y).
        scoring (str, list, or dict): Scoring metric(s) to evaluate. Can be a single metric string, a list of metrics, or a dict of metrics.
        groupby (str): Determines how results are grouped ('dataset' or 'model'). Defaults to 'dataset'.
        ordering (str or None): Metric name to order models by. Defaults to None (no ordering).
        
    Returns:
        pd.DataFrame: A MultiIndex DataFrame containing the mean score for each model and dataset.
    """
    if groupby not in ['dataset', 'model']:
        raise ValueError("The 'groupby' parameter must be either 'dataset' or 'model'")
    
    results = []

    for label, (X, y) in datasets.items():
        for model_name, model in models.items():
            cv_results = cross_validate(model, X, y, cv=10, scoring=scoring, return_train_score=False)

            result = {"Dataset": label, "Model": model_name}

            if isinstance(scoring, dict):
                metric_names = scoring.keys()
            elif isinstance(scoring, list):
                metric_names = scoring
            else:
                metric_names = [scoring]

            for metric in metric_names:
                score_key = f'test_{metric}' if isinstance(scoring, (list, dict)) else 'test_score'
                scores = cv_results[score_key]
                result[f"{metric} Mean"] = np.mean(scores)

            results.append(result)

    results_df = pd.DataFrame(results)

    if ordering:
        if groupby == 'dataset':
            results_df.sort_values(by=["Dataset", f"{ordering} Mean"], ascending=[True, False], inplace=True)
        else:  # groupby == 'model'
            results_df.sort_values(by=["Model", f"{ordering} Mean"], ascending=[True, False], inplace=True)

    index_columns = ['Dataset', 'Model'] if groupby == 'dataset' else ['Model', 'Dataset']
    results_df.set_index(index_columns, inplace=True)

    return results_df



def cross_val_confusion(model, X: pd.DataFrame, y: pd.DataFrame, cv: int=10) -> None:
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Benign", "Malignant"])
    
    print("Classification Report:")
    print(report)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    disp.plot(cmap="Blues")


def multiple_cross_val_confusions(models: dict, X:pd.DataFrame, y:pd.DataFrame, cv=10, max_cols=4)-> None:
    """
    Plots confusion matrices for a dictionary of models using cross-validation predictions, without color bars,
    and displays them with a maximum of 4 per row.
    
    Args:
        models (dict): A dictionary where keys are model names and values are model instances.
        X (pd.DataFrame or np.ndarray): Training features.
        y (pd.Series or np.ndarray): Target labels.
        cv (int): Number of cross-validation folds. Defaults to 5.
        max_cols (int): Maximum number of plots per row. Defaults to 4.
    """
    n_models = len(models)
    n_rows = math.ceil(n_models / max_cols)
    n_cols = min(n_models, max_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(model_name)
    
    # Hide unused subplots if any
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()



def perform_random_search(model, param_grid:dict , X_train :pd.DataFrame, y_train:pd.DataFrame, scoring_metric, n_iter: int =100, cv:int=10, n_jobs=-1, verbose=1) -> dict:
    """
    Performs RandomizedSearchCV on a single model with a given parameter grid.
    
    Args:
        model (sklearn estimator): The model to be optimized.
        param_grid (dict): Parameter grid for the model.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        scoring_metric (str or callable): The scoring metric to use for optimization.
        n_iter (int): Number of parameter settings that are sampled. Defaults to 20.
        cv (int): Number of cross-validation folds. Defaults to 5.
        n_jobs (int): Number of jobs to run in parallel. Defaults to -1 (use all processors).
        verbose (int): Controls the verbosity. Defaults to 1.
        
    Returns:
        dict: Best hyperparameters and best score.
    """
    print(f"\nPerforming RandomizedSearchCV for {model.__class__.__name__}...\n")
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        scoring=scoring_metric
    )
    
    search.fit(X_train, y_train)
    
    best_params = search.best_params_
    best_score = search.best_score_
    
    print(f"\nBest Score for {model.__class__.__name__}: {best_score}")
    print(f"Best Parameters for {model.__class__.__name__}: {best_params}")
    
    return {'best_params': best_params, 'best_score': best_score}



def evaluate_final_model(model, best_params: dict, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, modelname: str=None):
    """
    Trains and evaluates a final model using the provided best hyperparameters.

    Args:
    - model (estimator or class): The model instance.
    - best_params (dict): Dictionary of best hyperparameters found by RandomizedSearchCV or GridSearchCV.
    - X_train (array-like): Training features.
    - y_train (array-like): Training labels.
    - X_test (array-like): Testing features.
    - y_test (array-like): Testing labels.

    Returns:
    - model: Trained model.
    - report (str): Detailed classification report.
    """
    model.set_params(**best_params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    report = report.iloc[:2].reset_index()  # Resetting index to make it a column
    report = report.rename(columns={'index': 'Class'})
    if modelname:
      report['Model']=modelname
    else:  
        report['Model'] = model.__class__.__name__
    report.set_index(['Model', 'Class'], inplace=True)


    return model, report


def multiple_final_confusions(models, X: pd.DataFrame, y: pd.DataFrame, max_cols:int =4) -> None:
    """
    Plots confusion matrices for a dictionary of models using cross-validation predictions, without color bars,
    and displays them with a maximum of 4 per row.
    
    Args:
        models (dict): A dictionary where keys are model names and values are model instances.
        X (pd.DataFrame or np.ndarray): Training features.
        y (pd.Series or np.ndarray): Target labels.
        cv (int): Number of cross-validation folds. Defaults to 5.
        max_cols (int): Maximum number of plots per row. Defaults to 4.
    """
    n_models = len(models)
    n_rows = math.ceil(n_models / max_cols)
    n_cols = min(n_models, max_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X)
        
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(model_name)
    
    # Hide unused subplots if any
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_lime_reports(model, X_train : pd.DataFrame, X_test : pd.DataFrame, model_name, num_instances=100, instances_per_page=10, output_dir="lime_reports") -> None:
    """
    Generates paginated LIME reports for a given model and dataset.
    
    Parameters:
    - model: Trained model to explain.
    - X_train: Training data used to fit the model (for LIME initialization).
    - X_test: Test data to generate explanations for.
    - model_name: A string name for the model (e.g., 'best_model_XGB').
    - num_instances: Total number of instances to explain.
    - instances_per_page: Number of instances per HTML file.
    - output_dir: Directory to save the HTML reports.
    """

    os.makedirs(output_dir, exist_ok=True)
    

    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Benign', 'Malign'],
        mode='classification'
    )

    html_explanations = []
    for i in range(num_instances):
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba
        )
        html_content = exp.as_html()
        html_explanations.append(html_content)

    num_pages = (num_instances + instances_per_page - 1) // instances_per_page
    page_files = []

    for page in range(num_pages):
        start = page * instances_per_page
        end = min(start + instances_per_page, num_instances)
        page_file = f"{output_dir}/report_page_{page + 1}.html"
        page_files.append(page_file)
        
        with open(page_file, "w", encoding="utf-8") as outfile:
            outfile.write(f"""
            <html>
            <head>
                <title>{model_name} - LIME Report - Page {page + 1}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    h1 {{ text-align: center; }}
                    h2 {{ color: darkblue; }}
                    .patient-report {{ page-break-after: always; }}
                    a {{ text-decoration: none; color: darkred; }}
                </style>
            </head>
            <body>
            <h1>{model_name} - LIME Report - Page {page + 1}</h1>
            """)
            
            for i in range(start, end):
                outfile.write(f'<div id="patient_{i}" class="patient-report">')
                outfile.write(f'<h2>Report for Patient {i}</h2>')
                outfile.write(html_explanations[i])
                outfile.write("</div>")
            
            outfile.write("</body></html>")

    index_file = f"{output_dir}/index.html"
    with open(index_file, "w", encoding="utf-8") as outfile:
        outfile.write("""
        <html>
        <head>
            <title>Combined LIME Report - Index</title>
            <style>
                body { font-family: Arial, sans-serif; }
                h1 { text-align: center; }
                ul { list-style-type: none; }
                li { margin: 5px 0; }
                a { text-decoration: none; color: darkred; }
            </style>
        </head>
        <body>
            <h1>Combined LIME Report - Index</h1>
            <ul>
        """)

        for page_num, page_file in enumerate(page_files):
            outfile.write(f'<li><a href="{os.path.basename(page_file)}">Report Page {page_num + 1}</a></li>')

        outfile.write("</ul></body></html>")

    print(f"LIME reports generated successfully! Check the '{output_dir}' directory.")

