import pandas as pd
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product


def parse_json_and_run_ml(json_data, dataset_path):
    # Load JSON data
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        print("Error while parsing JSON:", e)
        return

    # Step 1: Read target and type of regression
    target = data['design_state_data']['target']['target']
    prediction_type = data['design_state_data']['target']['prediction_type']

    # Step 2: Read features and apply missing value imputation
    feature_handling = data['design_state_data']['feature_handling']
    features_to_impute = [feature for feature in feature_handling if 'impute_with' in feature_handling[feature]['feature_details']]
    feature_impute_values = {feature: feature_handling[feature]['feature_details']['impute_value'] for feature in features_to_impute}

    df = pd.read_csv(dataset_path)
    for feature in features_to_impute:
        impute_value = feature_impute_values[feature]
        df[feature].fillna(impute_value, inplace=True)

    # Step 3: Preprocess the "species" column
    species_column = df['species']
    df = df.drop('species', axis=1)
    encoder = OneHotEncoder(sparse=False)
    species_encoded = encoder.fit_transform(species_column.values.reshape(-1, 1))
    species_columns = [f'species_{i}' for i in range(species_encoded.shape[1])]
    df[species_columns] = pd.DataFrame(species_encoded, columns=species_columns)

    # Step 4: Compute feature reduction based on input
    feature_reduction_method = data['design_state_data']['feature_reduction']['feature_reduction_method']
    if feature_reduction_method == 'PCA':
        num_of_features_to_keep = data['design_state_data']['feature_reduction']['num_of_features_to_keep']
        pca = PCA(n_components=num_of_features_to_keep)
        feature_reduction_transformer = pca
    elif feature_reduction_method == 'Tree-based':
        feature_reduction_transformer = SelectKBest(score_func=f_regression,k="all")
    else:
        feature_reduction_transformer = None

    # Step 5: Create the model objects based on prediction_type specified in JSON
    model_objects = {}
    algorithms = data['design_state_data']['algorithms']
    for algo, params in algorithms.items():
        if params['is_selected']:
            if prediction_type == 'Regression':
                if algo == 'RandomForestRegressor':
                    model_objects[algo] = RandomForestRegressor()
                elif algo == 'GradientBoostingRegressor':
                    model_objects[algo] = GradientBoostingRegressor()
                elif algo == 'LinearRegression':
                    model_objects[algo] = LinearRegression()
                elif algo == 'RidgeRegression':
                    model_objects[algo] = Ridge()
                elif algo == 'LassoRegression':
                    model_objects[algo] = Lasso()
                elif algo == 'ElasticNetRegression':
                    model_objects[algo] = ElasticNet()
                elif algo == 'SVR':
                    model_objects[algo] = SVR()

    # Step 6: Run the fit and predict on each model with hyperparameter tuning using GridSearchCV
    for algo, model in model_objects.items():
        print(f"Running {algo}...")

        if algo in data['design_state_data']['hyperparameters']:
            param_grid = data['design_state_data']['hyperparameters'][algo]
            grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
            pipeline = Pipeline([
                ('feature_reduction', feature_reduction_transformer),
                ('model', grid_search)
            ])
        else:
            pipeline = Pipeline([
                ('feature_reduction', feature_reduction_transformer),
                ('model', model)
            ])

        pipeline.fit(df.drop(target, axis=1), df[target])
        predictions = pipeline.predict(df.drop(target, axis=1))

        # Step 7: Log standard model metrics
        mse = mean_squared_error(df[target], predictions)
        r2 = r2_score(df[target], predictions)
        print(f"{algo} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}")


if __name__ == '__main__':
    json_data = '''
    {
        "design_state_data": {
            "target": {
                "prediction_type": "Regression",
                "target": "petal_width",
                "type": "regression",
                "partitioning": true
            },
            "train": {
                "policy": "Split the dataset",
                "time_variable": "sepal_length",
                "sampling_method": "No sampling(whole data)",
                "split": "Randomly",
                "k_fold": false,
                "train_ratio": 0,
                "random_seed": 0
            },
            "feature_handling": {
                "sepal_length": {
                    "feature_name": "sepal_length",
                    "is_selected": true,
                    "feature_variable_type": "numerical",
                    "feature_details": {
                        "numerical_handling": "Keep as regular numerical feature",
                        "rescaling": "No rescaling",
                        "make_derived_feats": false,
                        "missing_values": "Impute",
                        "impute_with": "Average of values",
                        "impute_value": 0
                    }
                },
                "sepal_width": {
                    "feature_name": "sepal_width",
                    "is_selected": true,
                    "feature_variable_type": "numerical",
                    "feature_details": {
                        "numerical_handling": "Keep as regular numerical feature",
                        "rescaling": "No rescaling",
                        "make_derived_feats": false,
                        "missing_values": "Impute",
                        "impute_with": "custom",
                        "impute_value": -1
                    }
                },
                "petal_length": {
                    "feature_name": "petal_length",
                    "is_selected": true,
                    "feature_variable_type": "numerical",
                    "feature_details": {
                        "numerical_handling": "Keep as regular numerical feature",
                        "rescaling": "No rescaling",
                        "make_derived_feats": false,
                        "missing_values": "Impute",
                        "impute_with": "Average of values",
                        "impute_value": 0
                    }
                },
                "petal_width": {
                    "feature_name": "petal_width",
                    "is_selected": true,
                    "feature_variable_type": "numerical",
                    "feature_details": {
                        "numerical_handling": "Keep as regular numerical feature",
                        "rescaling": "No rescaling",
                        "make_derived_feats": false,
                        "missing_values": "Impute",
                        "impute_with": "custom",
                        "impute_value": -2
                    }
                },
                "species": {
                    "feature_name": "species",
                    "is_selected": true,
                    "feature_variable_type": "text",
                    "feature_details": {
                        "text_handling": "Tokenize and hash",
                        "hash_columns": 0
                    }
                }
            },
            "feature_reduction": {
                "feature_reduction_method": "Tree-based",
                "num_of_features_to_keep": 4,
                "num_of_trees": 5,
                "depth_of_trees": 6
            },
            "hyperparameters": {
                "stratergy": "Grid Search",
                "shuffle_grid": true,
                "random_state": 1,
                "max_iterations": 2,
                "max_search_time": 3,
                "parallelism": 5,
                "cross_validation_stratergy": "Time-based K-fold(with overlap)",
                "num_of_folds": 6,
                "split_ratio": 0,
                "stratified": true
            },
            "algorithms": {
                "RandomForestRegressor": {
                    "model_name": "Random Forest Regressor",
                    "is_selected": true,
                    "min_trees": 10,
                    "max_trees": 20,
                    "feature_sampling_statergy": "Default",
                    "min_depth": 20,
                    "max_depth": 25,
                    "min_samples_per_leaf_min_value": 5,
                    "min_samples_per_leaf_max_value": 10,
                    "parallelism": 0
                },
                "GradientBoostingRegressor": {
                    "model_name": "Gradient Boosted Trees",
                    "is_selected": true,
                    "num_of_BoostingStages": [67, 89],
                    "feature_sampling_statergy": "Fixed number",
                    "learningRate": [],
                    "use_deviance": true,
                    "use_exponential": false,
                    "fixed_number": 22,
                    "min_subsample": 1,
                    "max_subsample": 2,
                    "min_stepsize": 0.1,
                    "max_stepsize": 0.5,
                    "min_iter": 20,
                    "max_iter": 40,
                    "min_depth": 5,
                    "max_depth": 7
                },
                "LinearRegression": {
                    "model_name": "LinearRegression",
                    "is_selected": true,
                    "parallelism": 2,
                    "min_iter": 30,
                    "max_iter": 50,
                    "min_regparam": 0.5,
                    "max_regparam": 0.8,
                    "min_elasticnet": 0.5,
                    "max_elasticnet": 0.8
                },
                "RidgeRegression": {
                    "model_name": "RidgeRegression",
                    "is_selected": true,
                    "regularization_term": "Specify values to test",
                    "min_iter": 30,
                    "max_iter": 50,
                    "min_regparam": 0.5,
                    "max_regparam": 0.8
                }
            }
        }
    }
    '''

    dataset_path = '/content/drive/MyDrive/iris.csv'
    parse_json_and_run_ml(json_data, dataset_path)
