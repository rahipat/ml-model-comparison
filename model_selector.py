import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


class MLModelSelector:

    def __init__(self):
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_obj = None
        self.best_score = float('inf')
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def calculate_mape(self, actual, predicted):
        mask = actual != 0
        if mask.sum() == 0:
            return float('inf')
        return (np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100).mean()

    def calculate_metrics(self, actual, predicted):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mape = self.calculate_mape(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    def initialize_models(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.5, random_state=42, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=2000),

            'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),

            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),

            'K-Nearest Neighbors': KNeighborsRegressor(
                n_neighbors=7,
                weights='distance'
            ),
            'Support Vector Regression': SVR(
                kernel='rbf',
                C=100,
                epsilon=0.1,
                gamma='scale'
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }

    def get_feature_importance(self, model, model_name, feature_names):

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            self.feature_importance[model_name] = feature_imp
            return feature_imp
        return None

    def train_and_evaluate(self, X_train, y_train, X_test, y_test,
                           feature_names, scale_features=False, use_cv=False):

        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        for name, model in self.models.items():
            try:
                needs_scaling = name in ['K-Nearest Neighbors', 'Support Vector Regression',
                                         'Neural Network', 'Ridge Regression', 'Lasso Regression',
                                         'ElasticNet']

                X_tr = X_train_scaled if needs_scaling else X_train
                X_te = X_test_scaled if needs_scaling else X_test

                model.fit(X_tr, y_train)

                predictions = model.predict(X_te)
                predictions = np.maximum(predictions, 0)

                metrics = self.calculate_metrics(y_test, predictions)

                cv_score = None
                if use_cv:
                    cv_scores = cross_val_score(model, X_tr, y_train,
                                                cv=5, scoring='neg_mean_absolute_percentage_error',
                                                n_jobs=-1)
                    cv_score = -cv_scores.mean() * 100

                if name in ['Decision Tree', 'Random Forest', 'Extra Trees', 'Gradient Boosting']:
                    self.get_feature_importance(model, name, feature_names)

                result = {
                    'Model': name,
                    **metrics
                }
                if cv_score is not None:
                    result['CV_MAPE'] = cv_score

                self.results.append(result)

                if metrics['MAPE'] < self.best_score:
                    self.best_score = metrics['MAPE']
                    self.best_model = name
                    self.best_model_obj = model

                cv_str = f" | CV: {cv_score:6.2f}%" if cv_score else ""
                print(
                    f"✓ {name:25s} | MAPE: {metrics['MAPE']:6.2f}% | RMSE: {metrics['RMSE']:10.2f} | R²: {metrics['R2']:6.4f}{cv_str}")

            except Exception as e:
                print(f"✗ {name:25s} | Error: {str(e)[:50]}")

    def get_results_dataframe(self):
        df = pd.DataFrame(self.results)
        return df.sort_values('MAPE').reset_index(drop=True)

    def print_feature_importance(self, top_n=10):
        if self.best_model in self.feature_importance:
            print(f"\n{'=' * 80}")
            print(f"TOP {top_n} FEATURES FOR {self.best_model}")
            print('=' * 80)
            print(self.feature_importance[self.best_model].head(top_n).to_string(index=False))

    def print_summary(self):
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)

        results_df = self.get_results_dataframe()
        print(results_df.to_string(index=False))

        print("\n" + "=" * 80)
        print(f"🏆 BEST MODEL: {self.best_model}")
        print(f"   MAPE: {self.best_score:.2f}%")
        print("=" * 80)


def analyze_feature_correlations(crime_data, target='total_arrests'):
    numeric_cols = crime_data.select_dtypes(include=[np.number]).columns
    correlations = crime_data[numeric_cols].corrwith(crime_data[target]).abs()
    correlations = correlations.drop([target, 'id', 'year'], errors='ignore')
    correlations = correlations.sort_values(ascending=False)

    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS (with total_arrests)")
    print("=" * 80)
    print("\nTop 15 Most Correlated Features:")
    for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
        print(f"{i:2d}. {feature:25s}: {corr:.4f}")

    return correlations


def select_best_features(crime_data, target='total_arrests', method='correlation', top_k=15):

    exclude_cols = ['id', 'year', target]
    feature_cols = [col for col in crime_data.columns if col not in exclude_cols]

    if method == 'correlation':
        correlations = crime_data[feature_cols + [target]].corr()[target].abs()
        correlations = correlations.drop(target).sort_values(ascending=False)
        selected_features = correlations.head(top_k).index.tolist()

    elif method == 'all':
        selected_features = feature_cols

    elif method == 'crime_types':
        crime_features = ['homicide', 'rape', 'robbery', 'aggravated_assault', 'burglary',
                          'larceny', 'motor_vehicle_theft', 'arson', 'violent_crime',
                          'property_crime', 'population']
        selected_features = [f for f in crime_features if f in feature_cols]

    return selected_features


def main():

    print("Loading crime data...")
    crime = pd.read_csv("crime_data.csv")

    print(f"Dataset shape: {crime.shape}")
    print(f"Columns: {len(crime.columns)}")
    print(f"Years: {crime['year'].min()} - {crime['year'].max()}")

    correlations = analyze_feature_correlations(crime)

    training_data = crime[crime["year"] <= 2012].copy()
    test_data = crime[crime["year"] > 2012].copy()

    print(f"\nTraining samples: {len(training_data)} ({crime['year'].min()}-2012)")
    print(f"Test samples: {len(test_data)} (2013-{crime['year'].max()})")

    print("\n" + "=" * 80)
    print("FEATURE SELECTION STRATEGY")
    print("=" * 80)

    selected_features = select_best_features(crime, method='correlation', top_k=15)
    print(f"\nUsing top 15 correlated features:")
    print(", ".join(selected_features))

    target = "total_arrests"

    X_train = training_data[selected_features].values
    y_train = training_data[target].values
    X_test = test_data[selected_features].values
    y_test = test_data[target].values

    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATING 12 ML MODELS")
    print("=" * 80 + "\n")

    selector = MLModelSelector()
    selector.initialize_models()
    selector.train_and_evaluate(X_train, y_train, X_test, y_test,
                                selected_features, scale_features=True, use_cv=False)

    selector.print_summary()
    selector.print_feature_importance(top_n=10)

    results_df = selector.get_results_dataframe()
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\n✓ Results saved to 'model_comparison_results.csv'")

    if selector.feature_importance:
        for model_name, importance_df in selector.feature_importance.items():
            filename = f"feature_importance_{model_name.replace(' ', '_')}.csv"
            importance_df.to_csv(filename, index=False)
        print(f"✓ Feature importance saved for {len(selector.feature_importance)} models")

    return selector


if __name__ == "__main__":
    selector = main()
