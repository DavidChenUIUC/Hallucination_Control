import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle


class HallucinationPredictor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.seed = 1234
        self.model = RandomForestClassifier(random_state=self.seed)
    
    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        print("Data loaded successfully.")
    
    def preprocess_data(self):
        data_clean = self.data.drop(columns=['sample_id', 'system','hallucination_level', 'MQAG counting', 'hallucination_type_extrinsic', 'hallucination_type_NULL','hallucination_type_intrinsic'])
        self.X = data_clean.drop(columns=['label'])  # features
        self.y = data_clean['label']  # target
        print("Data preprocessed successfully.")
    
    def feature_importance(self):
        feature_importances = self.model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        # print("|- importance ranking : \n" , features_df)
        return features_df
    def k_fold_cross_validation(self, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        accuracies = []

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        acc = np.mean(accuracies)
        print(f"Base line average accuracy over {k}-fold cross-validation: {acc}")
        return acc
    def save_model(self, model, filename="random_forest"):
        with open(filename+'.pkl', 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename}")
        
    def tune_and_plot(self, start=10, end=15):
        accuracies = []
        feature_counts = list(range(start, end + 1))
        best_acc, best_model = -1, None
        self.k_fold_cross_validation()
        
        for n_features in feature_counts:
            top_features = self.feature_importance()['Feature'].head(n_features).tolist()
            X_sub = self.X[top_features]

            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            fold_accuracies = []

            for train_index, test_index in kf.split(X_sub):
                X_train, X_test = X_sub.iloc[train_index], X_sub.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                model = RandomForestClassifier(random_state=self.seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                fold_accuracies.append(accuracy_score(y_test, y_pred))
            acc = np.mean(fold_accuracies)
            if acc > best_acc:
                best_acc = acc
                best_model = model
                
            print(f'Number of top features: {n_features}, Accuracy: {acc}')
            accuracies.append(np.mean(fold_accuracies))

        # Plotting the accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(feature_counts, accuracies, marker='o')
        plt.title('Model Accuracy vs. Number of Top Features')
        plt.xlabel('Number of Top Features')
        plt.ylabel('Cross-Validated Accuracy')
        plt.xticks(feature_counts)
        plt.grid(True)
        # plt.show()
        plt.tight_layout()
        plt.savefig('model_accuracy_vs_num_top_features.png')
        print('|- Plot saved successfully.')
        
        self.save_model(best_model)
        print('|- Best model saved')

# Usage
file_path = './hallucination_dataset_null_and_hal_with_gpt.csv'
predictor = HallucinationPredictor(filepath=file_path)
predictor.load_data()
predictor.preprocess_data()
predictor.tune_and_plot(start=1, end=len(predictor.X.columns))
