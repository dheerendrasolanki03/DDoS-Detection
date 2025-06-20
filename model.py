import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore

import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

class DDoSModel:
    def __init__(self, DDOS_Dataset_Cleaned_50000):
        self.raw_data = pd.read_csv(DDOS_Dataset_Cleaned_50000)
        self.numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        self.label_encoder = None
        self.rf_model = None
        self.dl_model = None

    def run_eda(self):
        print("Dataset Shape:", self.raw_data.shape)
        print("\nColumns:", self.raw_data.columns.tolist())
        print("\nMissing Values:\n", self.raw_data.isnull().sum())
        print("\nData Types:\n", self.raw_data.dtypes)

        missing = self.raw_data.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing.plot(kind='bar', figsize=(8, 4), color='purple', title='Missing Values per Column')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
        else:
            print("No Missing Values Detected.")

        # Remove unwanted keywords from numeric features before plotting
        excluded_keywords = ['iat', 'psh', 'header', 'segment']
        filtered_cols = [col for col in self.numeric_cols if not any(keyword in col.lower() for keyword in excluded_keywords)]

        clean_data = self.raw_data[filtered_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if not clean_data.empty:
            for i in range(0, len(filtered_cols), 4):
                subset = filtered_cols[i:i + 4]
                clean_data[subset].hist(bins=20, figsize=(12, 8), edgecolor='black', grid=False)
                plt.tight_layout()
                plt.show()

            plt.figure(figsize=(15, 6))
            sns.boxplot(data=clean_data, orient='h')
            plt.title('Boxplot of Filtered Numeric Features')
            plt.tight_layout()
            plt.show()
        else:
            print("No valid numeric data for distribution or boxplot.")

        label_col = [col for col in self.raw_data.columns if col.strip().lower() == 'label']
        if label_col:
            self.raw_data[label_col[0]].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6))
            plt.title('Label Distribution')
            plt.ylabel('')
            plt.tight_layout()
            plt.show()
        else:
            print("Target column 'Label' not found.")

    def preprocess(self):
        self.raw_data.columns = self.raw_data.columns.str.strip()
        if 'Label' not in self.raw_data.columns:
            raise ValueError("Target column 'Label' not found.")

        clean_data = self.raw_data.replace([np.inf, -np.inf], np.nan).dropna()
        X = clean_data.drop('Label', axis=1)
        y = clean_data['Label']

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        self.input_shape = X_scaled.shape[1]

    def train_rf_model(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)

    def train_dl_model(self):
        model = Sequential([
            Dense(128, input_shape=(self.input_shape,), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
        ]

        model.fit(self.X_train, self.y_train, epochs=30, batch_size=64,
                  validation_split=0.2, callbacks=callbacks, verbose=1)

        self.dl_model = model

    def evaluate_rf(self):
        y_pred = self.rf_model.predict(self.X_test)
        print("Random Forest Classifier Results")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

    def evaluate_dl(self):
        y_pred_probs = self.dl_model.predict(self.X_test).flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)
        print("Deep Learning Model Results")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))


# Run the full pipeline
ddos = DDoSModel('DDOS_Dataset_Cleaned_50000.csv')

print("Running EDA...")
ddos.run_eda()

print("\nPreprocessing Data...")
ddos.preprocess()

print("\nTraining Random Forest...")
ddos.train_rf_model()

print("\nTraining Deep Learning Model...")
ddos.train_dl_model()

print("\n Evaluating Random Forest...")
ddos.evaluate_rf()

print("\n Evaluating Deep Learning Model...")
ddos.evaluate_dl()










