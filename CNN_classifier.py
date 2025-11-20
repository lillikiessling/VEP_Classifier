import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras import backend as K, regularizers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation


# TODO: combine to one class since CNN architecture is the same!
# ---------------------------------------------------------
#  1D CNN Classifier
# ---------------------------------------------------------

class CNN1D:
    def __init__(self, X, y, n_splits=10, random_state=42, model_type="cnn"):
        self.X = X
        self.y = np.array(y)
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_type = model_type
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # --- 1D CNN ---
    def build_cnn(self, input_shape, n_classes):
        model = Sequential([  
            Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-4)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    # --- Training ---
    def fit(self, epochs=25, batch_size=8, verbose=0):
        y_true, y_pred = [], []
        n_classes = len(np.unique(self.y))
        label_enc = pd.Series(self.y).astype("category")
        label_codes = label_enc.cat.codes
        label_decoder = dict(enumerate(label_enc.cat.categories))

        # Detect structure (dict or array)
        if isinstance(self.X, dict):
            X_data = np.hstack([self.X[k] for k in self.X.keys()])
        else:
            X_data = np.array(self.X)

    
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X_data, label_codes), 1):
            # Split train/test
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = label_codes[train_idx], label_codes[test_idx]

            # 1D signals (raw, DWT)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)
            input_shape = X_train.shape[1:]
            model = self.build_cnn(input_shape, n_classes)

            # One-hot encode labels
            y_train_cat = to_categorical(y_train, num_classes=n_classes)
            y_test_cat = to_categorical(y_test, num_classes=n_classes)

            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            # Print accuracy progress
            print(f"{'Epoch':>5} | {'Train Acc':>10} | {'Val Acc':>10}")
            print("-" * 33)
            for epoch in range(epochs):
                train_acc = history.history['accuracy'][epoch]
                val_acc = history.history['val_accuracy'][epoch]
                print(f"{epoch+1:5d} | {train_acc:10.4f} | {val_acc:10.4f}")
            print("-" * 33)
            print(f"Final Training Acc: {history.history['accuracy'][-1]:.4f} | "
                  f"Validation Acc: {history.history['val_accuracy'][-1]:.4f}")


            preds = model.predict(X_test)
            preds = np.argmax(preds, axis=1)

            y_pred.extend([label_decoder[i] for i in preds])
            y_true.extend([label_decoder[i] for i in y_test])

            # SHAP values
            explainer = shap.DeepExplainer(model, X_train[:100])
            shap_values = explainer.shap_values(X_test)
        
        return np.array(y_true), np.array(y_pred), shap_values


    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        
        report = classification_report(
            y_true,
            y_pred,
            target_names=np.unique(self.y),
            digits=3,
            output_dict=True  # <-- this makes it return a dict
        )

        print(confusion_matrix(y_true, y_pred, labels=["BC_Only","RGC_Only","BC_and_RGC"]))
        
        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": report,
        }
    
# ---------------------------------------------------------
# Multichannel 1D CNN Classifier
# ---------------------------------------------------------
class Multichannel_1DCNN:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.n_classes = None

    def build_cnn(self, n_points, n_levels):
        model = Sequential([
            Input(shape=(n_points, n_levels)),
            Conv1D(filters=16, kernel_size=5, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    # --- Training ---
    def fit_nfoldcv(self, X, y, n_splits=5, random_state=42, epochs=25, batch_size=8):
        X = np.array(X)
        y = np.array(y)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.n_classes = len(np.unique(y))
        y_true, y_pred = [], []
        label_enc = pd.Series(y).astype("category")
        label_codes = label_enc.cat.codes
        label_decoder = dict(enumerate(label_enc.cat.categories))

        all_shap_values = []
        all_test_indices = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, label_codes), 1):
            print(f"\n--- Fold {fold}/{n_splits} ---")
            K.clear_session()
            # Split train/test
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = label_codes[train_idx], label_codes[test_idx]
            # One-hot encode labels
            y_train_cat = to_categorical(y_train, num_classes=self.n_classes)
            y_test_cat = to_categorical(y_test, num_classes=self.n_classes)

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )

            n_points = X_train.shape[1]
            n_levels = X_train.shape[2]

            model = self.build_cnn(n_points, n_levels)
            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping]
            )

            # print("Model output shape:", model.output_shape)
            # for epoch in range(len(history.history['accuracy'])):
            #     train_acc = history.history['accuracy'][epoch]
            #     val_acc = history.history['val_accuracy'][epoch]
            #     print(f"{epoch+1:5d} | {train_acc:10.4f} | {val_acc:10.4f}")
            best_epoch = np.argmin(history.history['val_loss']) + 1
            best_val_loss = history.history['val_loss'][best_epoch - 1]
            best_val_acc = history.history['val_accuracy'][best_epoch - 1]
            best_train_acc = history.history['accuracy'][best_epoch - 1]
            print(f"[Fold {fold}] Early stopped at epoch {len(history.history['val_loss'])} "
            f"→ best epoch = {best_epoch}, val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, train_acc={best_train_acc:.4f}")

            preds = model.predict(X_test)
            preds = np.argmax(preds, axis=1) # Convert one-hot to class labels

            y_pred.extend([label_decoder[i] for i in preds])
            y_true.extend([label_decoder[i] for i in y_test])

            # SHAP values
            background = X_train[:min(100, len(X_train))]
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(X_test) # dimension (number_testsamples_in_fold, n_points, n_levels, n_classes)
            # average over test samples in fold
            shap_values = np.mean(shap_values, axis=0)  # dimension (n_points, n_levels, n_classes)
            all_shap_values.append(shap_values)
            all_test_indices.extend(test_idx)
        return np.array(y_true), np.array(y_pred), all_shap_values, all_test_indices
    

    def fit_traintest(self, X_train, y_train, X_test, y_test, epochs=25, batch_size=8):
        self.n_classes = len(np.unique(y_train))
        y_true, y_pred = [], []
        label_enc = pd.Series(y_train).astype("category")
        label_codes = label_enc.cat.codes
        label_decoder = dict(enumerate(label_enc.cat.categories))

        # One-hot encode labels
        # Encode string labels as integers
        label_enc_train = pd.Series(y_train).astype("category")
        categories = label_enc_train.cat.categories
        cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=False)
        y_train_codes = pd.Series(y_train, dtype=cat_type).cat.codes.to_numpy()
        y_test_codes = pd.Series(y_test, dtype=cat_type).cat.codes.to_numpy()

        # One-hot encode numeric codes
        y_train_cat = to_categorical(y_train_codes, num_classes=self.n_classes)
        y_test_cat = to_categorical(y_test_codes, num_classes=self.n_classes)


        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        n_points = X_train.shape[1]
        n_levels = X_train.shape[2]

        model = self.build_cnn(n_points, n_levels)
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping]
        )

        print("Model output shape:", model.output_shape)
        for epoch in range(len(history.history['accuracy'])):
            train_acc = history.history['accuracy'][epoch]
            val_acc = history.history['val_accuracy'][epoch]
            print(f"{epoch+1:5d} | {train_acc:10.4f} | {val_acc:10.4f}")
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = history.history['val_loss'][best_epoch - 1]
        best_val_acc = history.history['val_accuracy'][best_epoch - 1]
        best_train_acc = history.history['accuracy'][best_epoch - 1]
        print(f"→ best epoch = {best_epoch}, val_loss={best_val_loss:.4f}, "
              f"val_acc={best_val_acc:.4f}, train_acc={best_train_acc:.4f}")

        preds = np.argmax(model.predict(X_test), axis=1)
        y_pred = [categories[i] for i in preds]
        y_true = [categories[i] for i in y_test_codes]


        # SHAP values
        background = X_train[:min(100, len(X_train))]
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(X_test) # dimension (n_points, n_levels, n_classes)
        return np.array(y_true), np.array(y_pred), shap_values

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        report = classification_report(
            y_true,
            y_pred,
            target_names=np.unique(y_true),
            digits=3,
            output_dict=True 
        )
        print(confusion_matrix(y_true, y_pred, labels=["BC_Only","RGC_Only","BC_and_RGC"]))
        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": report,
        }
    