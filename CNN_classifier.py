from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras import Input
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import shap



class TemporalAttention(layers.Layer):
    """Attention over time dimension for each subband branch."""
    def __init__(self, return_attention=False, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time_steps, features)
        e = tf.nn.tanh(tf.tensordot(inputs, self.w, axes=[[2], [0]]))  # (batch, time_steps, 1)
        alpha = tf.nn.softmax(e, axis=1)  # attention weights
        context = tf.reduce_sum(alpha * inputs, axis=1)  # weighted sum

        if self.return_attention:
            return context, alpha
        return context

class AttentionLayer(layers.Layer): 
    """Additive attention over subband feature maps, returns both context and weights.""" 
    def __init__(self, return_attention=False, **kwargs): 
        super(AttentionLayer, self).__init__(**kwargs) 
        self.return_attention = return_attention 
    
    def build(self, input_shape): 
        self.w = self.add_weight(shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True) 
        super(AttentionLayer, self).build(input_shape) 

    def call(self, inputs): 
        # inputs: (batch, n_branches, feature_dim) 
        e = tf.nn.tanh(tf.tensordot(inputs, self.w, axes=[[2], [0]])) 
        alpha = tf.nn.softmax(e, axis=1) # attention weights 
        context = tf.reduce_sum(alpha * inputs, axis=1) # weighted sum 
        if self.return_attention: 
            return context, alpha # both return context
        return context



# ---------------------------------------------------------
#  Multi-Branch CNN Model
# ---------------------------------------------------------
class MultiBranchCNN:
    def __init__(self, input_shape, n_branches, n_classes, lr=1e-4):
        """
        Args:
            input_shape: (time_length, 1)
            n_branches: number of DWT/WPD subbands
            n_classes: number of output classes
        """
        self.input_shape = input_shape
        self.n_branches = n_branches
        self.n_classes = n_classes
        self.model = self._build_model(lr)
    
    def _build_model(self, lr=1e-4):
        # One input per subband
        inputs = []
        branches = []

        lr_schedule = ExponentialDecay(initial_learning_rate=3e-4, decay_steps=10000, decay_rate=0.9, staircase=True )

        for i in range(self.n_branches):
            inp = layers.Input(shape=self.input_shape, name=f"subband_{i}")
            # #x = layers.Conv1D(16, 5, activation='relu', padding='same')(inp)
            # x = Conv1D(8, 5, activation='relu', padding='same')(inp)
            # x = layers.BatchNormalization()(x)
            # x = layers.MaxPooling1D(2)(x)
            # #x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
            # x = layers.Conv1D(16, 5, activation='relu', padding='same')(x)
            # x = layers.BatchNormalization()(x)
            # x = TemporalAttention(return_attention=False, name=f"temporal_attention_{i}")(x)
            # branches.append(x)
            # inputs.append(inp)

            x = Conv1D(16, kernel_size=5, activation='relu')(inp)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(32, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = TemporalAttention(return_attention=False, name=f"temporal_attention_{i}")(x)
            branches.append(x)
            inputs.append(inp)
    

        merged = layers.Lambda(lambda tensors: tf.stack(tensors, axis=1), name="merged_subbands")(branches)

        attention_layer = AttentionLayer(return_attention=False, name="subband_attention")
        attention_out = attention_layer(merged)
        x = layers.Dense(16, activation='relu')(attention_out)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.n_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y, n_splits=5, epochs=20, batch_size=32, verbose=1):
        """
        Args:
            X: np.ndarray of shape (n_samples, n_branches, time_length)
            y: np.ndarray of integer labels
        """
        y = np.asarray(y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        histories = []
        fold = 1

        all_preds = np.zeros_like(y)
        all_true = np.zeros_like(y)

        #early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        for train_idx, val_idx in skf.split(X, y):
            print(f"\n----- Fold {fold} -----")
            fold += 1

            # B one input array per branch
            X_train = [X[train_idx, i, :][:, :, np.newaxis] for i in range(X.shape[1])]
            X_val   = [X[val_idx, i, :][:, :, np.newaxis]   for i in range(X.shape[1])]
            y_train, y_val = y[train_idx], y[val_idx]

            # Build new model for each fold
            self.model = self._build_model(lr=1e-4)
            
            hist = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                #callbacks=[early_stop]
            )
            histories.append(hist)

            # Predict validation fold
            y_pred = np.argmax(self.model.predict(X_val), axis=1)
            all_preds[val_idx] = y_pred
            all_true[val_idx] = y_val

        print("\nCross-validation complete.")
        return all_true, all_preds, histories


    # -------------------------
    #  Evaluation
    # -------------------------
    def evaluate(self, y_true, y_pred, label_encoder, label_decoder):
        y_true_labels = [label_decoder[i] for i in y_true]
        y_pred_labels = [label_decoder[i] for i in y_pred]

        acc = accuracy_score(y_true_labels, y_pred_labels)
        bal_acc = balanced_accuracy_score(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')

        report = classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=list(label_encoder.keys()),
            digits=3,
            output_dict=True
        )

        print("\nConfusion Matrix:\n",
              confusion_matrix(y_true_labels, y_pred_labels, labels=list(label_encoder.keys())))
        print("\nClassification Report:\n",
              classification_report(y_true_labels, y_pred_labels, digits=3))

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": report,
        }


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
#  2D CNN Classifier
# ---------------------------------------------------------


class CNN2DClassifier:
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.label_encoder = {lbl: i for i, lbl in enumerate(np.unique(y))}
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}

    # --- CNN architecture ---
    def build_model(self, input_shape, n_classes, lr=1e-3):
        model = models.Sequential([
            layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            #layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    # --- Training with StratifiedKFold ---
    def fit(self, epochs=30, batch_size=16, verbose=1):
        y_int = np.array([self.label_encoder[lbl] for lbl in self.y])
        y_true_all, y_pred_all = [], []

        input_shape = self.X.shape[1:]
        n_classes = len(self.label_encoder)
        print(f"Input shape: {input_shape}, Classes: {self.label_encoder}")

        for fold, (train_idx, test_idx) in enumerate(self.cv.split(self.X, y_int), 1):
            print(f"\n--- Fold {fold}/{self.n_splits} ---")

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = y_int[train_idx], y_int[test_idx]

            y_train_cat = to_categorical(y_train, num_classes=n_classes)
            y_test_cat = to_categorical(y_test, num_classes=n_classes)

            model = self.build_model(input_shape, n_classes)
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)
            ]

            model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks
            )

            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

            acc = accuracy_score(y_test, y_pred)
            print(f"Fold {fold} Accuracy: {acc:.3f}")

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        metrics = self.evaluate(y_true_all, y_pred_all)
        return y_true_all, y_pred_all, metrics

    # --- Evaluation ---
    def evaluate(self, y_true, y_pred):
        # Decode numeric labels
        y_true_labels = [self.label_decoder[i] for i in y_true]
        y_pred_labels = [self.label_decoder[i] for i in y_pred]

        acc = accuracy_score(y_true_labels, y_pred_labels)
        bal_acc = balanced_accuracy_score(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')

        report = classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=list(self.label_encoder.keys()),
            digits=3,
            output_dict=True
        )

        print("\nConfusion Matrix:\n",
            confusion_matrix(y_true_labels, y_pred_labels, labels=list(self.label_encoder.keys())))
        print("\nClassification Report:\n",
            classification_report(y_true_labels, y_pred_labels, digits=3))

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": report,
        }

