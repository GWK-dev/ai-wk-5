# deep_learning_tensorflow.py - MNIST Classification with CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        print("📊 Loading MNIST Dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_cnn_model(self):
        print("🏗️ Building CNN Model Architecture...")
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model architecture built!")
        model.summary()
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=5):
        print("🚀 Training CNN Model...")
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        print("✅ Model training completed!")
    
    def evaluate_model(self, x_test, y_test):
        print("📈 Evaluating Model Performance...")
        
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"📉 Test Loss: {test_loss:.4f}")
        
        if test_accuracy > 0.95:
            print("🎉 Target achieved: Accuracy > 95%!")
        else:
            print("⚠️ Target not achieved: Accuracy < 95%")
        
        return test_accuracy, test_loss, y_pred_classes, y_true_classes
    
    def visualize_results(self, x_test, y_true_classes, y_pred_classes):
        print("🎨 Generating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Sample predictions
        correct_indices = np.where(y_true_classes == y_pred_classes)[0]
        incorrect_indices = np.where(y_true_classes != y_pred_classes)[0]
        
        for i in range(3):
            if i < len(correct_indices):
                idx = correct_indices[i]
                axes[1, 1].imshow(x_test[idx].reshape(28, 28), cmap='gray')
                axes[1, 1].set_title(f'Correct: True={y_true_classes[idx]}, Pred={y_pred_classes[idx]}')
                break
        
        plt.tight_layout()
        plt.savefig('mnist_cnn_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, epochs=5):
        (x_train, y_train), (x_test, y_test) = self.load_and_preprocess_data()
        self.build_cnn_model()
        self.train_model(x_train, y_train, x_test, y_test, epochs)
        test_accuracy, test_loss, y_pred_classes, y_true_classes = self.evaluate_model(x_test, y_test)
        self.visualize_results(x_test, y_true_classes, y_pred_classes)
        
        return test_accuracy, test_loss

if __name__ == "__main__":
    mnist_classifier = MNISTClassifier()
    accuracy, loss = mnist_classifier.run_complete_analysis(epochs=5)
    
    print("\n" + "="*50)
    print("🎉 DEEP LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final Results: Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")