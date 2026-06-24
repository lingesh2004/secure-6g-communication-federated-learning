from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy:", acc)
    print("Confusion Matrix:")
    print(cm)
