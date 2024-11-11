from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def main():
    # 0 = dog, 1 = cat, 2 = bird
    data_labels = ['dog', 'cat', 'bird']
    # "Run" 4 images of dogs, 4 of cats, and 4 of birds
    actual_results = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    prediction = [2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 0, 2]
    ideal_results = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    print(f'Accuracy = {accuracy_score(actual_results, prediction)*100:.2f}%')
    print(f'Confusion matrix:\n{confusion_matrix(actual_results, prediction)}')
    print(f'Classification Report:\n{classification_report(
        actual_results, prediction, target_names=data_labels)}')

    # Generate a confusion matrix for Ideal Matrix
    cm = confusion_matrix(actual_results, prediction)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=data_labels)
    disp.plot()
    plt.title(' Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
