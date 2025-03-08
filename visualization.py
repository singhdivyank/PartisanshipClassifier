import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

def value_plots(values_df, fig_name):
    """
    value counts of issue labels
    """
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(values_df["issue_label"].astype(str), values_df["count"], color='skyblue')
    plt.xlabel("Issue Labels")
    plt.ylabel("Count")
    plt.title("Value Counts of Issue Labels")
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, str(height), ha='center', va='bottom')

    plt.show()
    plt.savefig(fig_name)
    print(f"saved value plots to :: {fig_name}")

def plot_hist(probabilities, fig_name, title):
    """
    histogram for posterior probabilities using seaborn
    :param probabilities: Pandas Series representing posterior probabilities
    """
    
    # set plot size
    plt.figure(figsize=(10, 6))
    # plot histogram
    sns.histplot(probabilities, bins=20, kde=True, color="green", alpha=0.7, edgecolor="black")
    # add vertical lines
    plt.axvline(x=0.7, color="blue", linestyle="--", linewidth=2, label="p>0.7")
    plt.axvline(x=0.4, color="red", linestyle="--", linewidth=2, label="p<0.4")
    # title
    plt.title(title, fontsize=16)
    # axis labels
    plt.xlabel("Probability", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    # formatting
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # show plot
    plt.show()
    # save plot
    plt.savefig(fig_name)

    print(f"saved histogram {fig_name}")


class CreatePlots:
    def __init__(self, y_true, y_preds, y_scores, num_classes) -> None:
        self.y_true = y_true
        self.y_preds = y_preds
        self.y_scores = y_scores
        self.num_classes = num_classes
    
    def create_plots(self):
        self.plot_roc()
        self.plot_pr()
        self.plot_cm()
    
    def plot_roc(self):
        """
        plot the ROC curve
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("ROC Curve", fontsize=16)
        
        for i in range(min(2, self.num_classes)):
            fpr, tpr, _ = roc_curve((np.array(self.y_true) == i).astype(int), self.y_scores[:, i])
            axes[i].plot(fpr, tpr, label=f"Class: {i}")
            axes[i].plot([0, 1], [0, 1], 'k--')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f"Class: {i}")
            axes[i].legend(loc='lower right')
            axes[i].grid()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        fig.savefig("roc.png")

        print("saved ROC curve")
    
    def plot_pr(self):
        """
        plot PR curve
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("PR Curve", fontsize=16)
        
        for i in range(min(2, self.num_classes)):
            precision, recall, _ = precision_recall_curve((np.array(self.y_true) == i).astype(int), self.y_scores[:, i])
            axes[i].plot(recall, precision, label=f"Class: {i}")
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f"Class: {i}")
            axes[i].legend(loc='lower left')
            axes[i].grid()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        fig.savefig("pr.png")

        print("saved PR curve")

    def plot_cm(self):
        """
        confusion matrix
        """

        cm = confusion_matrix(self.y_true, self.y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        _, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
        plt.savefig('cm.png')

        print("saved Confusion Matrix")
