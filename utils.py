import matplotlib.pyplot as plt
import numpy as np
import torchvision
import seaborn as sns
from sklearn.metrics import classification_report  


# Plot matrice di confusione
def plot_confusion_matrix(cm, class_names, writer, global_step):
    fig, ax = plt.subplots(figsize=(40, 32))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    # Converti la figura in immagine NumPy
    fig.canvas.draw()
    cm_image = np.array(fig.canvas.renderer.buffer_rgba())

    # Logga l'immagine su TensorBoard
    writer.add_image("Confusion Matrix", cm_image, global_step=global_step, dataformats="HWC")
   
    plt.close(fig)  # Chiude la figura per evitare memory leak




# Plot e salvataggio della matrice di confusione, in cui viene evidenziato dove si sbaglia di più
def plot_confusion_matrix_with_errors(cm, class_names, writer, global_step):
    fig, ax = plt.subplots(figsize=(40, 32))
    
    # Creiamo una heatmap con colori diversi per diagonale ed errori
    mask_diagonal = np.eye(cm.shape[0], dtype=bool)  # Maschera per la diagonale
    mask_off_diagonal = ~mask_diagonal  # Maschera per gli errori
    
    # Disegna la diagonale in verde
    sns.heatmap(cm, mask=mask_off_diagonal, annot=True, fmt='d', cmap='Greens', 
                 xticklabels=class_names, yticklabels=class_names, cbar=False, ax=ax)
    
    # Disegna gli errori fuori diagonale in rosso
    sns.heatmap(cm, mask=mask_diagonal, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names, cbar=False, ax=ax)
    
    top_confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:  # Ignora i casi in cui la classe è corretta
                misclassification_count = cm[i][j]
                if misclassification_count > 0:  # Se c'è stata una confusione
                    top_confused_pairs.append((class_names[i], class_names[j], misclassification_count))
    
    # Ordina le coppie di classi in base al numero di errori
    top_confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Trova le celle con il massimo numero di errori fuori diagonale
    num_max_errors = 10  # Numero di errori più gravi da evidenziare
    
    # Mostra le prime 10 coppie di classi più confuse
    print("\nCoppie di classi più confuse:")
    for pair in top_confused_pairs[:num_max_errors]:
        print(f"Class {pair[0]} is often confused with {pair[1]} ({pair[2]} times)")
    
    # Evidenzia le coppie di classi più confuse
    for i, j, count in top_confused_pairs[:num_max_errors]:  # Prendi le prime n coppie più confuse
        idx_i = class_names.index(i)
        idx_j = class_names.index(j)
        ax.add_patch(plt.Rectangle((idx_j, idx_i), 1, 1, fill=False, edgecolor='red', lw=3))  # Bordo rosso

    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)
    plt.title('Confusion Matrix - Correct vs Misclassified', fontsize=20)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    # Converti la figura in immagine NumPy
    fig.canvas.draw()
    cm_image = np.array(fig.canvas.renderer.buffer_rgba())

    # Logga l'immagine su TensorBoard
    writer.add_image("Confusion Matrix - Correct vs Misclassified", cm_image, global_step=global_step, dataformats="HWC")
   
    plt.close(fig)  # Chiude la figura per evitare memory leak
    

# funzione per analizzare le performance del modello, valutando quali sono le classi che vengono maggiormente confuse
def analyze_class_performance(y_true, y_pred, cm, class_names, writer , top_n=5):
    
    # Calcola metriche per ogni classe
    TP = np.diag(cm)  # Veri Positivi
    FP = cm.sum(axis=0) - TP  # Falsi Positivi
    FN = cm.sum(axis=1) - TP  # Falsi Negativi
    TN = cm.sum() - (FP + FN + TP)  # Veri Negativi

    precision = np.divide(TP, (TP + FP + 1e-6))  # Precision per classe
    recall = np.divide(TP, (TP + FN + 1e-6))  # Recall per classe
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1-score per classe

    # Accuracy globale
    accuracy = np.sum(TP) / np.sum(cm)
    print(f"\nAccuracy globale: {accuracy:.2%}")

    # Mostra classification report per avere tutte le metriche
    print("\nDettagli per classe:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    ### aggiunta tensor board ###
    classif_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    report_text = "| Classe | Precisione | Recall | F1-Score | Supporto | Errori |\n"
    report_text += "|:------:|:----------:|:------:|:--------:|:--------:|:------:|\n"
    for i, class_name in enumerate(class_names):
        metrics = classif_report[class_name]
        errors = int(FN[i] + FP[i])  # Errori = falsi positivi + falsi negativi
        report_text += f"| {class_name} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1-score']:.2f} | {int(metrics['support'])} | {errors} |\n"

    writer.add_text('Performance/Classification_Report_with_Errors', report_text)
    
    
    sorted_classes = sorted(zip(class_names, precision, recall, f1_scores), key=lambda x: x[3], reverse=True)

    full_analysis_text = "**Classi ordinate per F1-score:**\n\n"
    
    full_analysis_text += "**Migliori classi (F1-score più alto):**\n"
    print("\nTop classi con migliore performance (F1-score più alto):")
    for i in range(min(top_n, len(sorted_classes))):
        full_analysis_text += f"- {sorted_classes[i][0]}: Precisione {sorted_classes[i][1]:.2f}, Recall {sorted_classes[i][2]:.2f}, F1 {sorted_classes[i][3]:.2f}\n"
        print(f"{sorted_classes[i][0]} - Precisione: {sorted_classes[i][1]:.2f}, Recall: {sorted_classes[i][2]:.2f}, F1-score: {sorted_classes[i][3]:.2f}")

    full_analysis_text += "\n**Peggiori classi (F1-score più basso):**\n"
    print("\nTop classi con peggiori performance (F1-score più basso):")
    for i in range(min(top_n, len(sorted_classes))):
        full_analysis_text += f"- {sorted_classes[-(i+1)][0]}: Precisione {sorted_classes[-(i+1)][1]:.2f}, Recall {sorted_classes[-(i+1)][2]:.2f}, F1 {sorted_classes[-(i+1)][3]:.2f}\n"
        print(f"{sorted_classes[-(i+1)][0]} - Precisione: {sorted_classes[-(i+1)][1]:.2f}, Recall: {sorted_classes[-(i+1)][2]:.2f}, F1-score: {sorted_classes[-(i+1)][3]:.2f}")


    # Identifica le classi più frequentemente confuse
    most_confused = np.argmax(cm, axis=1)  # Trova per ogni classe la predizione errata più comune

    full_analysis_text += "\n## Classi Più Frequentemente Confuse\n\n"
    print("\n Classi più frequentemente confuse:")
    for i, cls in enumerate(class_names):
        if most_confused[i] != i:  # Se non è se stessa
            print(f"{cls} viene spesso scambiata con {class_names[most_confused[i]]}")
            full_analysis_text += f"- {cls} → confusa con **{class_names[most_confused[i]]}**\n"


    # Trova le classi che sono confuse più frequentemente
    max_errors = {}
    for i, class_name in enumerate(class_names):
        errors = FN[i] + FP[i]  # Falsi Positivi + Falsi Negativi per ogni classe
        max_errors[class_name] = errors

    # Ordina le classi in base agli errori
    sorted_max_errors = sorted(max_errors.items(), key=lambda x: x[1], reverse=True)

    print("\nClassi più confuse:")
    full_analysis_text += "\n## Classi Più Confuse (Totale Errori)\n\n"
    
    for class_name, errors in sorted_max_errors[:10]:  # Mostra le prime 10 classi con più errori
        print(f"Class {class_name}: {errors} errori")
        full_analysis_text += f"- {class_name}: {errors} errori\n"
    
    writer.add_text('Performance/Full_Analysis', full_analysis_text)
    
    
