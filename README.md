# Deep Learning - Classificazione Stanford Cars

## 📚 Descrizione del Progetto

Questo progetto si concentra sulla classificazione di immagini di automobili utilizzando il dataset [Stanford Cars](https://pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html), che contiene 16.185 immagini suddivise in 196 classi. L'obiettivo è sviluppare un modello di deep learning in grado di identificare correttamente la marca, il modello e l'anno di un'auto a partire da un'immagine.

## 🗂️ Struttura del Repository

```
DeepLearning/
├── dataset/                    # Directory per il dataset Stanford Cars
│   ├── Stanford_Cars           # Dataset stanford
│   ├── dataset_analysis.ipynb  # Per analizzare il dataset
│   └── dataset.py              # Caricamento del dataset
│
├── runs/                       # Salvataggio delle informazioni per visualizzare i risultati su TensorBoard
│   ├── Name_Run [...]         
│   └── visual_runs_tensorboard.ipynb
│
├── best_models/                # Salvataggio dei migliori modelli addestrati
│   ├── best_model_Adam.pth     # Miglior modello con ottimizzatore Adam
│   └── best_model_SGD.pth      # Miglior modello con ottimizzatore SGD
│        
├── hpc_test.py                 # Per addestramento e valutazione 
├── utils.py                    # Funzioni di utilità
├── solver.py                   # Funzioni per l'addestramento, la valutazione e il test del modello
├── model.py                    # Reti pre-addestrate
├── inference.ipynb             # Inferenza con i best_models
├── requirements.txt            # Dipendenze del progetto
└── README.md                   # Informazioni sul progetto
```

## ⚙️ Installazione

1. **Clona il repository:**

   ```bash
   git clone https://github.com/ariannaCella/StanfordCars_DeepLearningProject.git
   ```

2. **Crea un ambiente virtuale (opzionale ma consigliato):**
   ```bash
   conda create -n myenv_stanford_cars python=3.9 -y
   conda activate myenv_stanford_cars
   ```

3. **Installa le dipendenze:**

   ```bash
   conda install numpy matplotlib seaborn scikit-learn pytorch torchvision -c pytorch
   ```
   Facoltativi:
   ```bash
   pip install kagglehub tensorboard
   ```

## 📦 Dataset (Stanford Cars)

Hai due opzioni:

### 1. **Download automatico da Kaggle**

```python
import kagglehub
path = kagglehub.dataset_download("rickyyyyyyy/torchvision-stanford-cars")
```

Usa `--dataset_path /kaggle/input/torchvision-stanford-cars` per puntare al dataset (path restituito dal precedente import).

---

### 2. **Download manuale**

Scarica da:  
https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars

Estrai i file nella directory:

```
dataset/stanford_cars/
```
Usa `--dataset_path dataset/stanford_cars` per puntare al dataset.

---

### Analisi del Dataset

Usando il notebook  `dataset_analysis.ipynb` è possibile esplorare il dataset, visualizzare immagini e analizzare il bilanciamento tra classi.

---
## 🚀 Avvio addestramento

```bash
python hpc_test.py \
  --model_name run_resnet18 \
  --run_name run_resnet18 \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.00001 \
  --dataset_path /kaggle/input/torchvision-stanford-cars \
  --opt Adam \
  --aug \
  --seed 42
```

---

## 💾 Output

- Modello salvato come `stanford_net_<model_name>.pth`
- Log salvati in `runs/`
  
---

## 📈 TensorBoard (opzionale)

TensorBoard può essere usato per visualizzare i risultati graficamente: Training Loss, Validation Loss, Validation Accuracy, Test Accuracy, Matrice di Confusione e analisi delle classi maggiormente confuse.

```bash
tensorboard --logdir=runs
```
Oppure eseguire il notebook `visual_runs_tensorboard.ipynb` su Colab.

---
## 🚀 Avvio inferenza

Per eseguire l'inferenza è possibile usare il notebook  `inference.ipynb`.


