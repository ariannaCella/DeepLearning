# Deep Learning - Classificazione delle Auto con Stanford Cars

![Esempio di immagine del dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## 📚 Descrizione del Progetto

Questo progetto si concentra sulla classificazione di immagini di automobili utilizzando il dataset [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), che contiene 16.185 immagini suddivise in 196 classi. L'obiettivo è sviluppare un modello di deep learning in grado di identificare correttamente la marca, il modello e l'anno di un'auto a partire da un'immagine.

## 🗂️ Struttura del Repository

```
DeepLearning/
├── dataset/                 # Directory per il dataset Stanford Cars
│   ├── cars_train/
│   ├── cars_test/
│   ├── devkit/
│   └── ...
├── notebooks/               # Notebook Jupyter per esplorazione e addestramento
│   └── car_classification.ipynb
├── models/                  # Modelli salvati
├── utils/                   # Funzioni di utilità
├── requirements.txt         # Dipendenze del progetto
└── README.md                # Questo file
```

## 🔧 Installazione

1. **Clona il repository:**

   ```bash
   git clone https://github.com/ariannaCella/DeepLearning.git
   cd DeepLearning
   ```

2. **Crea un ambiente virtuale (opzionale ma consigliato):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   ```

3. **Installa le dipendenze:**

   ```bash
   pip install -r requirements.txt
   ```

## 📅 Preparazione del Dataset

A causa delle dimensioni del dataset (oltre 1 GB), non è incluso direttamente nel repository. Segui questi passaggi per scaricarlo e prepararlo:

1. **Scarica il dataset Stanford Cars:**

   - Vai alla pagina ufficiale: [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
   - Scarica i seguenti file:
     - `cars_train.tgz`
     - `cars_test.tgz`
     - `cars_train_annos.mat`
     - `cars_test_annos.mat`
     - `devkit.tar`

2. **Estrai i file scaricati:**

   - Posiziona tutti i file scaricati nella directory `dataset/` del repository.
   - Estrai i file `.tgz` e `.tar`:

     ```bash
     tar -xvzf cars_train.tgz -C dataset/
     tar -xvzf cars_test.tgz -C dataset/
     tar -xvzf devkit.tar -C dataset/
     ```

3. **Verifica la struttura della directory `dataset/`:**

   Dopo l'estrazione, la struttura dovrebbe essere simile a questa:

   ```
   dataset/
   ├── cars_train/
   ├── cars_test/
   ├── devkit/
   ├── cars_train_annos.mat
   └── cars_test_annos.mat
   ```

## 🚀 Esecuzione del Progetto

1. **Avvia Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

2. **Apri il notebook:**

   - Naviga nella directory `notebooks/` e apri `car_classification.ipynb`.

3. **Segui le istruzioni nel notebook:**

   - Il notebook guida attraverso il processo di caricamento dei dati, pre-processing, definizione del modello, addestramento e valutazione.

## 🧠 Architettura del Modello

Il modello utilizza una rete neurale convoluzionale (CNN) basata su architetture pre-addestrate come ResNet50 o EfficientNet, con tecniche di fine-tuning per adattarsi al dataset specifico. L'addestramento include strategie di data augmentation e ottimizzazione per migliorare le prestazioni.

## 📊 Risultati Attesi

Dopo l'addestramento, il modello dovrebbe raggiungere un'accuratezza elevata nella classificazione delle immagini delle auto. I risultati dettagliati, inclusi grafici di accuratezza e perdita, sono disponibili nel notebook.

## 📌 Note Importanti

- **Dataset:** Assicurati di scaricare e posizionare correttamente il dataset nella directory `dataset/` come descritto sopra.
- **Ambiente:** È consigliato utilizzare un ambiente virtuale per evitare conflitti di dipendenze.
- **GPU:** L'addestramento del modello può richiedere l'uso di una GPU per tempi di esecuzione ragionevoli.

## 📄 Licenza

Questo progetto è distribuito sotto la licenza MIT. Consulta il file [LICENSE](LICENSE) per ulteriori dettagli.

---

Se hai domande o suggerimenti, sentiti libero di aprire un'issue o contattare direttamente. Buon divertimento con il deep learning! 🚗💻
