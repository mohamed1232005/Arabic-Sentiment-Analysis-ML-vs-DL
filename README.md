# 📊 Arabic-Sentiment Analysis: ML vs DL

This repository contains the implementation of **Arabic sentiment analysis** using two approaches:

- **ML (Machine Learning):** Traditional pipeline with **TF-IDF + Classical Models** (Logistic Regression, Naive Bayes, Linear SVM).  
- **DL (Deep Learning):** Neural models (**RNN, LSTM, GRU**) with embeddings.  

The aim is to **compare ML and DL approaches** on the same dataset and evaluate their effectiveness.

---

## 📑 Table of Contents

- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
  - [Preprocessing](#preprocessing-ml)
  - [Feature Extraction](#feature-extraction-ml)
  - [Models](#models-ml)
  - [Evaluation Metrics](#evaluation-metrics-ml)
- [DL Pipeline](#-dl-pipeline)
  - [Preprocessing](#preprocessing-dl)
  - [Embedding & Tokenization](#embedding--tokenization)
  - [Models](#models-dl)
  - [Evaluation Metrics](#evaluation-metrics-dl)
- [Comparative Results](#-comparative-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Limitations](#-limitations)
- [Future Work](#-future-work)

---

## 🔍 Introduction

Sentiment analysis determines whether a review is **positive, neutral, or negative**.  
In this project:

- The **ML approach** applies handcrafted features (**TF-IDF**) with classical classifiers.  
- The **DL approach** uses **neural networks** with embeddings to capture sequential patterns in Arabic text.  

This enables a **direct comparison between ML baselines and DL architectures**.

---

## 📊 Dataset

- **Files:** `train.csv`, `test.csv`  
- **Columns:**  
  - `review_description` → text (Arabic reviews).  
  - `rating` → sentiment label.  

### 📌 Stats
- Train: **25,629 rows**  
- Test: **6,407 rows**  

| Label | Meaning   | Count (Train) | Count (Test) |
|-------|-----------|--------------:|-------------:|
| -1    | Negative  | 9,065         | 2,200        |
| 0     | Neutral   | 1,201         | 300          |
| 1     | Positive  | 15,363        | 3,907        |

⚠️ For **DL models**, the neutral class `0` is dropped → binary classification only.

---

## 🏛 ML Pipeline

### Preprocessing (ML)

- Lowercasing  
- Stopword removal (NLTK)  
- Lemmatization (spaCy)  

```python
import spacy
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    return " ".join(tokens)
```
### Feature Extraction (ML)

TF-IDF Vectorizer with unigrams + bigrams.

Equation:

𝑇
𝐹
𝐼
𝐷
𝐹
(
𝑡
,
𝑑
)
=
𝑇
𝐹
(
𝑡
,
𝑑
)
×
log
⁡
𝑁
1
+
𝐷
𝐹
(
𝑡
)
TFIDF(t,d)=TF(t,d)×log
1+DF(t)
N
	​
### Models (ML)

Logistic Regression

Naive Bayes

Linear SVM (best ML model)

### Evaluation Metrics (ML)

Accuracy

F1-score (macro, weighted)

Confusion Matrix

### 🧠 DL Pipeline
Preprocessing (DL)

Normalize Arabic characters

Remove punctuation

Encode labels (-1 → 0, 1 → 1)
### Embedding & Tokenization

Keras Tokenizer (20,000 vocab)

Pad sequences to length 100
### Models (DL)

All models: Embedding → Recurrent Layer → Dropout → Dense(sigmoid)
### Evaluation Metrics (DL)

Accuracy

Precision, Recall, F1-score

Confusion Matrix
## 📈 Comparative Results
| Model               | Classes | Accuracy | Macro F1 | Notes                    |
| ------------------- | :-----: | -------: | -------: | ------------------------ |
| Logistic Regression |    3    |   \~0.78 |   \~0.74 | Fast baseline            |
| Naive Bayes         |    3    |   \~0.72 |   \~0.68 | Lightweight              |
| Linear SVM          |    3    | **0.82** | **0.79** | Best ML model            |
| RNN                 |    2    |   \~0.84 |   \~0.83 | Struggles with long deps |
| LSTM                |    2    |   \~0.87 |   \~0.86 | Stable                   |
| GRU                 |    2    | **0.89** | **0.88** | Best DL model            |

✅ Conclusion: ML (Linear SVM) is a strong baseline, but DL (GRU) achieves the best performance overall.

## ⚙️ Installation
git clone https://github.com/<your-username>/Sentiment-Analysis-ML-vs-DL.git
cd Sentiment-Analysis-ML-vs-DL
pip install -r requirements.txt

## 🚀 Usage

Train & evaluate ML models:

jupyter notebook ML_Train.ipynb
jupyter notebook ML_Test.ipynb


Train & evaluate DL models:

jupyter notebook DL_Train.ipynb
jupyter notebook DL_Test.ipynb

##📂 Project Structure
Sentiment-Analysis-ML-vs-DL/
│
├── train.csv
├── test.csv
│
├── ML_Train.ipynb
├── ML_Test.ipynb
├── DL_Train.ipynb
├── DL_Test.ipynb
│
├── artifacts/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── rnn_model.h5
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   └── tokenizer.pkl
│
└── README.md

## 📦 Dependencies

Python 3.8+

pandas, numpy, matplotlib, seaborn

scikit-learn

nltk, spacy

tensorflow / keras

## ⚠️ Limitations

Neutral class imbalance

DL models simplify to binary classification

GPU needed for DL training

Pickle compatibility issues between environments

## 🔮 Future Work

Use pre-trained embeddings (AraVec, FastText).

Try transformers (AraBERT, DistilBERT).

Tune hyperparameters extensively.

Expand dataset for balanced classes.

Package as REST API for deployment.
