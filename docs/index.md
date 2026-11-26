# RAIN RAIN GO AWAY OR GO HERE
### Will Los Baños get at least 5 mm of rain tomorrow?

**CMSC 170 – Introduction to Artificial Intelligence**  
**Group FEBS**

- Enzo Joyosa
- Franz Jeremy Ellao
- Brylle Pamulaklakin
- Szhan Wayne Timosan

---

## 1. Title and Group Members

# RAIN RAIN GO AWAY OR GO HERE
### Will Los Baños get at least 5 mm of rain tomorrow?

This project is developed by Group FEBS as part of CMSC 170: Introduction to Artificial Intelligence at the University of the Philippines Los Baños.

**Group Members:**
- Enzo Joyosa
- Franz Jeremy Ellao
- Brylle Pamulaklakin
- Szhan Wayne Timosan

---

## 2. Problem Statement and Motivation

Rainfall affects everyday activities, transportation, agriculture, and planning in Los Baños, Laguna. Knowing whether rain is likely to occur the following day can help people make more informed decisions about their daily activities.

This project aims to predict whether Los Baños will receive at least 5 mm of rainfall on the following day using historical daily weather data. The problem is framed as a binary classification task: predicting whether the next day's rainfall will meet or exceed the 5 mm threshold.

We aim to compare different Artificial Intelligence and machine learning methods implemented from scratch and evaluate their performance in predicting next-day rainfall. Through this comparison, we seek to understand the strengths, limitations, and performance differences of the selected methods in handling weather-related data.

---

## 3. Dataset Description

*To be completed.*

- Dataset source
- Dataset size
- Features
- Target variable
- Data preprocessing

---

## 4. Methods

*To be completed.*

- Majority baseline
- Persistence baseline
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Perceptron

---

## 5. Experimental Setup

*To be completed.*

- Data splitting
- Training and validation process
- Test set
- Evaluation metrics
- Hyperparameters

---

## 6. Results

### Validation Set (2018–2021)

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority (always dry) | 0.5818 | 0.0000 | 0.0000 | 0.0000 |
| Persistence (same as today) | 0.7426 | 0.6923 | 0.6923 | 0.6923 |
| KNN | 0.7337 | 0.6934 | 0.6514 | 0.6717 |
| Naive Bayes | 0.7050 | 0.6304 | 0.7119 | 0.6687 |
| **Perceptron** | **0.7611** | **0.7266** | **0.6874** | **0.7065** |

### Test Set (2022–2025)

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority (always dry) | 0.5199 | 0.0000 | 0.0000 | 0.0000 |
| Persistence (same as today) | 0.7603 | 0.7504 | 0.7504 | 0.7504 |
| KNN | 0.7363 | 0.7423 | 0.6904 | 0.7154 |
| Naive Bayes | 0.7336 | 0.6950 | 0.7932 | 0.7408 |
| **Perceptron** | **0.7678** | **0.7586** | **0.7575** | **0.7580** |

Figures (confusion matrices and F1 bar chart) are in the `figures/` and `figures/test/` folders of the repository.

---

## 7. Discussion

### Performance Analysis

The Perceptron achieved the highest F1 on both the validation set (0.7065) and the test set (0.7580), making it the best-performing method in this project. Its linear decision boundary, combined with averaging over all weight updates, produced a stable and well-calibrated classifier.

The Persistence baseline — predicting tomorrow's weather will be the same as today's — was surprisingly strong (validation F1 0.6923, test F1 0.7504). This reflects the autocorrelated nature of rainfall: rainy days tend to cluster together during the wet season. Any learned method must beat this baseline to be useful, and all three did on the test set.

KNN (validation F1 0.6717, test F1 0.7154) performed moderately well. Its main weakness is that it relies on distance in a 14-dimensional space, where features with different scales can dominate unless carefully normalised. It was also the slowest method (~3.7 s on validation), since it computes distances to every training sample at prediction time.

Naive Bayes (validation F1 0.6687, test F1 0.7408) had the highest recall on the test set (0.7932), meaning it caught the most actual rainy days. However, its precision was lower, indicating more false alarms. The binning of continuous features introduces information loss, and the conditional independence assumption does not hold perfectly for correlated weather variables such as temperature and humidity.

### Limitations

- The dataset covers only one location (Los Baños, Laguna). Results may not generalise to other areas.
- Features are limited to daily aggregates from NASA POWER. Sub-daily patterns (e.g. afternoon convective storms) are not captured.
- The 5 mm threshold is fixed. A different threshold would change the label balance and all reported scores.
- The Perceptron is a linear model and cannot capture non-linear interactions between features without feature engineering.

### Possible Improvements

- Tuning hyperparameters (K for KNN, number of bins for Naive Bayes, epochs for the Perceptron) on the validation set more systematically.
- Adding lag features (yesterday's rainfall, 3-day rolling mean) to give the models short-term memory.
- Trying a non-linear method such as a decision tree or multi-layer perceptron as a future extension.

---

## 8. Conclusion

This project compared three machine learning methods — K-Nearest Neighbors, Naive Bayes, and the averaged Perceptron — on the task of predicting whether Los Baños, Laguna will receive at least 5 mm of rain the following day, using 25 years of NASA POWER daily weather data (2001–2025).

All three methods were implemented from scratch using only Python's standard library, NumPy, and Matplotlib, and evaluated with a strict time-based split to prevent data leakage.

The Perceptron was the strongest method, achieving a test F1 of 0.7580, surpassing both baselines and the other two classifiers. Naive Bayes was the best at catching rainy days (highest recall), while KNN offered a middle ground. All three methods beat the majority baseline by a large margin, and all three beat the persistence baseline on the test set — confirming that learning from historical weather features adds genuine predictive value beyond simply repeating today's observation.

The project demonstrated that simple, interpretable models can perform well on weather prediction when given clean, well-engineered features and a fair evaluation setup.

---

## 9. References and Acknowledgments

### Dataset Source

NASA POWER (Prediction of Worldwide Energy Resources).  
https://power.larc.nasa.gov/

The daily weather dataset used in this project was obtained from NASA POWER:

> The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth Science Division.

### AI Tool Disclosure

AI tools, including Claude (claude.ai) by Anthropic and ChatGPT by OpenAI, were used as assistance during the development of this project for guidance, troubleshooting, documentation, and writing support. All code, implementations, experimental results, and conclusions are subject to review and verification by the project members.

### Other References

*Additional references, documentation, and sources used in the implementation and analysis will be listed here.*

---

## 10. Repository and Presentation

### GitHub Repository

[CMSC 170 Project Repository](https://github.com/szhan17/cmsc170-project)

### Presentation Slides

*Presentation slides link will be added here.*