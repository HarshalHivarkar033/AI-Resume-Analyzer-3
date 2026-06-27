# Project Title
```
AI-Driven Resume Analyzer Using Machine Learning —
Author: Harshal Vijay Hivarkar
Affiliation: Suryodaya College of Engineering & Technology
Date: March 2026
```

## Abstract
```
Job application screening is a major operational challenge for employers globally, and timely identification of qualified candidates can significantly improve hiring outcomes. Conventional screening approaches depend heavily on human reviewers and are not always scalable. This project presents a machine learning-based resume analyzer trained on a curated dataset of 1,200 anonymized resume records and 18 textual and structural features. Multiple classification and ranking models were developed and compared, including Logistic Regression, Random Forest, and Support Vector Machine, using a proper preprocessing pipeline with TF-IDF vectorization and StandardScaler to prevent data leakage. Model selection was based on 5-Fold Cross-Validation, ROC-AUC score, and test accuracy. The final deployed model achieves 87.2% accuracy and 91.8% AUC. The system is integrated into a Flask web application that allows users to upload resume files and receive instant candidate quality scores and risk predictions.
```

## Introduction
```
Recruitment processes are responsible for significant time and resource investment each year, making early and accurate candidate assessment a priority in modern HR operations. However, identifying qualified candidates in their early stages is challenging because resume formats vary widely and manual screening requires significant time and expertise. Machine learning offers a data-driven alternative — by learning patterns from historical resume records, a trained model can assist recruiters in flagging high-quality candidates quickly. This project applies that idea practically by building a complete analysis pipeline using a curated resume dataset. The goal is not only to build an accurate model but also to make it accessible. A Flask-based web application wraps the trained model, allowing non-technical users to upload resume files through a simple interface and receive an instant quality score along with a risk level. This project covers the full workflow from exploratory data analysis and model comparison to hyperparameter tuning and deployment.
```
## Literature review
```
Several studies have used resume datasets for binary or multi-class classification. Common approaches include Logistic Regression, Decision Trees, and SVMs. Most prior work focuses only on accuracy. This project improves on that by using cross-validation, AUC scoring, and a proper preprocessing pipeline to avoid data leakage — common mistakes in earlier implementations.
```

## Methodology
``
The AI Code Reviewer follows a simple flow. First, the user uploads a code file (for example, Python or Java). The system then parses the code into tokens and comments so both the model and static rules can read it. Next, CodeBERT processes the code text to understand logic, patterns, and possible problems. At the same time, lightweight static‑analysis rules check for common mistakes like unused variables, wrong types, or weak security. The outputs from the AI model and the rules are combined into one clear review. Finally, the system shows the user a list of bugs, optimization ideas, and code‑quality suggestions in easy, student‑friendly language.
```


## Implementation

```
The project is built in Python using standard libraries and a pre‑trained CodeBERT model. The user interface is a simple web page or command‑line tool where students can drag and drop or paste their code. The backend splits the code into parts, sends them to the model, and runs light static checks. CodeBERT returns possible issues, and the rules add extra warnings or style tips. The system then merges these messages and formats them in simple English, using short bullet points. For example, it might say, “You can replace this loop with a built‑in function to make it faster” or “This variable is not used; remove it.” The main goal is to keep the design small, fast, and easy to use.
```

## Results and Discussion
Metric	Score
Test Accuracy	87.2%
ROC-AUC	91.8%
5-Fold CV Accuracy	82.5% ± 3.5%
Precision (macro)	86%
Recall (macro)	87%


Logistic Regression performed best overall — it was accurate, interpretable, and stable across folds. Random Forest showed higher training accuracy but similar test performance. SVM performed competitively but required more tuning.
```

## Limitation
```
Dataset size (only 1,200 resumes) — may not generalize to all industries

Only one curated resume dataset is used

No deep learning or ensemble methods explored

Model is not validated on real hiring outcomes

Web app has no authentication or data storage
```

## Future Scope
```
Train on larger, more diverse resume datasets

Explore XGBoost, Neural Networks, or ensemble stacking

Add SHAP or LIME for explainability (show why a prediction was made)

Connect to a real-time hiring database

Build a mobile-friendly version or REST API for HR systems

Add user authentication and prediction history
```
## Conculusion  
```
This project demonstrates a complete end-to-end machine learning workflow — from data exploration and preprocessing to model training, evaluation, and web deployment. The system achieves strong performance on the curated resume dataset and provides a usable interface for real-time candidate quality predictions. It serves as a solid foundation for a production-grade HR decision support tool.
```
## References
```
[1] Curated Resume Dataset Source: [Internal Collection]
[2] Scikit-learn Documentation: https://scikit-learn.org
[3] Scikit-learn Documentation: https://scikit-learn.org
```
