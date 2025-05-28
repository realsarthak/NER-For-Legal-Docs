# 🧠 Named Entity Recognition for Indian Legal Documents

This project focuses on developing a custom Named Entity Recognition (NER) model to extract meaningful entities from Indian legal texts, such as case names, court names, judgment dates, and parties involved.

## 📂 Project Overview

This NER model is trained using [spaCy](https://spacy.io/) on a custom JSON dataset of legal documents containing:

- `case_name`
- `judgement_date`
- `question`
- `answer`

### 🚀 Features

- Custom NER model trained on legal domain
- Recognizes:
  - `PARTY` (names of people involved)
  - `COURT` (names of courts)
  - `DATE` (judgment dates)
  - `ORG` (legal bodies or forums)
- Trained using spaCy's config pipeline
- Easily extendable to more entities and documents

---


