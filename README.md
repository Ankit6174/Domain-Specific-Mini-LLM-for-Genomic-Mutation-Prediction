# Domain-Specific Mini-LLM for Genomic Mutation Prediction

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction/deploy.yml?style=for-the-badge)
![GitHub top language](https://img.shields.io/github/languages/top/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction?style=for-the-badge)

A transformer-based, multi-task learning model that predicts the type, genomic position, and clinical impact of DNA mutations directly from raw sequence data.

###  [**View Live Demo**](https://dna-mutation-prediction.onrender.com/) 

![Project Screenshot](https://i.imgur.com/your-screenshot-url.png)  ---

## The Problem

Predicting the clinical significance of DNA mutations is a major challenge in genomics. The sheer volume of genetic data and sequence variability makes it difficult for researchers and clinicians to quickly identify which mutations might be pathogenic. This project aims to solve this by using a domain-specific transformer model to provide fast, accurate predictions, accelerating research and diagnostics.

## Key Features

* **Multi-Task Prediction:** The model simultaneously predicts:
    * **Clinical Significance:** (e.g., Pathogenic, Benign)
    * **Mutation Type:** The specific base change (e.g., A → T)
    * **Chromosome & Genomic Position:** The exact location of the mutation.
* **Interactive Web Interface:** A user-friendly web app to input a DNA sequence and receive detailed, visualized results.
* **Data Visualization:** Displays key metrics like GC/AT content, nucleotide frequency, and sliding window analysis.
* **Full-Stack Application:** Features a complete frontend, backend server, and database integration for a seamless user experience.

---

## Tech Stack

This project is a full-stack application built with a modern technology stack:

| Category      | Technology                                                                                                                                                                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Frontend** | ![HTML](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![SASS](https://img.shields.io/badge/SASS-hotpink.svg?style=for-the-badge&logo=SASS&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) |
| **Backend** | ![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white) ![Express.js](https://img.shields.io/badge/express.js-%23404d59.svg?style=for-the-badge&logo=express&logoColor=%2361DAFB) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
| **ML/DS** | ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) |
| **Database** | ![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)                                                                                                                                         |
| **Deployment**| ![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)           |

---

## Getting Started: Local Installation

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Node.js and Python installed on your machine.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction.git](https://github.com/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction.git)
    cd Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction
    ```

2.  **Set up the Node Server (Frontend & API):**
    ```sh
    cd "Node Server"
    npm install
    npm run dev
    ```
    This will start the frontend server, typically on `http://localhost:8001`.

3.  **Set up the Python Server (ML Model):**
    ```sh
    cd "Python Server"
    pip install -r requirements.txt  # You may need to create this file
    python server.py
    ```
    This will start the Flask server for the model, typically on `http://localhost:5000`.

---

## Author

**Ankit Ahirwar**


* [GitHub](https://github.com/Ankit6174)
* [LinkedIn](https://www.linkedin.com/in/Ankit6174) 

### **Final Action Item for You:**

1.  **Get a Screenshot:** Take a nice screenshot of your live application.
2.  **Upload to an Image Host:** Upload the screenshot to a site like [Imgur](https://imgur.com/upload).
3.  **Update the URL:** Replace `https://i.imgur.com/your-screenshot-url.png` in the template with your actual screenshot URL.