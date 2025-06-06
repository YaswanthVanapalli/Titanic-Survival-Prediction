
# 🚢 Titanic Survival Prediction using k-NN

This project predicts whether a passenger survived the Titanic shipwreck using the k-Nearest Neighbors (k-NN) algorithm. It is built using Python and uses data from the famous Titanic dataset on Kaggle.

## 📂 Project Overview

This project involves:

- Collecting and loading the Titanic dataset (train and test files)
- Exploring and understanding the data
- Cleaning and preparing the dataset for training
- Building a k-NN model
- Evaluating the model
- Making predictions on unseen data

## 💡 What is k-NN?

The **k-Nearest Neighbors (k-NN)** algorithm is a simple, intuitive machine learning technique that classifies data based on the majority class among its 'k' nearest neighbors.

## 🛠️ Tools & Libraries Used

- Python 🐍
- NumPy
- Pandas
- Scikit-learn (sklearn)
- Matplotlib (if used)
- Seaborn (if used)

## 📁 Dataset

The dataset used in this project is from Kaggle’s [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition. It contains information about passengers such as age, gender, ticket class, etc.

### Files:

- `train.csv` – Used to train the model
- `test.csv` – Used to test predictions

## 📊 Features Considered

Some of the features used for prediction:

- Passenger Class (`Pclass`)
- Gender (`Sex`)
- Age (`Age`)
- Number of Siblings/Spouses (`SibSp`)
- Number of Parents/Children (`Parch`)
- Fare (`Fare`)
- Port of Embarkation (`Embarked`)

## 🧹 Data Preprocessing Steps

- Handling missing values
- Converting categorical variables to numeric using encoding
- Feature scaling using StandardScaler
- Splitting dataset into training and validation sets

## 🤖 Model Training

The `k-Nearest Neighbors` algorithm was trained using the cleaned dataset. Model tuning (e.g. choosing the best value for `k`) was done to improve performance.

## 📈 Results

The model was evaluated based on accuracy and confusion matrix. The predictions were then submitted to Kaggle for scoring.

## 🚀 How to Run

1. **Clone this Repository**  
   If you have Git installed, open a terminal and run:
   ```bash
   git clone https://github.com/your-username/titanic-knn.git
   cd titanic-knn
2. Install Required Libraries
  Make sure Python is installed, then install the necessary libraries:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn
  ```
3. Download the Titanic Dataset

Download `train.csv` and `test.csv` from the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic/data) and place them in the same folder as the notebook.

4. Open the Notebook
  If using Jupyter:
  ```bash
  jupyter notebook "Titanic Survival Prediction using k-NN.ipynb"
  ```

  Or if you're using Google Colab, upload the notebook and dataset files there.

5. Run All Cells
  Go through each cell step-by-step, or use "Run All" to execute the entire notebook. The model will train and produce predictions.

6. Submit to Kaggle (Optional)
  The notebook will create a `.csv` file with predictions. You can submit it to Kaggle to see your accuracy on their leaderboard.


