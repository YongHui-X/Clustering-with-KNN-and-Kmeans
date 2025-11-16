# Customer Segmentation Using K-Means Clustering

## Project Overview
This project demonstrates **customer segmentation** for a credit card dataset using **K-Means clustering**. The dataset contains customer demographics, income, employment, debt, and credit card default status. Missing entries in the `Defaulted` column are handled using a **K-Nearest Neighbors (KNN) classifier**.  

---

## Dataset
`cust_seg.csv` contains the following columns:  

- `Age` – Customer’s age  
- `Edu` – Education level  
- `Years Employed` – Number of years employed  
- `Income` – Annual income  
- `Card Debt` – Credit card debt  
- `Other Debt` – Other outstanding debts  
- `Defaulted` – Default status (0 = Non-default, 1 = Default)  
- `DebtIncomeRatio` – Ratio of total debt to income  

---

## Key Steps

1. **Data Cleaning & Preprocessing**
   - Removed the `Customer Id` column.  
   - Stripped whitespace from column names.  

2. **Handling Missing Data**
   - Predicted missing `Defaulted` values using a **KNN classifier** based on numerical features.  
   - Ensured a complete dataset for clustering and plotting.  

3. **Exploratory Data Analysis (EDA)**
   - Plotted **4 histograms** (`Age`, `Income`, `Years Employed`, `DebtIncomeRatio`) coloured by default status with **KDE lines**.  
   - Created a **bar chart** showing the number of defaulted vs non-defaulted customers.  

4. **Feature Scaling**
   - Standardized all numerical features using **StandardScaler** for KNN and K-Means.  

5. **Clustering**
   - Applied **K-Means** with 4 clusters on the scaled dataset.  
   - Visualized cluster assignments in a **3D plot** using `Age`, `Income`, and `DebtIncomeRatio`.  

---

## Key Insights
- 3D clustering shows natural groupings of customers based on age, income, and debt-to-income ratio.  
- Clusters reveal patterns such as:  
  - Younger customers with low debt  
  - High-income customers with high debt  
  - Customers more likely to default

### Before K-Means:
<img width="1280" height="612" alt="clustering histogram" src="https://github.com/user-attachments/assets/c3423634-804a-4075-9957-f3655229774d" />

### After k-means:
<img width="1280" height="612" alt="clustering ws diagram" src="https://github.com/user-attachments/assets/2c94f543-637c-4874-a474-b8d25ddc45f3" />
<img width="640" height="480" alt="clustering_defaultcases" src="https://github.com/user-attachments/assets/1ee9b4f0-cf9b-400e-b191-62104a36be83" />


---

## Libraries Used
- `pandas` – Data manipulation  
- `numpy` – Numerical computations  
- `matplotlib` & `seaborn` – Data visualization  
- `scikit-learn` – StandardScaler, KNN, K-Means  

---

## How to Run
1. Ensure `cust_seg.csv` is in the project directory.  
2. Install required packages:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
