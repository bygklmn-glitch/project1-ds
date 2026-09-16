import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


n = 100
np.random.seed(42)


data = {
    "Student_ID": range(1, n+1),
    "Study_Hours": np.random.randint(1, 8, n),          # 1–7 hours
    "Attendance": np.random.randint(40, 100, n),        # 40–99 %
    "Previous_Marks": np.random.randint(30, 100, n),    # 30–99 marks
    "Assignments": np.random.randint(20, 100, n),       # 20–99 marks
    "Internal_Marks": np.random.randint(20, 100, n),    # 20–99 marks
}


final_result = []
for i in range(n):
    if (data["Study_Hours"][i] >= 4 and 
        data["Attendance"][i] >= 60 and 
        data["Previous_Marks"][i] >= 50):
        final_result.append("Pass")
    else:
        final_result.append("Fail")


data["Final_Result"] = final_result


df = pd.DataFrame(data)
df.to_csv("student_data.csv", index=False)
print("✅ student_data.csv created with 100 records")
print(df.head())


data = pd.read_csv("student_data.csv")


data['Final_Result'] = data['Final_Result'].map({'Pass':1, 'Fail':0})


X = data[['Study_Hours','Attendance','Previous_Marks','Assignments','Internal_Marks']]
y = data['Final_Result']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


sns.scatterplot(x=data['Study_Hours'], y=data['Previous_Marks'], hue=data['Final_Result'])
plt.title("Study Hours vs Previous Marks")
plt.show()


new_student = [[5, 80, 70, 75, 68]] 
new_student_scaled = scaler.transform(new_student)
prediction = models["Random Forest"].predict(new_student_scaled)
print("\nPrediction for new student:", "Pass" if prediction[0]==1 else "Fail")

