# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# data analysing
df = pd.read_csv('kidney_disease.csv')
df.head(3)

# %%
df.info()

# %%
df.drop(columns='id', axis=1, inplace=True)
df.head(3)

# %%
df.describe()

# %%
# data preprocessing
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell', 
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white blood_cell_count', 'red blood_cell_count', 
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema', 
              'anomia', 'class']

# %%
df.head(3)

# %%
# convert this columns data-type from object to float which are wrongly intrepreted
txt_cols = ['packed_cell_volume', 'white blood_cell_count','red blood_cell_count']

for col in txt_cols:
    print(f'{col} : {df[col].dtype}')

# %%
for col in txt_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f'{col} : {df[col].dtype}')

# %%
# handling missing values
missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

# %%
def mean_val_imputation(df: pd.DataFrame, col: str) -> None:
    mean_val = df[col].mean()
    df[col].fillna(value=mean_val, inplace=True)
    
def mode_val_imputation(df: pd.DataFrame, col: str) -> None:
    mode_val = df[col].mode()[0]
    df[col].fillna(value=mode_val, inplace=True)

# %%
numeric_cols = [col for col in df.columns if df[col].dtype != 'object']

for col in numeric_cols:
    mean_val_imputation(df, col)
    
non_numeric_cols = [col for col in df.columns if df[col].dtype == 'object']

for col in non_numeric_cols:
    mode_val_imputation(df, col)

df.info()

# %%
# correct wrong catogries by replacing them to correct catogry
print(f'diabetes_mellitus: {df['diabetes_mellitus'].unique()}')
print(f'coronary_artery_disease: {df['coronary_artery_disease'].unique()}')
print(f'class: {df['class'].unique()}')

# %%
df['diabetes_mellitus'] = df['diabetes_mellitus'].replace(to_replace={' yes': 'yes', '\tyes': 'yes','\tno': 'no' })
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace={'\tno': 'no'})
df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})

# %%
print(f'diabetes_mellitus: {df['diabetes_mellitus'].unique()}')
print(f'coronary_artery_disease: {df['coronary_artery_disease'].unique()}')
print(f'class: {df['class'].unique()}')

# %%
catogrical_col = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'appetite', 'hypertension',
                  'diabetes_mellitus', 'coronary_artery_disease', 'peda_edema', 'anomia', 'class']

for col in catogrical_col:
    print(f'{col}: {df[col].unique()}')


# %%
# feature encoding (converting catogrical column to numeric column)
# here only 2 catogry so map yes -> 1 and no -> 0 simple
df['red_blood_cells'] = df['red_blood_cells'].map({'normal': 1, 'abnormal': 0})
df['pus_cell'] = df['pus_cell'].map({'normal': 1, 'abnormal': 0})
df['pus_cell_clumps'] = df['pus_cell_clumps'].map({'present': 1, 'notpresent': 0})
df['bacteria'] = df['bacteria'].map({'present': 1, 'notpresent': 0})
df['appetite'] = df['appetite'].map({'good': 1, 'poor': 0})
df['hypertension'] = df['hypertension'].map({'yes': 1, 'no': 0})
df['diabetes_mellitus'] = df['diabetes_mellitus'].map({'yes': 1, 'no': 0})
df['coronary_artery_disease'] = df['coronary_artery_disease'].map({'yes': 1, 'no': 0})
df['peda_edema'] = df['peda_edema'].map({'yes': 1, 'no': 0})
df['anomia'] = df['anomia'].map({'yes': 1, 'no': 0})
df['class'] = df['class'].map({'ckd': 1, 'not ckd': 0})

# %%
# data visualising
plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, linewidths=0.5)
plt.show()

# %%
class_corr = df.corr()['class'].abs().sort_values(ascending=False)[1:]
class_corr

# %%
df['class'].value_counts()

# %%
# training the model
from sklearn.model_selection import train_test_split

features = df.drop('class', axis=1)
result = df['class']

features_train, features_test, result_train, result_test = train_test_split(features, result, test_size=0.25, random_state=25)

print(f'Features train data shape: {features_train.shape}')
print(f'Features test data shape: {features_test.shape}')

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# %%
models = [
    ('Gaussian Native Bayes', GaussianNB()),
    ('K Neighbors', KNeighborsClassifier(n_neighbors=8)),
    ('Random Forest', RandomForestClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Support Vector Machine', SVC(kernel='linear'))
]

# %%
# result evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(name, model):
    print(f'Evaluation of {name}', end='\n')
    model.fit(features_train, result_train)
    result_predict = model.predict(features_test)
    print(f'Confusion matrix: {confusion_matrix(result_test, result_predict)}', end='\n')
    print(f'Accuracy: {accuracy_score(result_test, result_predict) * 100}%', end='\n')
    print(f'Precision: {precision_score(result_test, result_predict)}', end='\n')
    print(f'Recall: {recall_score(result_test, result_predict)}', end='\n')
    print(f'F1 score: {f1_score(result_test, result_predict)}', end='\n')
    print('-' * 40, end='\n\n')

# %%
for name, model in models:
    evaluate_model(name, model)

# %%
#manual input
user_input = pd.DataFrame([
    {
        'age': 48.0,
        'blood_pressure': 70.0,
        'specific_gravity': 1.0005,
        'albumin': 4.0,
        'sugar': 0.0,
        'red_blood_cells': 1,
        'pus_cell': 0,
        'pus_cell_clumps': 1,
        'bacteria': 0,
        'blood_glucose_random': 117.0,
        'blood_urea': 56.0,
        'serum_creatinine': 3.8,
        'sodium': 111.0,
        'potassium': 2.5,
        'haemoglobin': 11.2,
        'packed_cell_volume': 32,
        'white blood_cell_count': 6700.0,
        'red blood_cell_count': 3.9,
        'hypertension': 1,
        'diabetes_mellitus': 0,
        'coronary_artery_disease': 0,
        'appetite': 0,
        'peda_edema': 1,
        'anomia': 1
    },
    {
        'age': 45,
        'blood_pressure': 80.0,
        'specific_gravity': 1.025,
        'albumin': 0,
        'sugar': 0,
        'red_blood_cells': 1,
        'pus_cell': 1,
        'pus_cell_clumps': 0,
        'bacteria': 0,
        'blood_glucose_random': 82.0,
        'blood_urea': 49.0,
        'serum_creatinine': 0.6,
        'sodium': 147.0,
        'potassium': 4.4,
        'haemoglobin': 15.9,
        'packed_cell_volume': 46.0,
        'white blood_cell_count': 9100.0,
        'red blood_cell_count': 4.7,
        'hypertension': 0,
        'diabetes_mellitus': 0,
        'coronary_artery_disease': 0,
        'appetite': 1,
        'peda_edema': 0,
        'anomia': 0
    }
])

# %%
def predict_ckd(model, input_data):
    predictions = model.predict(input_data)
    return ['CKD' if p == 1 else 'Not CKD' for p in predictions]

results = predict_ckd(models[2][1], user_input)
print(results)


