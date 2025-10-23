# Salve como 2_train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

INPUT_CSV_FILE = 'treino_classificado.csv'
OUTPUT_MODEL_FILE = 'screw_classifier_model.pkl'


# simple error handling
try:
    data = pd.read_csv(INPUT_CSV_FILE)
except FileNotFoundError:
    print(f"Erro: Arquivo '{INPUT_CSV_FILE}' não encontrado.")
    print("Execute primeiro o '1_data_collector.py' para gerar os dados.")
    exit()

if len(data) < 20: # increase the minimum
    print(f"Aviso: O arquivo CSV tem poucos dados ({len(data)} linhas). O modelo pode não ser preciso.")

X = data[['area', 'aspect_ratio', 'extent', 'solidity']]
y = data['is_screw']

if len(y.unique()) < 2:
    print("Erro: O dataset de treino contém apenas uma classe. Não é possível treinar o modelo.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Total de amostras: {len(data)}")
print(f"Usando {len(X_train)} amostras para treinar e {len(X_test)} para testar.")

print("\n--- Treinando o modelo RandomForest ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Treinamento concluído!")

print("\n--- Avaliando o desempenho do modelo ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo no conjunto de teste: {accuracy * 100:.2f}%")
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred, target_names=['Nao Parafuso', 'Parafuso']))

joblib.dump(model, OUTPUT_MODEL_FILE)
print(f"\nModelo salvo com sucesso como '{OUTPUT_MODEL_FILE}'!")

