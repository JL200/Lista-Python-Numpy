'''

Lista 2 - João Luiz de Castro Pereira
Exercícios sobre Numpy

'''
import numpy as np





'''questão 1:'''

#definição de vetores:
y_prediction = np.array([1, 2, 3])
y_i = np.array([0, 0, 3])


#definição da função
def calculate_eqm(y_pred,yi):
    n = len(yi)
    resp = np.sum((y_pred-yi)**2)
    return resp/n

resposta = calculate_eqm(y_prediction,y_i)

print(resposta)





'''questão 2:'''

#definição de matriz aleatória:
X = np.random.randn(64,512)

#definição da função:
def process_EEG_signal(matriz_sinais):
    media = np.mean(matriz_sinais,0)
    X_processado = matriz_sinais - media
    return X_processado
X2 = process_EEG_signal(X)

xdif = X-X2


print(f'Matriz original: {X} \n Matriz processada: {X2} \n Dimensões da matriz processada: {X2.shape}')
print(f'matriz diferença:{xdif}')





'''questão 3:'''

#definição de vetor:

teste = np.random.randn(3,10)
print(teste)

#definindo função:
def locate_outliers(X):
    outliers = np.zeros_like(X, dtype=bool)
    out_valores = []
    n_out = 0
    
    for i in range(X.shape[1]):
        feature = X[:,i]
        Q1, Q3 = np.quantile(feature, [0.25, 0.75])
        IQR = Q3 - Q1
        outliers_bool = ((feature < Q1 - 1.5 * IQR) | (feature > Q3 + 1.5 * IQR))
        outliers[:,i][outliers_bool] = True
        out_valores.append(feature[outliers_bool])
        n_out += sum(outliers_bool)
    
    return outliers, n_out, out_valores



is_outlier, outliers_count, outliers = locate_outliers(teste)

print(is_outlier)
print(outliers_count)
print(outliers)