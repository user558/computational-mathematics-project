import numpy as np

# Randomized SVD:
# Параметр А - матрица m x n
# Параметр rank - Желаемое приближение ранга
# Параметр n_oversamples - Дополнительное количество случайных векторов для случайной выборки для обеспечения дальнейшей обработки
# return - U, S и Vt как в усеченном SVD.

def rand_svd(A, rank, n_oversamples=None):

    if n_oversamples is None:
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

# Шаг 1
    Q = find_Q(A, n_samples)

# Шаг 2
    B = Q.T @ A
    U_1, S, Vt = np.linalg.svd(B)
    U = Q @ U_1

# Усечение
#[:, :rank] - rank столбцов
#[:rank], Vt[:rank, :] - rank строк
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    return U, S, Vt


def find_Q(A, n_samples):
# Дана матрица A и число выборок, вычисляет ортогональную матрицу, которая приближает ранг A.
# Параметр А - Матрица m x n.
# Параметр n_samples - Число гауссовских случайных выборок.
# Параметр n_subspace_iters - Число итераций подпространства.
# return - Ортонормированный базис для приближенного ранга A.
    (m, n) = np.shape(A)
    W = np.random.randn(n, n_samples)
    Y = A @ W

    return Q_QR(Y)

def Q_QR(A):
#QR-разложение через отражения Хаусхолдера
    (r, c) = np.shape(A)
    #единичная матрица размера r*r
    Q = np.identity(r)
    R = np.copy(A)
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_cnt = np.identity(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_cnt, R) # R = P(r-1)*...*P(2)*P(0)*A
        Q = np.dot(Q, Q_cnt) # Q = P(0) *P(2)*...*P(r-1)
    return Q

A = np.array([[3,4,3],[1,2,3],[4,2,1]])
U, D, VT = rand_svd(A,3)
print(U)
