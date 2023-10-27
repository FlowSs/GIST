import numpy as np
import torch

# Taken from https://github.com/js-d/sim_metric/blob/main/dists/scoring.py
## CCA
def pwcca_sim(A, B):
    
    evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    evals_a = (evals_a + np.abs(evals_a)) / 2
    inv_a = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_a])

    evals_b, evecs_b = np.linalg.eigh(B @ B.T)
    evals_b = (evals_b + np.abs(evals_b)) / 2
    inv_b = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_b])
    
    
    cov_ab = A @ B.T
    
    temp = (
        (evecs_a @ np.diag(inv_a) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b) @ evecs_b.T)
    )
    try:
        u, s, _ = np.linalg.svd(temp) 
    except:
        u, s, _ = np.linalg.svd(temp * 100)
        s = s / 100
    
    transformed_a = (u.T @ (evecs_a @ np.diag(inv_a) @ evecs_a.T) @ A).T

    in_prod = transformed_a.T @ A.T
    weights = np.sum(np.abs(in_prod), axis=1)
    weights = weights / np.sum(weights)
    dim = min(len(weights), len(s))
    
    return np.dot(weights[:dim], s[:dim])
    
## CKA
def lin_cka_sim(x, y):
    A, B = torch.tensor(x), torch.tensor(y)
    similarity = torch.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = torch.linalg.norm(A @ A.T, ord="fro") * torch.linalg.norm(
        B @ B.T, ord="fro"
    )
    return (similarity / normalization).numpy()

def procrustes_sim(x, y):
    A, B = torch.tensor(x), torch.tensor(y)
    A_sq_frob = torch.sum(A ** 2)
    B_sq_frob = torch.sum(B ** 2)
    nuc = torch.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return  (1 - 0.5*(A_sq_frob + B_sq_frob - 2 * nuc)).numpy()



## Error-Corrected Disagreement
def error_dis(A, B):

    A = np.abs(A)
    B = np.abs(B)
    m_dis = np.mean(np.argmax(A, 1) != np.argmax(B, 1))

    return m_dis 

## J Div

def j_div(A, B):
    from scipy.special import kl_div

    A = np.abs(A)
    B = np.abs(B)
    return np.mean([kl_div(A[i],B[i]) + kl_div(B[i],A[i]) for i in range(len(A))])/2





