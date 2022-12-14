import numpy as np
import pytorch_lightning as pl
import scipy.sparse.linalg as linalg
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import *
from models import model_names
from trainer import LAModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

def normal(x, y, h, mean=[0, 0]):
    var = np.diag([1] * 2) * h**2
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, var)
    return rv.pdf(pos)

def normal_fourth(x, y, h, a=1):
    m1 = [-a/2, -a/2]
    m2 = [a/2, a/2]
    m3 = [-a/2, a/2]
    m4 = [a/2, -a/2]
    var = np.diag([1] * 2) * h**2
    pos = np.dstack((x, y))
    rv1 = multivariate_normal(m1, var)
    rv2 = multivariate_normal(m2, var)
    rv3 = multivariate_normal(m3, var)
    rv4 = multivariate_normal(m4, var)
    return rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos) + rv4.pdf(pos)

def R(x, y):
    return np.sqrt(x**2 + y**2)

def yita11_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask * ((6 * (3 - 4 * r)/np.pi)/H2)

def yita12_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask * ((12 * (5 * r**2 - 8 * r + 3)/np.pi)/H2)

def yita22_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((12 * (15 * r**2 - 20 * r + 6)/np.pi)/H2)

def yita23_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((-60 * (7 * r**3 - 15 * r**2 + 10 * r - 2)/np.pi)/H2)

def yita25_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((84*(24*r**5-70*r**4+70*r**3-25*r**2+1)/np.pi)/H2)

def yita2cos_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*(((-1 / (9*np.pi**4-104*np.pi**2+48)) *
                  ((81*np.pi/16)*(3*np.pi**4-32*np.pi**2+48)*np.cos(3*np.pi*r) +
                   2*np.pi*(9*np.pi**4-80*np.pi**2+48) * np.cos(2*np.pi*r) +
                   ((np.pi/16)*(45*np.pi**2+32*np.pi**2-48)*np.cos(np.pi*r) + 144*np.pi)))/H2)

def fd_A_with_bc(n, order=2):
    N2 = n**2
    A = sparse.lil_matrix((N2, N2))
    if order == 2:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j

                A[idx, idx] += 4
                A[idx, idx-1] = -1
                A[idx, idx+1] = -1
                A[idx, idx-n] = -1
                A[idx, idx+n] = -1
    elif order == 4:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j
                A[idx, idx] = 20
                A[idx, idx-1] = -4
                A[idx, idx+1] = -4
                A[idx, idx-n] = -4
                A[idx, idx+n] = -4

                A[idx, idx-n-1] = -1
                A[idx, idx-n+1] = -1
                A[idx, idx+n-1] = -1
                A[idx, idx+n+1] = -1

    # Homogeneous Dirichlet Boundary
    for i in range(0, n):
        idx = 0 * n + i
        A[idx, idx] = 1

        idx = (n-1) * n + i
        A[idx, idx] = 1

        idx = i * n
        A[idx, idx] = 1

        idx = i * n + n - 1
        A[idx, idx] = 1
    A = A.tocoo()
    return A

def fd_b_bc(f, h, order=2):
    n, _ = f.shape
    h2 = h**2
    b = np.zeros(n**2)

    if order == 2:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j
                b[idx] = f[i, j]*h2
    elif order == 4:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j
                b[idx] += (8*f[i, j] + f[i-1, j] + f[i+1, j] +
                           f[i, j-1] + f[i, j+1])/2*h2
    return b

def fd_A(num, order=2):
    n = num - 2
    N = n**2

    if order == 2:
        B = sparse.diags([4]*n) + sparse.diags([-1]*(n-1), -
                                               1) + sparse.diags([-1]*(n-1), 1)
        A = sparse.block_diag([B]*n)
        A += sparse.diags([-1]*(N-n), -n)
        A += sparse.diags([-1]*(N-n), n)
    elif order == 4:
        B = sparse.diags([20]*n) + sparse.diags([-4]*(n-1),
                                                1) + sparse.diags([-4]*(n-1), -1)
        A = sparse.block_diag([B]*n)
        A = A + sparse.diags([-4]*(n*(n-1)), n) + \
            sparse.diags([-4]*(n*(n-1)), -n)

        l = ([0] + [-1]*(n-1))*(n-1) + [0]
        A = A + sparse.diags(l, n-1) + sparse.diags(l, -n+1)

        l = (([-1]*(n-1) + [0]) * (n-1))[:-1]
        A = A + sparse.diags(l, n+1) + sparse.diags(l, -n-1)
    return A.tocoo()

def fd_b(f, h, order=2):
    n, _ = f.shape
    n = n - 2
    b = np.zeros(n**2)
    if order == 2:
        b += (f[1:-1, 1:-1].flatten())
        b *= (h**2)
    elif order == 4:
        kernel = np.array([[0, 1, 0], [1, 8, 1], [0, 1, 0]])
        b += convolve2d(f, kernel, mode='valid').flatten()[::-1]
        b *= (h**2/2)
    return b

def fd_A_neu(n, neus=['left', 'right'], diris=['top', 'bottom']):
    '''
    Generate linear system for 
        Top, Down Dirichlet Left, Right Neumann Boundary
    '''
    N2 = n**2
    A = sparse.lil_matrix((N2, N2))
    
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx] += 4
            A[idx, idx-1] = -1
            A[idx, idx+1] = -1
            A[idx, idx-n] = -1
            A[idx, idx+n] = -1
    
    # Neumann Boundary
    for k in neus:
        if k == 'top':
            for i in range(1, n-1):
                idx = 0 * n + i
                A[idx, idx] = 4
                A[idx, idx+n] = -2
                A[idx, idx-1] = A[idx, idx+1] = -1
                
        if k == 'bottom':
            for i in range(1, n-1):
                idx = (n-1) * n + i
                A[idx, idx] = 4
                A[idx, idx-n] = -2
                A[idx, idx-1] = A[idx, idx+1] = -1
                
        if k == 'left':
            for i in range(1, n-1):
                idx = i * n
                A[idx, idx] = 4
                A[idx, idx+1] = -2
                A[idx, idx-n] = A[idx, idx+n] = -1
                
        if k == 'right':
            for i in range(1, n-1):
                idx = i * n + n -1
                A[idx, idx] = 4
                A[idx, idx-1] = -2
                A[idx, idx-n] = A[idx, idx+n] = -1
                
    # Dirichlet Boundary
    for k in diris:
        if k == 'top':
            for i in range(0, n):
                idx = 0 * n + i
                A[idx, idx] = 1
                
        if k == 'bottom':
            for i in range(0, n):
                idx = (n-1) * n + i
                A[idx, idx] = 1
                
        if k == 'left':
            for i in range(0, n):
                idx = i * n
                A[idx, idx] = 1
            
        if k == 'right':
            for i in range(0, n):
                idx = i * n + n -1
                A[idx, idx] = 1
    A = A.tocoo()
    return A

def apply_dirichlet_bc_for_all(b, bc, g, order=2):
    '''
    bc --> True for soft, Flase for Hard
    g --> the value of boundary
    '''
    if g == 0:
        return b
    n = int(np.sqrt(len(b)))
    if bc:
        for i in range(0, n):
            idx = 0 * n + i
            b[idx] = g

            idx = (n-1) * n + i
            b[idx] = g

            idx = i * n
            b[idx] = g

            idx = i * n + n - 1
            b[idx] = g
    else:
        if order == 2:
            for i in range(0, n):
                idx = 0 * n + i
                b[idx] += g

                idx = (n-1) * n + i
                b[idx] += g

                idx = i * n
                b[idx] += g

                idx = i * n + n - 1
                b[idx] += g

        elif order == 4:
            for i in range(0, n):
                idx = 0 * n + i
                b[idx] += 6*g

                idx = (n-1) * n + i
                b[idx] += 6*g

                idx = i * n
                b[idx] += 6*g

                idx = i * n + n - 1
                b[idx] += 6*g
            b[0] -= g
            b[n-1] -= g
            b[(n-1)*n] -= g
            b[-1] -= g
    return b

def apply_neumann_bc(b, h, f, bcs={'left': 0, 'right': 0}):
    '''
    Apply Neumann boundary conditions on the left, right boundary.
    '''
    n = int(np.sqrt(len(b)))
    h2 = h**2
    for k in bcs.keys():
        g = bcs[k]
        if k == 'top':
            for i in range(1, n-1):
                idx = i
                b[idx] = h2 * f[0, i] - 2 * h * g

        if k == 'bottom':
            for i in range(1, n-1):
                idx = (n-1) * n + i
                b[idx] = h2 * f[-1, i] - 2 * h * g
        
        if k == 'left':
            for i in range(1, n-1):
                idx = i * n
                b[idx] = h2 * f[i, 0] - 2 * h * g

        if k == 'right':
            for i in range(1, n-1):
                idx = i * n + n - 1
                b[idx] = h2 * f[i, -1] - 2 * h * g
        
    return b

def apply_diri_bc(b, bcs={'top': 0, 'bottom': 0}):
    n = int(np.sqrt(len(b)))
    
    for k in bcs.keys():
        g = bcs[k]
        if k == 'top':
            for i in range(n):
                idx = 0 * n + i
                b[idx] = g
        if k == 'bottom':
            for i in range(n):
                idx = (n-1) * n + i
                b[idx] = g
        if k == 'left':
            for i in range(n):
                idx = i * n
                b[idx] = g
        if k == 'right':
            for i in range(n):
                idx = i * n + n - 1 
                b[idx] = g
    return b

def fv_mesh(a, N):
    h = 2*a / N
    left, bottom = -a + h/2, -a + h/2
    right, top = a - h/2, a - h/2
    
    x = np.linspace(left, right, N)
    y = np.linspace(bottom, top, N)
    return np.meshgrid(x, y)

def fv_b_func(a, N, f):
    h = 2*a / N
    xx, yy = fv_mesh(a, N)
    f_mat = f(xx, yy, h).flatten()
    return f_mat * h * h

def fv_b_point(N):
    b = np.zeros(N*N)
    idx = N//2 + N * (N//2)
    b[idx] = 1
    return b

def fv_A_dirichlet(n):
    n2 = n**2
    A = sparse.lil_matrix((n2, n2))
    # Interior points
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx] = -4
            A[idx, idx+1] = A[idx, idx-1] = A[idx, idx-n] = A[idx, idx+n] = 1
    
    # Boundary points
    for i in range(1, n-1):
        # Top
        idx = i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx+n] = 4/3
        
        # Bottom
        idx = (n-1) * n + i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx-n] = 4/3
        
        # Left
        idx = i * n
        A[idx, idx] = -6
        A[idx, idx+n] = A[idx, idx-n] = 1
        A[idx, idx+1] = 4/3
        
        # Right
        idx = i * n + n - 1
        A[idx, idx] = -6
        A[idx, idx+n] = A[idx, idx-n] = 1
        A[idx, idx-1] = 4/3
        
    # Four corners
    # Left top
    idx = 0
    A[idx, idx] = -8
    A[idx, idx+1] = A[idx, idx+n] = 4/3
    # Right Top
    idx = n-1
    A[idx, idx] = -8
    A[idx, idx-1] = A[idx, idx+n] = 4/3
    # Left Bottom
    idx = (n-1) * n
    A[idx, idx] = -8
    A[idx, idx+1] = A[idx, idx-n] = 4/3
    # Right Bottom
    idx = n2 - 1
    A[idx, idx] = -8
    A[idx, idx-1] = A[idx, idx-n] = 4/3
    
    A = A.tocoo()
    return A

def fv_A_neu(n):
    '''
    Left and Right are neumann, top and down are dirichlet
    '''
    n2 = n**2
    A = sparse.lil_matrix((n2, n2))
    # Interior points
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx] = -4
            A[idx, idx+1] = A[idx, idx-1] = A[idx, idx-n] = A[idx, idx+n] = 1
    
    # Boundary points
    for i in range(1, n-1):
        # Top
        idx = i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx+n] = 4/3
        
        # Bottom
        idx = (n-1) * n + i
        A[idx, idx] = -6
        A[idx, idx+1] = A[idx, idx-1] = 1
        A[idx, idx-n] = 4/3
        
        # Left
        idx = i * n
        A[idx, idx] = -3
        A[idx, idx+n] = A[idx, idx-n] = A[idx, idx+1] = 1
        
        # Right
        idx = i * n + n - 1
        A[idx, idx] = -3
        A[idx, idx+n] = A[idx, idx-n] = A[idx, idx-1] = 1
        
    # Four corners
    
    # Left top
    idx = 0
    A[idx, idx] = -5
    A[idx, idx+n] = 4/3
    A[idx, idx+1] = 1
    
    # Right Top
    idx = n-1
    A[idx, idx] = -5
    A[idx, idx+n] = 4/3
    A[idx, idx-1] = 1
    
    # Left Bottom
    idx = (n-1) * n
    A[idx, idx] = -5
    A[idx, idx-n] = 4/3
    A[idx, idx+1] = 1
    
    # Right Bottom
    idx = n2 - 1
    A[idx, idx] = -5
    A[idx, idx-n] = 4/3
    A[idx, idx-1] = 1
    
    A = A.tocoo()
    return A
        
def gen_hyper_dict(gridSize, batch_size, net, features, data_type, boundary_type, 
            numerical_method='fd', backward_type='jac', lr=1e-3, max_epochs=100, ckpt=False, gpus=1,
            dm=LADataModule, pl_model=LAModel):
    '''
    gridSize: How big mesh. 33, 65, 129
    batch_size: batch size. 8, 16, 24, 32
    input_type: F type or M type.
    net: The architecture.UNet or Attention UNet.
    features: To control the parameters size of network. [16, 32]
    data_type: One or four point source, Big or small area. [bigOne, bigFour, One, Four]
    boundary_type: Dirichlet or mixed with neumann.[D, N]
    backward_type: The loss function used to train the network.[jac, mse, conv, cg,...]
    lr:learning rate
    max_epochs: epochs
    ckpt: True for load parameters from ckpt
    '''
    exp_name = f'{numerical_method}_{backward_type}_{gridSize}_{net}_{features}_bs{batch_size}_{data_type}{boundary_type}'
    data_path = f'../data/{gridSize}/{data_type}/'
    mat_path = f'../data/{gridSize}/mat/'
    if ckpt:
        exp_name = 'resume_' + exp_name
    layers = list(2**i for i in range(int(np.log2(gridSize)) - 2))
    model = model_names[net](layers=layers, features=features, boundary_type=boundary_type, numerical_method=numerical_method)
    
    dc = {'trainer':{}, 'pl_model':{}, 'pl_dataModule':{}}
    dc['name'] = exp_name

    dc['trainer'] = {'max_epochs': max_epochs, 'precision': 32, 'check_val_every_n_epoch': 1, 'accelerator': 'cuda', 'devices': gpus}
    dc['trainer']['logger'] = TensorBoardLogger('../lightning_logs/', exp_name)
    dc['trainer']['callbacks'] = ModelCheckpoint(monitor= f'val_{backward_type}', mode='min', every_n_train_steps=0,
                                        every_n_epochs=1, train_time_interval=None, save_top_k=3, save_last=True,)
    
    n = gridSize
    a = 500 if 'big' in data_type else 1
    h = 2*a/(n-1) if numerical_method == 'fd' else 2*a/n
    dc['pl_model']['name'] = pl_model
    dc['pl_model']['args'] = [model, h, mat_path, lr, numerical_method, backward_type, boundary_type, gridSize//2]
    dc['pl_model']['ckpt'] = ckpt
    
    dc['pl_dataModule']['name'] = dm
    dc['pl_dataModule']['args'] = [data_path, batch_size, a, n, numerical_method]
    
    return dc

def main(kwargs):
    # Initilize the Data Module
    args = kwargs['pl_dataModule']['args']
    dm = kwargs['pl_dataModule']['name'](*args)

    # Initilize the model
    args = kwargs['pl_model']['args']
    pl_model = kwargs['pl_model']['name'](*args)
    
    # Initilize Pytorch lightning trainer
    pl_trainer = pl.Trainer(**kwargs['trainer'])

    ckpt = kwargs['pl_model']['ckpt']
    if ckpt:
        pl_trainer.fit(
            model=pl_model,
            datamodule=dm,
            ckpt_path=ckpt)
    else:
        pl_trainer.fit(
            model=pl_model,
            datamodule=dm)
    return True

def _getXsFVM_fs(xs, ys, n, a, q=1):
    h = (2*a)/n
    l = -a + h/2
    fs = []
    for point in zip(xs, ys):
        idx = int((point[0] - l) // h)
        idy = int((point[1] - l) // h)
        f = np.zeros((n, n))
        f[idx, idy] = q
        fs.append(f)
    return fs

def _getQsFVM_fs(n, a, Qs, locs=[[0, 0]]):
    h = (2*a)/n
    l = -a + h/2
    f = np.zeros((1, n, n))
    for point in locs:
        idx = int((point[0] - l) // h)
        idy = int((point[1] - l) // h)
        f[0, idx, idy] = 1
    
    return Qs * f

def _getJacMatrix(dir, A):
    D = sparse.diags(A.diagonal())
    L = sparse.tril(A, -1)
    U = sparse.triu(A, 1)

    invM = sparse.diags(1/A.diagonal())
    M = L + U
    sparse.save_npz(dir+'_jac_invM', invM.tocoo())
    sparse.save_npz(dir+'_jac_M', M.tocoo())
    return True


def _getMatrix(dir, n):
    Ad = -fd_A_with_bc(n)
    p = dir + 'fd_AD'
    sparse.save_npz(p, Ad)
    _getJacMatrix(p, Ad)
    del Ad

    An = -fd_A_neu(n)
    p = dir + 'fd_AN'
    sparse.save_npz(p, An)
    _getJacMatrix(p, An)
    del An

    Ad = fv_A_dirichlet(n)
    p = dir + 'fv_AD'
    sparse.save_npz(p, Ad)
    _getJacMatrix(p, Ad)
    del Ad

    An = fv_A_neu(n)
    p = dir + 'fv_AN'
    sparse.save_npz(p, An)
    _getJacMatrix(p, An)
    del An
    return True

def _getFdata(dir, fs, valfs,numerical_method='fd'):
    np.save(f'{dir}{numerical_method}_F.npy', fs)
    np.save(f'{dir}{numerical_method}_ValF.npy', valfs)
    return True

def _getBdata(dir, a, n, fs, valfs, numerical_method='fd'):  
    if numerical_method == 'fd':
        h = 2*a/(n-1)
        B = fs.reshape(-1, n**2) * h**2
        valB = valfs.reshape(-1, n**2) * h**2 
    
    elif numerical_method == 'fv':  
        B = fs.reshape(-1, n**2)
        valB = valfs.reshape(-1, n**2)

    np.save(f'{dir}{numerical_method}_B.npy', B)
    np.save(f'{dir}{numerical_method}_ValB.npy', valB)
    del B, valB
    return True

def _genLocsData(dir, a=1, Q=1, n=129, train_N=1000, val_N=100):
    p = Path(dir)
    if not p.is_dir():
        p.mkdir(exist_ok=False)
    h = 2*a/(n-1)
    x = np.linspace(-a, a, n)
    y = np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    train_xs = np.random.uniform(-a+3*h, a-3*h, train_N)
    train_ys = np.random.uniform(-a+3*h, a-3*h, train_N)
    val_xs = np.random.uniform(-a+3*h, a-3*h, val_N)
    val_ys = np.random.uniform(-a+3*h, a-3*h, val_N)

    # FDM
    fd_train_fs = list(normal(xx, yy, h, point) \
        for point in zip(train_xs, train_ys))
    fd_val_fs = list(normal(xx, yy, h, point) \
        for point in zip(val_xs, val_ys))
    fd_train_fs = np.stack(fd_train_fs, axis=0)
    fd_val_fs = np.stack(fd_val_fs, axis=0)

    _getFdata(dir, fd_train_fs, fd_val_fs, 'fd')
    _getBdata(dir, a, n, fd_train_fs, fd_val_fs, 'fd')

    # FVM
    fv_train_fs = _getXsFVM_fs(train_xs, train_ys, n, a, Q)
    fv_val_fs = _getXsFVM_fs(val_xs, val_ys, n, a, Q)
    fv_train_fs = np.stack(fv_train_fs, axis=0)
    fv_val_fs = np.stack(fv_val_fs, axis=0)

    _getFdata(dir, fv_train_fs, fv_val_fs, 'fv')
    _getBdata(dir, a, n, fv_train_fs, fv_val_fs, 'fv')
    return True

def _genQsData(dir, a=1, minQ=1, maxQ=2, n=130, train_N=2500, val_N=10, four=False):
    p = Path(dir)
    if not p.is_dir():
        p.mkdir(exist_ok=False)

    train_Qs = np.random.uniform(minQ, maxQ, (train_N, 1, 1))
    val_Qs = np.linspace(minQ, maxQ, val_N).reshape((val_N, 1, 1))

    # FDM
    h = (2*a)/(n-1)
    x = np.linspace(-a, a, n)
    y = np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    
    fd_train_fs = train_Qs * normal(xx, yy, h) \
        if not four else train_Qs * normal_fourth(xx, yy, h, a)
    fd_val_fs = val_Qs * normal(xx, yy, h) \
        if not four else val_Qs * normal_fourth(xx, yy, h, a)
    _getFdata(dir, fd_train_fs, fd_val_fs, 'fd')
    _getBdata(dir, a, n, fd_train_fs, fd_val_fs, 'fd')

    # FVM   
    fv_train_fs = _getQsFVM_fs(n, a, train_Qs, [[0, 0]]) \
        if not four else _getQsFVM_fs(n, a, train_Qs, [[a/2, a/2], [a/2, -a/2], [-a/2, a/2], [-a/2, -a/2]])
    fv_val_fs = _getQsFVM_fs(n, a, val_Qs, [[0, 0]]) \
        if not four else _getQsFVM_fs(n, a, val_Qs, [[a/2, a/2], [a/2, -a/2], [-a/2, a/2], [-a/2, -a/2]])
    _getFdata(dir, fv_train_fs, fv_val_fs, 'fv')
    _getBdata(dir, a, n, fv_train_fs, fv_val_fs, 'fv')
    return True

def _genData(path, n):
    '''
    Generate all types data and matrix.
    '''
    p = Path(f'{path}/{n}/')
    m = int(100/7)
    if not p.is_dir(): p.mkdir()
    with tqdm(total=100, desc=f'n={n}Generating') as pbar:
    # Generate the linear system
        pbar.set_description('Matrix Generating')
        p = Path(f'{path}/{n}/mat/')
        if not p.is_dir(): p.mkdir(exist_ok=False)
        _getMatrix(f'{path}/{n}/mat/', n)
        pbar.update(m)

        pbar.set_description('Type One Generating')
        p = f'{path}/{n}/One/'
        _genQsData(p, 1, 1, 2, n, 2000, 10, False)
        pbar.update(m)

        pbar.set_description('Type Four Generating')
        p = f'{path}/{n}/Four/'
        _genQsData(p, 1, 1, 2, n, 2000, 10, True)
        pbar.update(m)

        pbar.set_description('Type BigOne Generating')
        p = f'{path}/{n}/BigOne/'
        _genQsData(p, 500, 10000, 15000, n, 2500, 50, False)
        pbar.update(m)

        pbar.set_description('Type BigFour Generating')
        p = f'{path}/{n}/BigFour/'
        _genQsData(p, 500, 10000, 15000, n, 2500, 50, True)
        pbar.update(m)

        pbar.set_description('Type Locs Generating')
        p = f'{path}/{n}/Locs/'
        _genLocsData(p, 1, 1, n, 2000, 100)
        pbar.update(m)

        pbar.set_description('Type BigLocs Generating')
        p = f'{path}/{n}/BigLocs/'
        _genLocsData(p, 500, 10000, n, 5000, 500)
        pbar.update(m)
    
    return True


def gen_test_data(Qs, n, f, a=1, order=2, g=0, path='./data/test/'):
    p = Path(path)
    if not p.is_dir():
        p.mkdir(exist_ok=False)

    h = (2*a)/(n-1)
    x, y = np.linspace(-a, a, n), np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    f_mat = f(xx, yy, h)

    A = fd_A_with_bc(n, order).tocsr()
    for q in Qs:
        b = fd_b_bc(f_mat, h, order)
        b = apply_dirichlet_bc_for_all(b, True, g, order)
        u = spsolve(A, b)
        np.save(f'{path}{q:.3f}.npy', u)

    return True

if __name__ == '__main__':
    # yitas = [yita11_2d, yita12_2d, yita22_2d, yita23_2d, yita25_2d, yita2cos_2d]
    Ns = [33, 65, 129, 257]
    for n in Ns:
        _genData('../data', n)
