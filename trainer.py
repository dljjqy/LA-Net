import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from lbfgsnew import LBFGSNew
from scipy import sparse

def coo2tensor(A):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32)

def np2torch(A_path, label='jac'):
    '''
    Label: To identify which iterative method to use.
        Jacobian, Gauess Seidel, CG.
    '''
    A = sparse.load_npz(A_path)
    D = sparse.diags(A.diagonal())
    L = sparse.tril(A, -1)
    U = sparse.triu(A, 1)
    if label == 'jac' or 'mse':
        invM = sparse.linalg.inv(D.tocsc())
        M = L + U
    elif label == 'gauess':
        invM = sparse.linalg.inv((L+D).tocsc())
        M = U     
    return coo2tensor(A), coo2tensor(invM.tocoo()), coo2tensor(M.tocoo())

def pad_neu_bc(x, h, pad=(1, 1, 0, 0), g = 0):
    val = 2 * h * g
    x = F.pad(x, pad=pad, mode='reflect')
    
    if pad == (1, 1, 0, 0):
        x[..., :, 0] -= val
        x[..., :,-1] -= val
    elif pad == (0, 0, 1, 1):
        x[..., 0, :] -= val
        x[...,-1, :] -= val
    return x

def pad_diri_bc(x, pad=(0, 0, 1, 1), g = 0):
    x = F.pad(x, pad=pad, mode='constant', value=g)
    return x

def conv_rhs(x):
    kernel = torch.tensor([[[[0, -0.25, 0], [-0.25, 0, -0.25], [0, -0.25, 0]]]])
    kernel = kernel.type_as(x)
    return F.conv2d(x, kernel)


def gradient_descent(x, A, b):
    r = matrix_batched_vectors_multiply(A, x) - b
    Ar = matrix_batched_vectors_multiply(A, r)
    alpha = batched_vec_inner(r, r)/batched_vec_inner(r, Ar)
    y = x + alpha * r
    return y

def matrix_batched_vectors_multiply(A, y):
    """
    Sparse matrix multiply Batched vectors
    """
    y = torch.transpose(y, 0, 1)
    v = torch.sparse.mm(A, y)
    v = torch.transpose(v, 0, 1)
    return v

def batched_vec_inner(x, y):
    """
    inner product of Batched vectors x and y
    """
    b, n = x.shape
    return torch.bmm(x.view((b, 1, n)), y.view((b, n, 1))) 

def energy(x, A, b):
    Ax = matrix_batched_vectors_multiply(A, x)
    bx = batched_vec_inner(b, x)
    xAx = batched_vec_inner(x, Ax)
    return (xAx/2 - bx).mean()

def mse_loss(x, A, b):
    Ax = matrix_batched_vectors_multiply(A, x)
    return F.mse_loss(Ax, b)

def diri_rhs(x, f, h, g=0):
    '''
    All boundaries are Dirichlet type.
    Netwotk should output prediction without boundary points.(N-2 x N-2)
    '''
    x = pad_diri_bc(x, pad=(1, 1, 1, 1), g=g)
    rhs = conv_rhs(x)
    return rhs + h*h*f[..., 1:-1, 1:-1]

def neu_rhs(x, f, h, g_n=0, g_d=0):
    '''
    Left,right boundary are neumann type.
    Top,bottom boundary are dirichlet type.
    Network should output prediction with boundary points.(N x N)
    '''
    x = pad_neu_bc(x, h, pad=(1, 1, 0, 0), g=g_n)
    x = pad_diri_bc(x, (0, 0, 1, 1), g=g_d)
    rhs = conv_rhs(x)
    return rhs + h*h*f[...,1:-1, 1:-1]

class pl_Model(pl.LightningModule):
    def __init__(self, loss, net, data_path='./data/', lr=1e-3, label='jac'):
        super().__init__()
        self.loss = loss
        self.net = net
        self.lr = lr

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.A, self.invM, self.M = np2torch(data_path+'A.npz', label)
        self.invM = self.invM.to(device)
        self.M = self.M.to(device)
        self.A = self.A.to(device)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, b = batch
        y = self(x)
        y = y.flatten(1, 3)
        with torch.no_grad():
            rhs = self.rhs(y, b)
            # energy_loss = energy(y, self.A, b)
            mse_linalg_loss = mse_loss(y, self.A, b)
            jacobian_loss = self.loss(y, rhs)
        energy_loss = energy(y, self.A, b)
        
        self.log_dict({
            'Jacobian Iteration l1 Loss': jacobian_loss,
            'Mean Energy Loss':energy_loss,
            'MSE Linalg Loss':mse_linalg_loss
        })
        return {'loss' : energy_loss}

    def validation_step(self, batch, batch_idx):
        x, b = batch
        y = self(x)
        y = torch.flatten(y, 1, 3)

        rhs = self.rhs(y, b)
        energy_loss = energy(y, self.A, b)
        mse_linalg_loss = mse_loss(y, self.A, b)
        jacobian_loss = self.loss(y, rhs)

        self.log_dict({
            'Val Jacobian Iteration l1 Loss': jacobian_loss,
            'Val Mean Energy Loss':energy_loss,
            'Val MSE Linalg Loss':mse_linalg_loss
        })
        return {
            'Val Jacobian Iteration l1 Loss': jacobian_loss,
            'Val Mean Energy Loss':energy_loss,
            'Val MSE Linalg Loss':mse_linalg_loss}
    
    def rhs(self, y, b):
        Mx = matrix_batched_vectors_multiply(self.M, y)
        x_new = matrix_batched_vectors_multiply(self.invM, (b-Mx))
        return x_new

    def rhs_cg(self, y, b):
        r = b - matrix_batched_vectors_multiply(self.A, y)

        rr = batched_vec_inner(r, r)
        Ar = matrix_batched_vectors_multiply(self.A, r)
        alpha = rr / batched_vec_inner(r, Ar)
        
        r1 = r - alpha * Ar
        r1r1 = batched_vec_inner(r1, r1)
        beta = r1r1/rr
        p = r1 + beta * r

        alpha = r1r1/batched_vec_inner(p, matrix_batched_vectors_multiply(self.A, p))
        return y + alpha * p


    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]


class pl_lbfgs_Model(pl_Model):
    def __init__(self, *args, **kwargs):
        super(pl_lbfgs_Model, self).__init__(*args, **kwargs)
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        x, b = batch
        opt = self.optimizers()
        def closure():
            if torch.is_grad_enabled():
                opt.zero_grad()
            y = self(x).flatten(1, 3)
            with torch.no_grad():
                rhs = self.rhs(y, b)
            loss = self.loss(y, rhs)
            if loss.requires_grad:
                self.manual_backward(loss)
            return loss
        opt.step(closure)

        y = self(x).flatten(1, 3)
        with torch.no_grad():
            rhs = self.rhs(y, b)
            energy_loss = energy(y, self.A, b)
            mse_linalg_loss = mse_loss(y, self.A, b)
        jacobian_loss = self.loss(y, rhs)

        self.log_dict({
            'Jacobian Iteration l1 Loss': jacobian_loss,
            'Mean Energy Loss':energy_loss,
            'MSE Linalg Loss':mse_linalg_loss
        })

    def configure_optimizers(self):
        optimizer = LBFGSNew(self.parameters(), 
            history_size=7, max_iter=2, line_search_fn=True, batch_mode=True)
        return [optimizer]