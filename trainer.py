from turtle import forward
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

def np2torch(data_path, backward_type='jac', boundary_type='D'):
    '''
    backward_type: To identify which iterative method to use.
        Jacobian, Gauess Seidel, CG.
    '''
    A_path = f'{data_path}A{boundary_type}.npz'
    A = sparse.load_npz(A_path)
    D = sparse.diags(A.diagonal())
    L = sparse.tril(A, -1)
    U = sparse.triu(A, 1)
    if backward_type == 'jac' or 'mse':
        invM = sparse.linalg.inv(D.tocsc())
        M = L + U
    elif backward_type == 'gauess':
        invM = sparse.linalg.inv((L+D).tocsc())
        M = U     
    return coo2tensor(A), coo2tensor(invM.tocoo()), coo2tensor(M.tocoo())

def pad_neu_bc(x, h=0, pad=(1, 1, 0, 0), g = 0):
    val = 2 * h * g
    x = F.pad(x, pad=pad, mode='reflect')
    if val == 0:
        return x
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
    kernel = torch.tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
    kernel = kernel.type_as(x)
    return F.conv2d(x, kernel)

def gradient_descent(x, A, b):
    r = mmbv(A, x) - b
    Ar = mmbv(A, r)
    alpha = bvi(r, r)/bvi(r, Ar)
    y = x + alpha * r
    return y

def mmbv(A, y):
    """
    Sparse matrix multiply Batched vectors
    """
    y = torch.transpose(y, 0, 1)
    v = torch.sparse.mm(A, y)
    v = torch.transpose(v, 0, 1)
    return v

def bvi(x, y):
    """
    inner product of Batched vectors x and y
    """
    b, n = x.shape
    inner_values =  torch.bmm(x.view((b, 1, n)), y.view((b, n, 1))) 
    return inner_values.reshape(b, 1)

def energy(x, A, b):
    Ax = mmbv(A, x)
    bx = bvi(b, x)
    xAx = bvi(x, Ax)
    return (xAx/2 - bx).mean()

def mse_loss(x, A, b):
    Ax = mmbv(A, x)
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

class LAModel(pl.LightningModule):
    def __init__(self, net, data_path='./data/', lr=1e-3, backward_type='jac', boundary_type='D',
                        cg_max_iter=20):
        '''
            All right side computation:
            dir --> output N-2 x N-2 --> pad G ---> get all loss value.
            mixed --> output N-2 x N --> pad ghost points and G ---> get all conv type loss value.
            mixed --> output N-2 x N --> pad G ---> get linear type loss.
        '''
        super().__init__()
        self.net = net
        self.lr = lr
        self.backward_type = backward_type
        self.cg_max_iters = cg_max_iter
        if boundary_type == 'D':
            self.padder = lambda x:pad_diri_bc(x, (1, 1, 1, 1), g=0)
        elif boundary_type == 'N':
            self.padder = lambda x:pad_diri_bc(x, (0, 0, 1, 1), g=0)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.A, self.invM, self.M = np2torch(data_path, backward_type, boundary_type)
        self.invM, self.M, self.A = self.invM.to(device), self.M.to(device),self.A.to(device)
        
    def forward(self, x):
        y = self.net(x)
        y = self.padder(y)
        return y

    def training_step(self, batch, batch_idx):
        x, b = batch
        y = self(x)
        y = torch.flatten(y, 1, -1)
        with torch.no_grad():
            jac = self.rhs_jac(y, b)
            cg = self.rhs_cg(y, b)
        loss_values = {
            'mse' : mse_loss(y, self.A, b),
            'jac' : F.l1_loss(y, jac),
            'cg': F.l1_loss(y, cg, self.cg_max_iters),
            'energy' : energy(y, self.A, b)}

        self.log_dict(loss_values)
        return {'loss' : loss_values[self.backward_type]}

    def validation_step(self, batch, batch_idx):
        x, b = batch
        y = self(x)
        y = torch.flatten(y, 1, -1)
        jac = self.rhs_jac(y, b)
        cg = self.rhs_cg(y, b)

        loss_values = {
            'val_mse' : mse_loss(y, self.A, b),
            'val_jac' : F.l1_loss(y, jac),
            'val_cg': F.l1_loss(y, cg, self.cg_max_iters),
            'val_energy' : energy(y, self.A, b)}

        self.log_dict(loss_values)
        return loss_values
    
    def rhs_jac(self, x, b):
        Mx = mmbv(self.M, x)
        x_new = mmbv(self.invM, (b-Mx))
        return x_new

    def rhs_cg(self, x, b, max_iters=20):
        r = b - mmbv(self.A, x)
        p = r
        for _ in range(max_iters):
            rr = bvi(r, r)
            Ap = mmbv(self.A, p)
            alpha = rr / bvi(p, Ap)
            x = x + alpha * p
            r1 = r - alpha * Ap
            beta = bvi(r1, r1) / rr
            p = r1 + beta * p
            r = r1
        return x
                
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]

class ConvModel(pl.LightningModule):
    def __init__(self, net, boundary_type, lr=1e-3):
        super().__init__()
        self.net = net
        if boundary_type == 'D':
            self.padder = lambda x:pad_diri_bc(x, (1, 1, 1,1), g=0)
        elif boundary_type == 'N':
            self.padder = lambda x:pad_diri_bc(pad_neu_bc(x), (0, 0, 1, 1), g=0)
        self.lr = lr

    def forward(self, x):
        y = self.net(x)
        y = self.padder(y)
        return y
    
    def training_step(self, batch, batch_idx):
        x, f = batch
        y = self(x)
        with torch.no_grad():
            laplace = self.conv_rhs(y) + f
        conv_loss_value = F.l1_loss(x, laplace)
        self.log('conv_loss', conv_loss_value)
        return {'loss', conv_loss_value}

    def validation_step(self, batch, batch_idx):
        x, f = batch
        y = self(x)
        with torch.no_grad():
            laplace = self.conv_rhs(y) + f
        conv_loss_value = F.l1_loss(x, laplace)
        self.log('val_conv_loss', conv_loss_value)
        return {'loss', conv_loss_value}

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]

# class pl_lbfgs_Model(pl_Model):
#     def __init__(self, *args, **kwargs):
#         super(pl_lbfgs_Model, self).__init__(*args, **kwargs)
#         self.automatic_optimization = False
    
#     def training_step(self, batch, batch_idx):
#         x, b = batch
#         opt = self.optimizers()
#         def closure():
#             if torch.is_grad_enabled():
#                 opt.zero_grad()
#             y = self(x).flatten(1, 3)
#             with torch.no_grad():
#                 rhs = self.rhs(y, b)
#             loss = self.loss(y, rhs)
#             if loss.requires_grad:
#                 self.manual_backward(loss)
#             return loss
#         opt.step(closure)

#         y = self(x).flatten(1, 3)
#         with torch.no_grad():
#             rhs = self.rhs(y, b)
#             energy_loss = energy(y, self.A, b)
#             mse_linalg_loss = mse_loss(y, self.A, b)
#         jacobian_loss = self.loss(y, rhs)

#         self.log_dict({
#             'Jacobian Iteration l1 Loss': jacobian_loss,
#             'Mean Energy Loss':energy_loss,
#             'MSE Linalg Loss':mse_linalg_loss
#         })

#     def configure_optimizers(self):
#         optimizer = LBFGSNew(self.parameters(), 
#             history_size=7, max_iter=2, line_search_fn=True, batch_mode=True)
#         return [optimizer]