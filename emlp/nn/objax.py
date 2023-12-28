import jax
import jax.numpy as jnp
import objax.nn as nn
#import flax
#from flax import linen as nn
import objax.functional as F
import numpy as np
from emlp.reps import T,Rep,Scalar
from emlp.reps import bilinear_weights
from emlp.reps.product_sum_reps import SumRep
import collections
from oil.utils.utils import Named,export
import scipy as sp
import scipy.special
import random
import logging
from objax.variable import TrainVar, StateVar
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from objax.nn.init import orthogonal
from scipy.special import binom
from jax import jit,vmap
from functools import lru_cache as cache
import sys
from emlp.DisGNN.kDisGNN_model import kDisGNN
from emlp.DisGNN.script_utils import trainer_setup, test, train, get_cfgs
from emlp.DisGNN.utils.activation_fns import activation_fn_map
import torch

'''
    get hparams
'''
model_name   = "2FDis" 
dataset_name = "qm9"
config_path = "/home/snirhordan/ScalarEMLP/emlp/DisGNN/{}_{}.yaml".format(model_name, dataset_name) #TODO change config path

config = get_cfgs(config_path, None, None, "ethanol")

print("-"*20)
print(config)
print("-"*20)

scheduler_config = config.scheduler_config
optimizer_config = config.optimizer_config
model_config = config.model_config

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return nn.Sequential(args)

@export
class Linear(nn.Linear):
    """ Basic equivariant Linear layer from repin to repout."""
    def __init__(self, repin, repout):
        nin,nout = repin.size(),repout.size()
        super().__init__(nin,nout)
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w = TrainVar(orthogonal((nout, nin)))
        self.rep_W = rep_W = repout*repin.T
        
        rep_bias = repout
        self.Pw = rep_W.equivariant_projector()
        self.Pb = rep_bias.equivariant_projector()
        logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
    def __call__(self, x): # (cin) -> (cout)
        logging.debug(f"linear in shape: {x.shape}")
        W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb@self.b.value
        out = x@W.T+b
        logging.debug(f"linear out shape:{out.shape}")
        return out

@export
class BiLinear(Module):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""
    def __init__(self, repin, repout):
        super().__init__()
        Wdim, weight_proj = bilinear_weights(repout,repin)
        self.weight_proj = jit(weight_proj)
        self.w = TrainVar(objax.random.normal((Wdim,)))#xavier_normal((Wdim,))) #TODO: revert to xavier
        logging.info(f"BiW components: dim:{Wdim}")

    def __call__(self, x,training=True):
        # compatible with non sumreps? need to check
        W = self.weight_proj(self.w.value,x)
        out= .1*(W@x[...,None])[...,0]
        return out

@export
def gated(sumrep): #TODO: generalize to mixed tensors?
    """ Returns the rep with an additional scalar 'gate' for each of the nonscalars and non regular
        reps in the input. To be used as the output for linear (and or bilinear) layers directly
        before a :func:`GatedNonlinearity` to produce its scalar gates. """
    return sumrep+sum([Scalar(rep.G) for rep in sumrep if rep!=Scalar and not rep.is_permutation])

@export
class GatedNonlinearity(Module): #TODO: add support for mixed tensors and non sumreps
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""
    def __init__(self,rep):
        super().__init__()
        self.rep=rep
    def __call__(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        return activations

@export
class EMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    def __init__(self,rep_in,rep_out):
        super().__init__()
        self.linear = Linear(rep_in,gated(rep_out))
        self.bilinear = BiLinear(gated(rep_out),gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)
    def __call__(self,x):
        lin = self.linear(x)
        preact =self.bilinear(lin)+lin
        return self.nonlinearity(preact)

def uniform_rep_general(ch,*rep_types):
    """ adds all combinations of (powers of) rep_types up to
        a total of ch channels."""
    #TODO: write this function
    raise NotImplementedError

@export
def uniform_rep(ch,group):
    """ A heuristic method for allocating a given number of channels (ch)
        into tensor types. Attempts to distribute the channels evenly across
        the different tensor types. Useful for hands off layer construction.
        
        Args:
            ch (int): total number of channels
            group (Group): symmetry group

        Returns:
            SumRep: The direct sum representation with dim(V)=ch
        """
    d = group.d
    Ns = np.zeros((lambertW(ch,d)+1,),int) # number of tensors of each rank
    while ch>0:
        max_rank = lambertW(ch,d) # compute the max rank tensor that can fit up to
        Ns[:max_rank+1] += np.array([d**(max_rank-r) for r in range(max_rank+1)],dtype=int)
        ch -= (max_rank+1)*d**max_rank # compute leftover channels
    sum_rep = sum([binomial_allocation(nr,r,group) for r,nr in enumerate(Ns)])
    sum_rep,perm = sum_rep.canonicalize()
    return sum_rep

def lambertW(ch,d):
    """ Returns solution to x*d^x = ch rounded down."""
    max_rank=0
    while (max_rank+1)*d**max_rank <= ch:
        max_rank += 1
    max_rank -= 1
    return max_rank

def binomial_allocation(N,rank,G):
    """ Allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r to match the binomial distribution.
        For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N==0: return 0
    n_binoms = N//(2**rank)
    n_leftover = N%(2**rank)
    even_split = sum([n_binoms*int(binom(rank,k))*T(k,rank-k,G) for k in range(rank+1)])
    ps = np.random.binomial(rank,.5,n_leftover)
    ragged = sum([T(int(p),rank-int(p),G) for p in ps])
    out = even_split+ragged
    return out

def uniform_allocation(N,rank):
    """ Uniformly allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r. For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N==0: return 0
    even_split = sum((N//(rank+1))*T(k,rank-k) for k in range(rank+1))
    ragged = sum(random.sample([T(k,rank-k) for k in range(rank+1)],N%(rank+1)))
    return even_split+ragged

@export
class EMLP(Module,metaclass=Named):
    """ Equivariant MultiLayer Perceptron. 
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.

        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers

        Returns:
            Module: the EMLP objax module."""
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        super().__init__()
        logging.info("Initing EMLP (objax)")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        
        self.G=group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
        #assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        )
    def __call__(self,x,training=True):
        return self.network(x)

def swish(x):
    return jax.nn.sigmoid(x)*x

def MLPBlock(cin,cout):
    return Sequential(nn.Linear(cin,cout),swish)#,nn.BatchNorm0D(cout,momentum=.9),swish)#,

@export
class MLP(Module,metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[MLPBlock(cin,cout) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,x,training=True):
        y = self.net(x)
        return y

@export
class Standardize(Module):
    """ A convenience module to wrap a given module, normalize its input
        by some dataset x mean and std stats, and unnormalize its output by
        the dataset y mean and std stats. 

        Args:
            model (Module): model to wrap
            ds_stats ((μx,σx,μy,σy) or (μx,σx)): tuple of the normalization stats
        
        Returns:
            Module: Wrapped model with input normalization (and output unnormalization)"""
    def __init__(self,model,ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats=ds_stats
    def __call__(self,x,training):
        if len(self.ds_stats)==2:
            muin,sin = self.ds_stats
            return self.model((x-muin)/sin,training=training)
        else:
            muin,sin,muout,sout = self.ds_stats
            y = sout*self.model((x-muin)/sin,training=training)+muout
            return y



# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPode(Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[Sequential(nn.Linear(cin,cout),swish) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,z,t):
        return self.net(z)

@export
class EMLPode(EMLP):
    """ Neural ODE Equivariant MLP. Same args as EMLP."""
    #__doc__ += EMLP.__doc__.split('.')[1]
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        #super().__init__()
        logging.info("Initing EMLP")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G=group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
        #print(middle_layers[0].reps[0].G)
        #print(self.rep_in.G)
        reps = [self.rep_in]+middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        )
    def __call__(self,z,t):
        return self.network(z)

# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPH(Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[Sequential(nn.Linear(cin,cout),swish) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def H(self,x):#,training=True):
        y = self.net(x).sum()
        return y
    def __call__(self,x):
        return self.H(x)

@export
class EMLPH(EMLP):
    """ Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""
    #__doc__ += EMLP.__doc__.split('.')[1]
    def H(self,x):#,training=True):
        y = self.network(x)
        return y.sum()
    def __call__(self,x):
        return self.H(x)

@export
@cache(maxsize=None)
def gate_indices(sumrep): #TODO: add support for mixed_tensors
    """ Indices for scalars, and also additional scalar gates
        added by gated(sumrep)"""
    assert isinstance(sumrep,SumRep), f"unexpected type for gate indices {type(sumrep)}"
    channels = sumrep.size()
    perm = sumrep.perm
    indices = np.arange(channels)
    num_nonscalars = 0
    i=0
    for rep in sumrep:
        if rep!=Scalar and not rep.is_permutation:
            indices[perm[i:i+rep.size()]] = channels+num_nonscalars
            num_nonscalars+=1
        i+=rep.size()
    return indices




##############################################################################
@export
def radial_basis_transform(x, nrad = 100):
    """
    x is a vector
    """
    xmax, xmin = x.max(), x.min()
    gamma = 2*(xmax - xmin)/(nrad - 1) 
    mu    = np.linspace(start=xmin, stop=xmax, num=nrad)
    return mu, gamma

@export
def radial_basis_transform_wl(x, nrad, mmin=0, mmax=10):
    """
    x is B x N x N x 1 matrix
    
    nrad is n_hidden
    """
    gamma = 2*(mmax - mmin)/(nrad - 1)
    mu    = np.linspace(start=mmin, stop=mmax, num=nrad)
    scalars = jnp.expand_dims(scalars, axis=-1) - jnp.expand_dims(self.mu, axis=0) #(n,30,n_rad)
    return jnp.exp(-gamma*(scalars**2)) #(n,30,n_rad)

class GaussianFourierProjection(Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.embedding_size = embedding_size
    initializer = jax.nn.initializers.he_normal()
    self.weight = initializer(jax.random.PRNGKey(42), (1, embedding_size), jnp.float32)

  def __call__(self, x):
    #apply dot product to get samples of cos and sin
    x_proj = jnp.multiply(jnp.expand_dims(x, axis=-1) * 2 * jnp.pi, self.weight) 
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


def comp_inner_products(x, take_sqrt=True):
    """
    INPUT: batch (q1, q2, p1, p2)
    N: number of datasets
    dim: dimension  
    x: numpy tensor of size [N, 4, dim] 
    """
   
    n = x.shape[0]
    scalars = np.einsum('bix,bjx->bij', x, x).reshape(n, -1) # (n,16)
    if take_sqrt:
        xxsqrt = np.sqrt(np.einsum('bix,bix->bi', x, x)) # (n,4)
        scalars = np.concatenate([xxsqrt, scalars], axis = -1)  # (n,20)
    return scalars 


@export
def compute_scalars(x):
    """Input x of dim [n, 4, 3]"""
    x = np.array(x)    
    xx = comp_inner_products(x)  # (n,20)

    g  = np.array([0,0,-1])
    xg = np.inner(g, x) # (n,4)

    y  = x[:,0,:] - x[:,1,:] # x1-x2 (n,3)
    yy = np.sum(y*y, axis = -1, keepdims=True) # <x1-x2, x1-x2> | (n,) 
    yy = np.concatenate([yy, np.sqrt(yy)], axis = -1) # (n,2)

    yx = np.einsum('bx,bjx->bj', y, x) # <q1-q2, u>, u=q1-q0, q2-q0, p1, p2 | (n, 4)
    
    scalars = np.concatenate([xx,xg,yy,yx], axis=-1) # (n,30)
    return scalars

def comp_inner_products_jax(x:jnp.ndarray, take_sqrt=True):
    """
    INPUT: batch (q1, q2, p1, p2)
    N: number of datasets
    dim: dimension  
    x: numpy tensor of size [N, 4, dim] 
    """ 
    n = x.shape[0]
    scalars = jnp.einsum('bix,bjx->bij', x, x).reshape(n, -1) # (n, 16)
    if take_sqrt:
        xxsqrt = jnp.sqrt(jnp.einsum('bix,bix->bi', x, x)) # (n, 4)
        scalars = jnp.concatenate([xxsqrt, scalars], axis = -1)  # (n, 20)
    return scalars 

def distance_squared_matrix(x:jnp.array):
    """
    Recieves x (B, N, 3)
    Returns dists (B, N, N) batch of squared distance matrices for each point cloud
    """
    # this has the same affect as taking the dot product of each row with itself
    x2 = jnp.sum(jnp.square(x), axis=2) # shape of (m)
    xy = jnp.matmul(x, jnp.transpose(x,axes=(0,2,1)))
    x2 = x2.reshape(x2.shape[0],-1, 1)
    x3 = x2.reshape(x2.shape[0],1, -1)
    dists = x3 - 2*xy + x2 # (m, 1) repeat columnwise + (m, n) + (n) repeat rowwise -> (m, n)
    return dists

def comp_dist_matrix_jax(x:jnp.array, g:jnp.array=jnp.array([0,0,-1])):
    """
    INPUT: batch (q1, q2, p1, p2)
    N: number of datasets
    dim: dimension
    x: numpy tensor of size [N, 4, dim]
    """
    n = x.shape[0] #TODO: take_sqrt ?
    repeat_g = jnp.tile(g, (n,1))
    repeat_g = jnp.expand_dims(repeat_g, axis=1)
    mat = jnp.concatenate([x, repeat_g], axis=1)
    batched_dist = distance_squared_matrix(mat)#tested
    #add norms of for momentum features
    norms = jnp.linalg.norm(x[:,2:], axis=2)
    two_zeros = jnp.zeros((x.shape[0], 2))
    one_zero  = jnp.zeros((x.shape[0], 1))
    norms = jnp.concatenate([two_zeros, norms, one_zero], axis=1)
    norms_diag = jnp.diagflat(norms)
    embed_diag = []
    stride = mat.shape[1]
    for k in range(0, norms_diag.shape[0], stride):
        embed_diag.append( jnp.expand_dims(norms_diag[k:(k+stride), k:(k+stride)], axis=0)  )#TODO: finish unpacking block diags
    norms_diag = jnp.concatenate(embed_diag, axis=0)
    #print(norms_diag)
    #assert add op is legal
    assert(norms_diag.shape == batched_dist.shape)
    return jnp.array(norms_diag + batched_dist) # (n, 16)

def compute_scalars_jax(x:jnp.ndarray, g:jnp.ndarray=jnp.array([0,0,-1])):
    """Input x of dim [n, 4, 3]"""     
    xx = comp_inner_products_jax(x)  # (n,20)

    xg = jnp.inner(g, x) # (n,4)

    y  = x[:,0,:] - x[:,1,:] # q1-q2 (n,3)
    yy = jnp.sum(y*y, axis = -1, keepdims=True) # <q1-q2, q1-q2> | (n,) 
    yy = jnp.concatenate([yy, jnp.sqrt(yy)], axis = -1) # (n,2)

    yx = jnp.einsum('bx,bjx->bj', y, x) # <q1-q2, u>, u=q1-q0, q2-q0, p1, p2 | (n, 4)

    scalars = jnp.concatenate([xx,xg,yy,yx], axis=-1) # (n,30)
    return scalars
    
def compute_scalars_jax_wl(x:jnp.ndarray, n_hidden: int, g:jnp.ndarray=jnp.array([0,0,-1])):
    """Input x of dim [n, 4, 3]"""    
    n = x.shape[0] 
    xx = comp_gram_matrix_jax(x, n_hidden)  # (n,16)
    
    layer = TwoFDisLayer(10)
    
    xx = layer(xx, )

    xg = jnp.inner(g, x) # (n,4)

    y  = x[:,0,:] - x[:,1,:] # q1-q2 (n,3)
    yy = jnp.sum(y*y, axis = -1, keepdims=True) # <q1-q2, q1-q2> | (n,) 
    yy = jnp.concatenate([yy, jnp.sqrt(yy)], axis = -1) # (n,2)

    yx = jnp.einsum('bx,bjx->bj', y, x) # <q1-q2, u>, u=q1-q0, q2-q0, p1, p2 | (n, 4)

    scalars = jnp.concatenate([xx,xg,yy,yx], axis=-1) # (n,10 + 16*n_hidden)
    return scalars

@export
class BasicMLP_objax(Module):
    def __init__(
        self, 
        n_in, 
        n_out,
        n_hidden=100, 
        n_layers=2, 
    ):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), F.relu]
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(F.relu)
        layers.append(nn.Linear(n_hidden, n_out))
        
        self.mlp = Sequential(*layers)
    
    def __call__(self,x,training=True):
        return self.mlp(x)

@export
class BasicMLP_objax_wl(Module):
    def __init__(
        self, 
        n_in, 
        n_out,
        n_hidden=32, 
        n_layers=2,
        final_lin=False
    ):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), F.relu]
        if not final_lin:
            for _ in range(n_layers-1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(F.relu)
            layers.append(nn.Linear(n_hidden, n_out))
            layers.append(F.relu)
        else:
            for _ in range(n_layers):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(F.relu)
            layers.append(nn.Linear(n_hidden, n_out))
        
        self.mlp = nn.Sequential(layers)
    
    def __call__(self,x,training=True):
        return self.mlp(x)
@export        
class BasicMLP_objax_wl(Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=32,
        n_layers=2,
        final_lin=False
    ):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), F.relu]
        if not final_lin:
            for _ in range(n_layers-1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(F.relu)
            layers.append(nn.Linear(n_hidden, n_out))
            layers.append(F.relu)
        else:
            for _ in range(n_layers):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(F.relu)
            layers.append(nn.Linear(n_hidden, n_out))

        self.mlp = nn.Sequential(layers)

    def __call__(self,x,training=True):
        return self.mlp(x)

def radial_basis_transform(x, nrad = 100):
    """
    x is a vector
    """
    xmax, xmin = x.max(), x.min()
    gamma = 2*(xmax - xmin)/(nrad - 1)
    mu    = np.linspace(start=xmin, stop=xmax, num=nrad)
    return mu, gamma

class TwoFDisLayer(Module): #TODO: test

    def __init__(self,
                 hidden_dim: int,
                 activation_fn = F.relu,
                 **kwargs
                 ):
        super().__init__()

        self.hidden_dim = hidden_dim
        
        self.fourier    = GaussianFourierProjection(embedding_size=self.hidden_dim//3)
        
        self.dist_linear = nn.Linear(2*(self.hidden_dim//3), hidden_dim, use_bias=False)

        self.emb_lin_0 = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim)

        self.emb_lin_1 = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim)

        self.emb_lin_2 = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim)

        self.output_lin = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim, final_lin=True)

    def radial_basis_transform(self,scalars, nrad, mmin=0, mmax=10):
        """
        x is a B x N x N x n_feat
        """
        gamma = 2*(mmax - mmin)/(nrad - 1)
        mu    = jnp.linspace(start=mmin, stop=mmax, num=nrad)
        scalars = jnp.expand_dims(scalars, axis=-1) - jnp.expand_dims(mu, axis=0) #(n,16,n_rad)
        scalars = jnp.cos(-gamma*(scalars)) #
        scalars = self.dist_linear(scalars)
        
        return scalars

    def forward(self,4
                dist_mat: jnp.ndarray,
                **kwargs
                ):
        '''
            kemb: (B, N, N, hidden_dim)
        '''
        kemb = comp_dist_matrix_jax(dist_mat)
        B = kemb.shape[0]
        N = kemb.shape[1]
        
        kemb = self.fourier(kemb)
        kemb = self.dist_linear(kemb)
#       kemb = self.radial_basis_transform(kemb, nrad=self.hidden_dim//3)

        self_message, kemb_0, kemb_1 = self.emb_lin_0(jnp.copy(kemb).reshape(-1, self.hidden_dim)), self.emb_lin_1(jnp.copy(kemb).reshape(-1, self.hidden_dim)), self.emb_lin_2(jnp.copy(kemb).reshape(-1, self.hidden_dim))

        self_message, kemb_0, kemb_1 = self_message.reshape((B,N,N,self.hidden_dim)), kemb_0.reshape((B,N,N,self.hidden_dim)), kemb_1.reshape((B,N,N,self.hidden_dim))

        kemb_0, kemb_1 = (jnp.transpose(kemb_0, (0, 3, 1, 2)), jnp.transpose(kemb_1, (0, 3, 1, 2)))

        kemb_multed = jnp.transpose(jnp.matmul(kemb_0, kemb_1), (0, 2, 3, 1))

        kemb_out = self.output_lin(self_message * kemb_multed) + (self_message * kemb_multed)

        return kemb_out

    def __call__(self,kemb: jnp.ndarray):
            return self.forward(kemb)

class TwoFDisLayerTwo(Module): #TODO: test

    def __init__(self,
                 hidden_dim: int,
                 activation_fn = F.relu,
                 **kwargs
                 ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.emb_lin_0 = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim)

        self.emb_lin_1 = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim)

        self.emb_lin_2 = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim)

        self.output_lin = BasicMLP_objax_wl(n_in=hidden_dim, n_out=hidden_dim, final_lin=True)



    def forward(self,
                kemb: jnp.ndarray,
                **kwargs
                ):
        '''
            kemb: (B, N, N, hidden_dim)
        '''

        B = kemb.shape[0]
        N = kemb.shape[1]

        self_message, kemb_0, kemb_1 = self.emb_lin_0(jnp.copy(kemb).reshape(-1, self.hidden_dim)), self.emb_lin_1(jnp.copy(kemb).reshape(-1, self.hidden_dim)), self.emb_lin_2(jnp.copy(kemb).reshape(-1, self.hidden_dim))

        self_message, kemb_0, kemb_1 = self_message.reshape((B,N,N,self.hidden_dim)), kemb_0.reshape((B,N,N,self.hidden_dim)), kemb_1.reshape((B,N,N,self.hidden_dim))

        kemb_0, kemb_1 = (jnp.transpose(kemb_0, (0, 3, 1, 2)), jnp.transpose(kemb_1, (0, 3, 1, 2)))

        kemb_multed = jnp.transpose(jnp.matmul(kemb_0, kemb_1), (0, 2, 3, 1))

        kemb_out = self.output_lin(self_message * kemb_multed) + (self_message * kemb_multed)

        return kemb_out

    def __call__(self,kemb: jnp.ndarray):
            return self.forward(kemb)


def two_order_sumpool(kemb): #TODO test
  """Computes the second-order sum pool of a kernel embedding tensor.

  Args:
    kemb: A JAX tensor of shape (batch_size, num_patches, num_patches, embedding_dim).

  Returns:
    A JAX tensor of shape (batch_size, 2 * embedding_dim).
  """
  batch_size, num_patches, _, embedding_dim = kemb.shape
  idx = jnp.arange(num_patches)

  # Diagonal elements.
  kemb_diag = kemb[:, idx, idx, :]
  sum_kemb_diag = jnp.sum(kemb_diag, axis=1)

  # Off-diagonal elements (excluding the diagonal).
  sum_kemb_offdiag = jnp.sum(kemb, axis=(1, 2)) - sum_kemb_diag

  # Concatenate diagonal and off-diagonal sums.
  output = jnp.concatenate((sum_kemb_diag, sum_kemb_offdiag), axis=-1)
  return output

class TwoOrderOutputBlock(Module):
    def __init__(self,
                 hidden_dim: int,
                 activation_fn: F.relu
                 ):
        super().__init__()
        self.output_fn = BasicMLP_objax(n_in=2*hidden_dim, n_out=1)

        self.sum_pooling = two_order_sumpool

    def forward(self,
                kemb: jnp.array
                ):

        output = self.output_fn(self.sum_pooling(kemb=kemb))
        return output
    def __call__(self,kemb: jnp.ndarray):
            return self.forward(kemb)



@export
class InvarianceLayer_objax(Module):
    def __init__(
        self,  
        n_hidden, 
        n_layers, 
    ):
        super().__init__()
        #self.mlp = BasicMLP_objax(
        #    n_in=30, n_out=1, n_hidden=n_hidden, n_layers=n_layers
        #) 
        
        #self.g = jnp.array([0,0,-1])
        self.two_fdis    = TwoFDisLayer(hidden_dim=12)
        self.tw_fdis_two = TwoFDisLayerTwo(hidden_dim=12)
        self.output      = TwoOrderOutputBlock(hidden_dim=12, activation_fn=F.relu)

    def H(self, x):
        out = self.two_fdis(scalars)
        out = self.two_fdis_two(out)
        out = self.output(out)
        return out
        
    def __call__(self, x:jnp.ndarray):
        x = x.reshape(-1,4,3) # (n,4,3)
        return self.H(x)

@export
class InvarianceLayerWL_objax(Module): #TODO only for Hamiltonian
    def __init__(
        self,  
        n_hidden, 
        n_layers, 
    ):
        super().__init__()
        self.mlp = BasicMLP_objax(
            n_in=30, n_out=1, n_hidden=n_hidden, n_layers=n_layers
        ) 
        self.g = jnp.array([0,0,-1])
        self.kDisGNN = kDisGNN(
            z_hidden_dim=model_config.z_hidden_dim,
            ef_dim=model_config.ef_dim,
            rbf=model_config.rbf,
            max_z=model_config.max_z,
            rbound_upper=model_config.rbound_upper,
            rbf_trainable=model_config.rbf_trainable,
            activation_fn=activation_fn_map[model_config.activation_fn_name],
            k_tuple_dim=model_config.k_tuple_dim,
            block_num=model_config.block_num,
            pooling_level=model_config.get("pooling_level"),
            e_mode=model_config.get("e_mode"),
            model_name=model_name,
            use_mult_lin=model_config.get("use_mult_lin"),
            interaction_residual=model_config.get("interaction_residual"),
            )
    
    def H(self, x):
        #scalars = compute_scalars_jax(x, self.g)
        dist_mat = comp_dist_matrix_jax(x, self.g)
        out = self.kDisGNN(dist_mat)
        return out
    
    def __call__(self, x:jnp.ndarray):
        x = x.reshape(-1,4,3) # (n,4,3)
        return self.H(x)


@export
class EquivarianceLayer_objax(Module):
    def __init__(
        self,  
        n_hidden, 
        n_layers,
        mu, 
        gamma
    ):
        super().__init__()  
        self.mu = mu # (n_rad,)
        self.gamma = gamma
        self.n_in_mlp = len(mu)*30
        self.mlp = BasicMLP_objax(
          n_in=self.n_in_mlp, n_out=24, n_hidden=n_hidden, n_layers=n_layers
        ) 
        self.g = jnp.array([0,0,-1])

    def __call__(self, x, t): 
        x = x.reshape(-1,4,3) # (n,4,3)
        scalars = compute_scalars_jax(x, self.g) # (n,30)
        scalars = jnp.expand_dims(scalars, axis=-1) - jnp.expand_dims(self.mu, axis=0) #(n,30,n_rad)
        scalars = jnp.exp(-self.gamma*(scalars**2)) #(n,30,n_rad)
        scalars = scalars.reshape(-1, self.n_in_mlp) #(n,26*n_rad)
        out = jnp.expand_dims(self.mlp(scalars), axis=-1) # (n,24,1)
         
        y = x[:,0,:] - x[:,1,:] # x1-x2 (n,3) 
        output = jnp.sum(out[:,:16].reshape(-1,4,4,1) * jnp.expand_dims(x, 1), axis=1) # (n,4,3)
        output = output + out[:,16:20] * jnp.expand_dims(y,1)                          # (n,4,3)
        output = output + out[:,20:] * jnp.expand_dims(self.g,1)                        # (n,4,3)
    
        # x1 = jnp.sum(out[:,0:4,:]  *x, axis = 1) + out[:,16,:] * y + out[:,20,:] * g #(n,3)
        # x2 = jnp.sum(out[:,4:8,:]  *x, axis = 1) + out[:,17,:] * y + out[:,21,:] * g #(n,3)
        # p1 = jnp.sum(out[:,8:12,:] *x, axis = 1) + out[:,18,:] * y + out[:,22,:] * g #(n,3)
        # p2 = jnp.sum(out[:,12:16,:]*x, axis = 1) + out[:,19,:] * y + out[:,23,:] * g #(n,3)
        # jnp.concatenate([x1,x2,p1,p2], axis=-1)
        return output.reshape(-1, 12) #(n,12)
 
