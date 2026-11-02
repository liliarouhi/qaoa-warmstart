import numpy as np, itertools
rng=np.random.default_rng(0)
def rand_graph(n,p,weighted=True,integer=False):
    E=[];W=[]
    for i in range(n):
        for j in range(i+1,n):
            if rng.random()<p:
                E.append((i,j)); W.append(rng.integers(1,4) if integer else (rng.random() if weighted else 1.0))
    return E,np.array(W,float)
def cut_diag(n,E,W):
    z=np.arange(2**n); bits=((z[:,None]>>np.arange(n))&1)
    c=np.zeros(2**n)
    for (i,j),w in zip(E,W): c+=w*(bits[:,i]!=bits[:,j])
    return c
def mixer(psi,n,b):
    c,s=np.cos(b),-1j*np.sin(b)
    psi=psi.reshape([2]*n)
    for q in range(n):
        a0=np.take(psi,0,axis=q); a1=np.take(psi,1,axis=q)
        psi=np.stack([c*a0+s*a1,s*a0+c*a1],axis=q)
    return psi.reshape(-1)
def F(n,c,g,b):
    psi=np.ones(2**n,complex)/2**(n/2)
    for gl,bl in zip(g,b):
        psi=np.exp(-1j*gl*c)*psi; psi=mixer(psi,n,bl)
    return float(np.real(np.vdot(psi,c*psi)))
n=7;E,W=rand_graph(n,0.5);c=cut_diag(n,E,W)
g=rng.random(3)*3;b=rng.random(3)*3
f=F(n,c,g,b)
print("time reversal", abs(f-F(n,c,-g,-b)))
for l in range(3):
    bb=b.copy();bb[l]+=np.pi/2;print("beta shift pi/2 layer",l,abs(f-F(n,c,g,bb)))
Ei,Wi=rand_graph(n,0.5,integer=True);ci=cut_diag(n,Ei,Wi)
gg=g.copy();gg[1]+=2*np.pi;print("gamma 2pi integer weights",abs(F(n,ci,g,b)-F(n,ci,gg,b)))
gg=g.copy();gg[1]+=2*np.pi;print("gamma 2pi real weights (expect nonzero)",abs(f-F(n,c,gg,b)))
# p=1 analytic formula (Wang et al 2018) unweighted
Eu,Wu=rand_graph(n,0.5,weighted=False);cu=cut_diag(n,Eu,Wu)
adj=np.zeros((n,n));
for i,j in Eu: adj[i,j]=adj[j,i]=1
def f1(gm,bt):
    tot=0
    for u,v in Eu:
        du=adj[u].sum()-1; dv=adj[v].sum()-1; lam=(adj[u]*adj[v]).sum()
        tot+=0.5+0.25*np.sin(4*bt)*np.sin(gm)*(np.cos(gm)**du+np.cos(gm)**dv)-0.25*np.sin(2*bt)**2*np.cos(gm)**(du+dv-2*lam)*(1-np.cos(2*gm)**lam)
    return tot
for _ in range(3):
    gm,bt=rng.random()*3,rng.random()*3
    print("p1 formula err",abs(f1(gm,bt)-F(n,cu,[gm],[bt])))
# Mohar-Poljak bound
for _ in range(200):
    m=rng.integers(4,10);E2,W2=rand_graph(m,0.5)
    if not E2: continue
    L=np.zeros((m,m))
    for (i,j),w in zip(E2,W2): L[i,i]+=w;L[j,j]+=w;L[i,j]-=w;L[j,i]-=w
    mc=cut_diag(m,E2,W2).max()
    assert mc<=m*np.linalg.eigvalsh(L)[-1]/4+1e-9
print("Mohar-Poljak bound held on 200 random weighted graphs")
# QUBO -> MaxCut on n+1 vertices: min x^T Q x (x in {0,1}) equivalently max cut on graph with signed weights
m=6;Q=rng.normal(size=(m,m));Q=(Q+Q.T)/2
xs=np.array(list(itertools.product([0,1],repeat=m)))
qv=np.einsum('ki,ij,kj->k',xs,Q,xs)
# construct: x_i = (1 - s_i s_0)/2 with spins; cut indicator of edge(i,0) = x_i ; edge(i,j) cut = x_i xor x_j = x_i+x_j-2x_i x_j
# x^TQx = sum_i Q_ii x_i + sum_{i<j} 2Q_ij x_i x_j ; x_i x_j = (x_i+x_j - cut_ij)/2
# => = sum_i a_i x_i - sum_{i<j} Q_ij cut_ij, a_i = Q_ii + sum_{j!=i} Q_ij  ; x_i = cut_{i0}
a=np.diag(Q)+ (Q.sum(1)-np.diag(Q))
# minimize => maximize  -a_i cut_i0 + sum Q_ij cut_ij  : weights w_i0=-a_i, w_ij=Q_ij
E3=[(i+1,0) for i in range(m)]+[(i+1,j+1) for i in range(m) for j in range(i+1,m)]
W3=np.array([-a[i] for i in range(m)]+[Q[i,j] for i in range(m) for j in range(i+1,m)])
c3=cut_diag(m+1,[(min(e),max(e)) for e in E3],W3)
print("QUBO min",qv.min()," -maxcut",-c3.max(), "equal:",np.isclose(qv.min(),-c3.max()))
