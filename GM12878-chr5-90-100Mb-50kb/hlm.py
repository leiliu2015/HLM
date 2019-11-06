#
# Copyright (C) 2019 Lei Liu & Changbong Hyeon
# This file is part of HLM.
#
import os
import sys
from numpy import *
from numba import jit
from scipy.special import erf
from scipy.optimize import minimize
from sklearn.decomposition import PCA

#
set_printoptions(precision=3, linewidth=200)

#
@jit(nopython=True)
def calPs(cm, N):
    ps = zeros(N)
    ns = zeros(N)
    for i in range(0, N):
        for j in range(i+1, N):
            s = j-i
            if not isnan(cm[i,j]):
                ps[s] += cm[i,j]
                ns[s] += 1.0
    for s in range(0, N):
        if ns[s] > 0:
            ps[s] = ps[s]/float(ns[s])
    ps[0] = 1.0
    #
    return ps

#
@jit
def pickij(oe, oemin, N, L, W, sa, sb):
    mx = oe*1.0
    for i in range(0, N):
        for j in range(0, i+1):
            s = i - j
            if isnan(mx[i,j]) or (s<sa) or (s>=sb):
                mx[j,i] = oemin
            #
            mx[i,j] = oemin
    #
    ij = []
    while len(ij) < L:
        ix= argmax(mx)
        i = ix/N
        j = ix%N
        #
        if mx[i,j] <= oemin:
            break
        else:
            ij.append([i,j])
        #
        for u in range(i-W, i+W+1):
            for v in range(j-W, j+W+1):
                if (u<v) and (0<=u<N) and (0<=v<N):
                    mx[u,v] = oemin
    #
    ij = array(ij).astype(int)
    return ij

# given g_{ij} and r_cut, calculate contact probability p_{ij}
def cpij(gij, rc):
    cp = -2.0*rc*sqrt(gij/pi)*exp(-gij*rc*rc)
    cp+= erf(sqrt(gij)*rc)
    return cp

# given p_{ij} and r_cut, calculate gamma_{ij}
def zbrent(rc, cp, x1, x2):
    #default
    ITMAX = 500
    EPS = 1.0e-16 # machine float-point precision 
    tol = 1.0e-16 # convergent criteria
    #
    a = x1
    b = x2
    c = x2
    #
    fa= cpij(a, rc) - cp
    fb= cpij(b, rc) - cp
    #
    if (fa>0 and fb>0) or (fa<0 and fb<0):
        print('Root must be bracketed in zbrent')
        sys.exit()
    fc= fb
    #
    for iter in range(0, ITMAX):
        if (fb>0 and fc>0) or (fb<0 and fc<0):
            c = a
            fc= fa
            d = b-a
            e = b-a
        if (abs(fc) < abs(fb)):
            a = b
            b = c
            c = a
            fa= fb
            fb= fc
            fc= fa
        tol1 = 2.0*EPS*abs(b) + 0.5*tol
        xm= (c-b)/2.0
        if (abs(xm)<=tol1 or fb==0):
            return b
        #
        if (abs(e)>=tol1 and abs(fa)>abs(fb)):
            s = fb/fa
            if (a==c):
                p = 2.0*xm*s
                q = 1.0-s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.0*xm*q*(q-r) - (b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            #
            if (p>0):
                q = -q
            p = abs(p)
            min1 = 3.0*xm*q - abs(tol1*q)
            min2 = abs(e*q)
            #
            if (2.0*p < min(min1, min2)):
                e = d
                d = p/q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        #
        a = b
        fa= fb
        if (abs(d) > tol1):
            b += d
        else:
            # nrutil.h - SIGN(a,b)
            if xm >= 0:
                b += abs(tol1)
            else:
                b -= abs(tol1)
        #
        fb = cpij(b, rc) - cp
    #
    print('Maximum double of iterations exceeded in zbrent\n')
    sys.exit()

# calculate the weights w_{ij}
@jit
def calWs(ij, N, L):
    nx = zeros(N)
    for i,j in ij:
        nx[j-i] += 1.0
    #
    for s in range(0, N):
        if nx[s] > 0:
            nx[s] = 1.0/nx[s]
    #
    ws = zeros(L)
    for p in range(0, L):
        s = ij[p,1] - ij[p,0]
        ws[p] = nx[s]
    #
    sw= sum(ws)
    ws /= float(sw)
    return ws

# \sum_{p} w_{p} ( r^{2}_{p}(k) - r^{2}_{p,true} )^{2}, args = (N, [i,j], w, [r^{2}_{p,true}])
@jit
def func(x, *args):
    N, ij, wz, rz = args[0], args[1], args[2], args[3]
    #
    L = len(ij)
    k0= 3.0
    #
    km= zeros((N, N))
    for i in range(1, N):
        km[i-1,i] = -k0
        km[i,i-1] = -k0
    for p in range(0, L):
        if x[p] > 0:
            i = ij[p,0]
            j = ij[p,1]
            km[i,j] = -x[p]
            km[j,i] = -x[p]
    for i in range(0, N):
        sk = -sum(km[i])
        km[i,i] = sk
    S = linalg.inv(km[1:,1:])
    # r^{2}_{p}(k)
    rk= zeros(L)
    for p in range(0, L):
        i = ij[p,0]
        j = ij[p,1]
        a = i-1
        b = j-1
        if i == 0:
            rk[p] = 3.0*S[b,b]
        else:
            rk[p] = 3.0*( S[a,a]+S[b,b]-2.0*S[a,b] )
    #
    dr2= (rk-rz)**2
    fx = dot(wz, dr2)
    return fx

#
@jit
def ka2r2(ka, N, L, ij):
    k0= 3.0
    #
    km= zeros((N, N))
    for i in range(1, N):
        km[i-1,i] = -k0
        km[i,i-1] = -k0
    for p in range(0, L):
        i = ij[p,0]
        j = ij[p,1]
        km[i,j] = -ka[p]
        km[j,i] = -ka[p]
    for i in range(0, N):
        sk = -sum(km[i])
        km[i,i] = sk
    S = linalg.inv(km[1:,1:])
    # r^{2}_{p}(k)
    rk= zeros(L)
    for p in range(0, L):
        i = ij[p,0]
        j = ij[p,1]
        a = i-1
        b = j-1
        if i == 0:
            rk[p] = 3.0*S[b,b]
        else:
            rk[p] = 3.0*( S[a,a]+S[b,b]-2.0*S[a,b] )
    return rk

#
@jit
def ka2cm(ka, N, L, ij, rc):
    k0= 3.0
    #
    km= zeros((N, N))
    for i in range(1, N):
        km[i-1,i] = -k0
        km[i,i-1] = -k0
    for p in range(0, L):
        i = ij[p,0]
        j = ij[p,1]
        km[i,j] = -ka[p]
        km[j,i] = -ka[p]
    for i in range(0, N):
        sk = -sum(km[i])
        km[i,i] = sk
    S = linalg.inv(km[1:,1:])
    # p_{ij}
    cm= zeros((N, N))
    for i in range(0, N):
        for j in range(i+1, N):
            a = i-1
            b = j-1
            if i == 0:
                g = 0.5/S[b,b]
            else:
                g = 0.5/(S[a,a]+S[b,b]-2.0*S[a,b])
            #
            cij = -2.0*rc*sqrt(g/pi)*exp(-g*rc*rc) + erf(sqrt(g)*rc)
            #
            cm[i,j] = cij
            cm[j,i] = cij
    return cm

#
@jit(nopython=True)
def ka2km(ka, N, L, ij):
    k0= 3.0
    #
    km= zeros((N, N))
    for i in range(1, N):
        km[i-1,i] = k0
        km[i,i-1] = k0
    for p in range(0, L):
        i = ij[p,0]
        j = ij[p,1]
        km[i,j] = ka[p]
        km[j,i] = ka[p]
    return km

#
@jit
def embedij(ij, mx, N):
    pm = ones((N, N))*nan
    for i,j in ij:
        pm[i,j] = mx[i,j]
        pm[j,i] = mx[i,j]
    return pm

#
def savMxTx(fn, cmt, mx):
    M, N  = shape(mx)
    #
    fw = open(fn, 'w')
    #
    if not cmt == '':
        fw.write(cmt+'\n')
    #
    for i in range(0, M):
        lt = ''
        for j in range(0, N):
            if isnan(mx[i,j]):
                lt += "%12s " % ('NaN')
            else:
                lt += "%+12.5e " % (mx[i,j])
        fw.write(lt+'\n')
    fw.close()
    #
    return

# Main Func. #################################
if not len(sys.argv) == 2:
    print('usage:: python hlm-hic.py chr5_50kb.090-100Mb.Hi-C.cm')
    sys.exit()
#
N = 200 # number of monomers: 10Mb/50kb
rc= 1.0
L = int(2*N)
W = 2
sa= 2
sb= N
oemin = 1.0
random.seed(12345)
#
fo= 'chr5_50kb.090-100Mb'

# readin contact matrix p_{ij}
fc = fo+'.Hi-C.cm'
#
if not os.path.isfile(fc):
    print('Cannot find contact probability matrix')
    sys.exit()
else:
    #
    fr = open(fc, 'r')
    cm = []
    for line in fr.readlines():
        if not line[0] == '#':
            lt = line.strip()
            lt = lt.split()
            cm.append( array(map(float, lt)) )
    fr.close()
    cm = array(cm)
    #
    if not len(cm) == N:
        print('please check matrix size in contact probability matrix')
        sys.exit()
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # missed/centromere region
    zi= []
    vi= []
    for i in range(0, N):
        if isnan(cm[i,i]):
            zi.append(i)
        else:
            vi.append(i)

# p(s)
ps = calPs(cm, N)
# o/e
oe = eye(N)
for i in range(0, N):
    for j in range(i+1, N):
        oex = cm[i,j]/ps[j-i]
        oe[i,j] = oex
        oe[j,i] = oex
# pickout significant p_{ij}
ij= pickij(oe, oemin, N, L, W, sa, sb)
L = len(ij)

# target p_{ij} & r^{2}_{ij}
p1= cpij(1.5, rc)
cf= p1/ps[1] # scaling factor of Hi-C
c0= cm[ (ij[:,0], ij[:,1]) ]
r0= zeros(L)
for p in range(0, L):
    cij = min(c0[p]*cf, p1)
    gij = zbrent(rc, cij, 0, 100)
    r0[p] = 1.5/gij

# weight w_{ij}
w0= calWs(ij, N, L) # weight over s
w0= w0/(r0**2) #w0= w0/(r0**4)

# constrained optimization
x0 = zeros(L)+0.0
xp = (N, ij, w0, r0,) # args of func
bd = [[0,10] for p in range(0, L)]
#
ck = minimize(func, x0, args=xp, method='L-BFGS-B', bounds=bd, options={'disp':False,'maxfun':3000000})
print ck
#
if ck['success']:
    ka = ck['x']
    # r^{2}_{p}(k)
    rk = ka2r2(ka, N, L, ij)
    # {i,j,r^2_{true}, r^2_{k}}
    ra = zeros((L, 4))
    for p in range(0, L):
        ra[p,0] = ij[p,0]
        ra[p,1] = ij[p,1]
        ra[p,2] = r0[p]
        ra[p,3] = rk[p]
    savMxTx(fo+".oeL%dW%d.r2"%(L,W), "#shape: %d\n#i j r^2_{true}, r^2_{k}"%(L), ra)
else:
    sys.exit()

# output
savMxTx(fo+'.oe', "#shape: %d min: %+12.5e max: %+12.5e"%(N, oe.min(), oe.max()), oe)
savMxTx(fo+".oeL%dW%d"%(L,W), "#shape: %d"%(N), embedij(ij, oe, N))
savMxTx(fo+".oeL%dW%d.cm"%(L,W), "#shape: %d"%(N), ka2cm(ka, N, L, ij, rc))
savMxTx(fo+".oeL%dW%d.km"%(L,W), '', ka2km(ka, N, L, ij))
#
hlmkm = ka2km(ka, N, L, ij)
fw = open(fo+".oeL%dW%d.km.list"%(L,W), 'w')
for i in range(0, N):
    for j in range(i+1, N):
        if hlmkm[i,j]>0:
            fw.write("%3d %3d %+12.5e\n" % (i,j,hlmkm[i,j]))
            fw.write("%3d %3d %+12.5e\n" % (j,i,hlmkm[i,j]))
fw.close()

# calculate chi_{++/+-/--} by appling PCA analysis to o/e matrix
pc = PCA(n_components=1).fit(oe[vi][:,vi]).transform(oe[vi][:,vi])
pcf= zeros(N) # pca[i], float
idx= 0
for i in range(0, N):
    if i in zi:
        pcf[i] = nan
    else:
        pcf[i] = pc[idx]
        #
        idx += 1
#
oeab = zeros((3, 2)) # {--,-+,++}
for i in range(0, N):
    for j in range(i+1, N):
        k = sign(pcf[i]) + sign(pcf[j])
        if not isnan(k):
            if k == -2:
                oeab[0,0] += oe[i,j]
                oeab[0,1] += 1.0
            elif k == 0:
                oeab[1,0] += oe[i,j]
                oeab[1,1] += 1.0
            elif k == 2:
                oeab[2,0] += oe[i,j]
                oeab[2,1] += 1.0
            else:
                print('check oe')
                sys.exit()
#
for i in range(0, 3):
    if oeab[i,1] > 0:
        oeab[i,0] /= oeab[i,1]
# {-/+} -> {A/B}
if oeab[0,0] > oeab[2,0]:
    oeab = oeab[::-1]
    pcf *= -1.0
#
# save PCA analysis results
fw = open(fo+'.oePCA', 'w')
ct = "#AA: %11.5e %d\n" % (oeab[0,0], oeab[0,1])
ct+= "#AB: %11.5e %d\n" % (oeab[1,0], oeab[1,1])
ct+= "#BB: %11.5e %d\n" % (oeab[2,0], oeab[2,1])
ct+= "#i -/+ pca(i)"
fw.write(ct+'\n')
for i in range(0, N):
    k  = pcf[i]
    lt = "%4d " % (i)
    if isnan(k):
        lt += "%+2d %12s" % (-1, 'NaN') # treat nan as A type
    else:
        if k >= 0:
            lt += "%+2d %+12.5e" % (1, k)
        else:
            lt += "%+2d %+12.5e" % (-1,k)
    fw.write(lt+'\n')
fw.close()

# save P(s)
fw = open(fo+'.ps', 'w')
ct = "#shape: %d" % (N)
fw.write(ct+'\n')
for i in range(0, N):
    lt = "%4d " % (i)
    if isnan(ps[i]):
        lt += "%11s" % ('NaN')
    else:
        lt += "%11.5e" % (ps[i])
    fw.write(lt+'\n')
fw.close()

