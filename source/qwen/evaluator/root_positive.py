"""Independent positive control for the polynomial-root property oracle."""
import math
import numpy as np
import numpy.polynomial.polynomial as poly
from decimal import Decimal,localcontext

def find_zero(xs):
    def residual(x):
        try:return abs(sum(c*math.pow(x,i) for i,c in enumerate(xs)))
        except (OverflowError,ValueError):return float('inf')
    candidates=[0.]
    roots=poly.polyroots(xs)
    for z in roots:
        if abs(z.imag)>1e-3:continue
        x=float(z.real)
        candidates.extend([x,float(round(x))])
        lo=hi=x
        for _ in range(32):
            lo=math.nextafter(lo,-math.inf);hi=math.nextafter(hi,math.inf)
            candidates.extend([lo,hi])
    best=min(candidates,key=residual)
    if residual(best)<=1e-8:return best
    with localcontext() as ctx:
        ctx.prec=100
        coeff=[Decimal(int(c)) for c in xs]
        for z in roots:
            if abs(z.imag)>1e-3:continue
            x=Decimal(str(z.real))
            for _ in range(80):
                y=Decimal(0);dy=Decimal(0)
                for c in reversed(coeff):dy=dy*x+y;y=y*x+c
                if not dy:break
                nxt=x-y/dy
                if nxt==x:break
                x=nxt
            xf=float(x);lo=hi=xf;candidates.append(xf)
            for _ in range(64):
                lo=math.nextafter(lo,-math.inf);hi=math.nextafter(hi,math.inf)
                candidates.extend([lo,hi])
    return min(candidates,key=residual)

if __name__=='__main__':
    from evalplus.data import get_human_eval_plus
    from evalplus.eval._special_oracle import _poly
    t=get_human_eval_plus()['HumanEval/32']
    bad=[]
    for i,x in enumerate(t['plus_input']):
        residual=abs(_poly(*x,find_zero(*x)))
        if residual>t['atol']:bad.append({'index':i,'residual':residual})
    print('Independent root control violations',bad)
