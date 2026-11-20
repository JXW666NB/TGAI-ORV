import json, re, numpy as np
def rd(p):
    raw=open(p,'rb').read()
    for e in ('utf-8','utf-16','gbk'):
        try: t=raw.decode(e)
        except: continue
        if 'losses' in t: return t
    raise SystemExit(p)
def losses(p):
    t=rd(p); m=None
    for m in re.finditer(r'^losses\s+(\[.*\])\s*$', t, re.M): pass
    return np.asarray(json.loads(m.group(1)),float)
o=losses(r'lab/rent/q_out/orv_16L_200.log')
a=losses(r'lab/rent/q_out/pyt_match_16L_200.log')
b=losses(r'lab/rent/q_out/pyt_match_16L_200_s7.log')
print('n', len(o), len(a), len(b))
print('step  ORV      P1234    d      P7       d')
for i in list(range(0,20))+list(range(20,200,10))+[199]:
    print('%3d %8.4f %8.4f %+7.4f %8.4f %+7.4f' % (i,o[i],a[i],a[i]-o[i],b[i],b[i]-o[i]))
for lo,hi in ((0,10),(10,20),(20,40),(40,60),(60,80),(80,100),(100,120),(120,150),(150,200)):
    print('mean d [%3d,%3d)  P1234 %+0.4f   P7 %+0.4f   ORV mean %.4f' % (lo,hi,np.mean((a-o)[lo:hi]),np.mean((b-o)[lo:hi]),np.mean(o[lo:hi])))
print('last10: ORV %.4f  P1234 %.4f  P7 %.4f' % (np.mean(o[-10:]),np.mean(a[-10:]),np.mean(b[-10:])))
print('last20: ORV %.4f  P1234 %.4f  P7 %.4f' % (np.mean(o[-20:]),np.mean(a[-20:]),np.mean(b[-20:])))
print('P seed spread last20 = %.4f' % (np.mean(a[-20:])-np.mean(b[-20:])))
