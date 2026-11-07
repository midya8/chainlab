"""Reproduce the HMM training and decoding-audit graphic from actual outputs."""
from pathlib import Path
from chainlab.demo import run_demo


def render():
    r=run_demo();history=r['training']['history'];parts=[]
    def add(s):parts.append(s)
    add('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="780" viewBox="0 0 900 780" role="img" aria-labelledby="title desc">')
    add('<title id="title">ChainLab: train sequences, distinguish marginal states from valid paths</title>')
    add('<desc id="desc">EM training log likelihood increases across20 updates. A separate exact three-state example has marginal modes A/A with a forbidden transition while Viterbi returns valid path B/A.</desc>')
    add('<rect width="900" height="780" fill="#fff"/><g font-family="Arial,sans-serif" fill="#251f21">')
    def text(x,y,s,size=16,weight='400',color='#251f21'):
        add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{s}</text>')
    text(40,40,'Learn sequence structure. Audit the path.',28,'600')
    text(40,69,'Baum-Welch training on 20 independent synthetic sequences, 40 symbols each.',16)
    left,top,w,h=90,130,740,205
    xs=lambda step:left+step/20*w
    ys=lambda ll:top+(-600-ll)/250*h
    text(40,108,'Training log likelihood (higher is better)',16)
    for ll in (-850,-800,-750,-700,-650,-600):
        add(f'<line x1="{left}" x2="{left+w}" y1="{ys(ll)}" y2="{ys(ll)}" stroke="#eae9ea"/>')
        text(43,ys(ll)+5,str(ll),14)
    for i in (0,5,10,15,20):text(xs(i)-6,top+h+24,str(i),14)
    add('<polyline points="'+' '.join(f'{xs(row["iteration"]):.2f},{ys(row["log_likelihood"]):.2f}' for row in history)+'" fill="none" stroke="#34715f" stroke-width="3"/>')
    text(380,top+h+46,'Accepted EM updates',15)
    text(40,406,f'Initial {history[0]["log_likelihood"]:.4f}; final {history[-1]["log_likelihood"]:.4f}. No smoothing or validation selection.',16)
    text(40,450,'Posterior modes are not a path decoder',23,'600')
    text(40,478,'Separate exact counterexample: observations x, x; transition A to A has probability zero.',16)
    posterior=r['posterior_mode_counterexample']['state_posteriors']
    text(40,518,'State posterior',15)
    for x,name in zip((300,470,640),('A','B','C')):text(x,518,name,16,'600')
    for i,row in enumerate(posterior):
        y=553+i*35;text(40,y,f'Time {i}',16)
        best=max(range(3),key=row.__getitem__)
        for j,(x,p) in enumerate(zip((300,470,640),row)):
            text(x,y,f'{p:.2f}',18,'600' if j==best else '400','#34715f' if j==best else '#251f21')
    text(40,637,'Marginal modes: A / A. Forbidden transition, joint probability zero.',18,'600','#b82e39')
    text(40,671,'Viterbi path: B / A. Valid; posterior probability 0.35.',18,'600','#34715f')
    text(40,713,'Marginals condition on the whole observed sequence. Do not multiply them as a path score.',15)
    text(40,742,'Development fixture, seed23. Finite EM stop; no true-state recovery or universal ranking claim.',15)
    add('</g></svg>')
    return '\n'.join(parts)+'\n'


if __name__=='__main__':Path('assets/sequence-audit.svg').write_text(render(),encoding='utf-8')
