"""Render the frozen relation selection diagnostic with discrete endpoints."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dacf_research.prepare_inputs import read
from partial_modules_stage2.inputs import H

if __name__=='__main__':
    eval_=read(H/'evaluation/result.json')['equal_originals']
    rows=read(H/'scope_run/control/result.json')['relations']['g0']
    plans=read(H/'plan/tree.json')['clock']
    times={r['frame']:r['seconds'] for r in plans}
    palette={'time':'#606670','gray':'#5974a5','hsv':'#5974a5','edge':'#5974a5','clip':'#5974a5',
             'round_robin':'#6b4c91'}
    fig,(top,chart)=plt.subplots(2,1,figsize=(12.2,8.1),gridspec_kw={'height_ratios':[1,2.55]},sharex=True,layout='constrained')
    for span in eval_['catalogue_roles'].values():
        a,b=span['truth_frame_window']
        for ax in (top,chart):ax.axvspan(times[a],times[b],color='#f4d28a',alpha=.38,lw=0)
    for axis,key,color in [('X','ref_x','#1a7f75'),('Y','ref_y','#b05e39')]:
        chosen=[(r['pts'][1]/1000,r['axes'][key]) for r in rows if r['axes'][key] is not None]
        top.vlines([a for a,b in chosen],0,[b for a,b in chosen],color=color,alpha=.67,lw=2)
        top.scatter([a for a,b in chosen],[b for a,b in chosen],color=color,s=18,label=axis)
    top.axhline(0,color='#637080',lw=.8,alpha=.35);top.set_ylabel('Signed projected change\n(native px per sampled edge)')
    top.legend(title='First local episode, relative to rail',loc='upper right',frameon=False)
    top.set_title('Scoped relation evidence and equal-original requests',loc='left',fontweight='bold',fontsize=15)
    sorted_names=['time','gray','hsv','edge','clip','ref_x','ref_y','ref_gap','peer_x','peer_y','peer_gap',
                  'own_luma','own_color','relative_luma','relative_color','round_robin']
    values={r['route']:r for r in eval_['arms']}
    for y,r in enumerate(sorted_names):
        chart.axhline(y,lw=.55,color='#d3d8dd',zorder=0)
        chart.scatter([times[f] for f in values[r]['new_frame_indices']],[y]*3,color=palette.get(r,'#1a7f75'),s=46,edgecolors='white',linewidths=.6,zorder=3)
    for t in eval_['initial_seed_pts']:chart.axvline(t/1000,color='#64727f',ls=':',lw=1,alpha=.9)
    chart.set_yticks(range(len(sorted_names)),[r.replace('_',' ') for r in sorted_names])
    chart.set_xlim(-.1,8.03);chart.set_ylim(len(sorted_names)-.55,-.65)
    chart.set_xlabel('Source seconds')
    chart.legend(handles=[Line2D([0],[0],color='#64727f',ls=':',label='Common initial original'),
        Line2D([0],[0],color='#d5a73a',lw=6,alpha=.38,label='Sealed evaluation window')],loc='lower right',frameon=True)
    fig.savefig(H/'evaluation/relationship_selection.png',dpi=152,facecolor='white')
