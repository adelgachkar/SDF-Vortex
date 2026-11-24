from pathlib import Path
import csv, sys, math
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from sdf_local_lattice import Parameters, LocalSDFLattice
SEEDS=[3,7,11,19,29]
HORIZONS=[100,250,500,1000]
CSV=ROOT/'adel_sdf_parameter_sweep.csv'; PDF=ROOT/'adel_sdf_research_note.pdf'; PLOT=ROOT/'adel_sdf_sweep_plot.png'
rows=[]
for seed in SEEDS:
 for steps in HORIZONS:
  sim=LocalSDFLattice(Parameters(seed=seed)); initial=sim.diagnostics(); hist=sim.run(steps); final=hist[-1]; stab=sim.stability_report()
  pol_growth=final['mean_polarization']-initial['mean_polarization']
  rows.append(dict(seed=seed,steps=steps,simulated_time=final['time'],dt=sim.p.dt,dt_omega_bound=stab['dt_omega_bound'],heuristic_stable=stab['heuristic_stable'],initial_mean_polarization=initial['mean_polarization'],final_mean_polarization=final['mean_polarization'],polarization_growth=pol_growth,polarization_growth_per_step=pol_growth/steps,final_ring_order=final['ring_order'],ring_order_drift=final['ring_order']-initial['ring_order'],ring_order_min=min(x['ring_order'] for x in hist),ring_order_max=max(x['ring_order'] for x in hist),final_rms_strain=final['rms_strain'],final_energy_proxy=final['mean_energy_proxy'],max_abs_phase=final['max_abs_phase']))
with CSV.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# Mean trend by horizon
means=[]
for h in HORIZONS:
 r=[x for x in rows if x['steps']==h]
 means.append((h,np.mean([x['polarization_growth'] for x in r]),np.std([x['polarization_growth'] for x in r]),np.mean([x['final_ring_order'] for x in r])))
plt.figure(figsize=(8,4.3)); ax=plt.gca(); ax.errorbar([x[0] for x in means],[x[1] for x in means],yerr=[x[2] for x in means],marker='o',capsize=4,color='#146c7e',label='Polarization growth (mean ± SD)'); ax.set_xlabel('Integration steps'); ax.set_ylabel('Change in mean polarization'); ax.grid(alpha=.25); ax2=ax.twinx(); ax2.plot([x[0] for x in means],[x[3] for x in means],marker='s',color='#c45a36',label='Mean final ring order'); ax2.set_ylabel('Ring order'); ax.set_title('Seed-aggregated response across integration horizons'); plt.tight_layout(); plt.savefig(PLOT,dpi=180); plt.close()
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='Small',parent=styles['BodyText'],fontSize=8.5,leading=12)); styles.add(ParagraphStyle(name='TitleX',parent=styles['Title'],fontSize=23,leading=27,textColor=colors.HexColor('#163d4a')))
doc=SimpleDocTemplate(str(PDF),pagesize=letter,rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=44)
story=[Paragraph('Local SDF Lattice: Parameter-Sweep Research Note',styles['TitleX']),Paragraph('Prepared for Adel | Exploratory computational note',styles['Heading2']),Spacer(1,8),Paragraph(f'We evaluated {len(rows)} runs across {len(SEEDS)} deterministic seeds ({SEEDS}) and four integration horizons ({HORIZONS} steps), using the model defaults (12 cells, dt=0.02, default couplings). Each run starts from a newly initialized lattice. The CSV contains one row per seed-horizon run and includes endpoint telemetry, the stability heuristic, and ring-order extrema.',styles['BodyText']),Spacer(1,12),Paragraph('Summary by horizon',styles['Heading1'])]
data=[['Steps','Mean polarization growth (SD)','Mean final ring order','Heuristic stable runs']]
for h,g,sd,ro in means:
 stable=sum(x['heuristic_stable'] for x in rows if x['steps']==h)
 data.append([str(h),f'{g:.6g} ({sd:.2g})',f'{ro:.6f}',f'{stable}/{len(SEEDS)}'])
t=Table(data,colWidths=[.8*inch,2.4*inch,1.5*inch,1.45*inch]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#163d4a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#bdc9cc')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#eef4f4')]),('FONTSIZE',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)])); story += [t,Spacer(1,13),Image(str(PLOT),width=7.1*inch,height=3.8*inch),Paragraph('Figure 1. Seed-aggregated endpoint polarization growth and ring order; error bars are one standard deviation across five seeds.',styles['Small']),Spacer(1,12),Paragraph('Stability limits',styles['Heading1'])]
ratio=rows[0]['dt_omega_bound']; story.append(Paragraph(f'The implemented diagnostic reports dt × omega_bound = {ratio:.4f} for this parameterization and labels a run heuristic-stable when this ratio is below 0.5. All five seeds at every tested horizon receive the same stability classification because this bound depends on the parameters and graph, not on seed or run length. Thus the sweep finds no horizon-specific instability within {max(HORIZONS)} steps under the built-in criterion. This is not a proof of bounded trajectories: it is explicitly a timestep heuristic, and finite-state checks only catch non-finite values.',styles['BodyText']))
story += [Spacer(1,9),Paragraph('Mean polarization growth',styles['Heading1']),Paragraph('Across all runs, polarization change is computed as final minus initial mean node-wise polarization magnitude. Horizon-wise averages and between-seed standard deviations are given above; the growth-per-step and each seed-specific endpoint are retained in the CSV. This quantity is a model diagnostic, not a conserved quantity or validated physical observable. Different seeds vary only the initial phase perturbation in this implementation.',styles['BodyText']),Spacer(1,9),Paragraph('Ring-order invariants and caveats',styles['Heading1']),Paragraph('Ring order is the mean cosine of oriented face fluxes formed from fixed link phases. Since link_A is initialized once and is not evolved by step(), this diagnostic is invariant in time and across seeds for fixed parameters and topology. The recorded within-run minimum, maximum, and drift expose this check directly; drift should be zero up to floating-point precision. Its value characterizes the chosen background link configuration, not dynamically conserved flux. The lattice is explicitly described by its source as an exploratory phenomenological toy model, not validated physics; results should not be interpreted as empirical stability limits or physical predictions.',styles['BodyText']),Spacer(1,12),Paragraph('Reproducibility',styles['Heading1']),Paragraph('Implementation: LocalSDFLattice from sdf_local_lattice.py, unchanged defaults; NumPy seeded generators; seeds 3, 7, 11, 19, 29; horizons 100, 250, 500, 1000. CSV telemetry is at one record per run. Summary statistics use population-independent sample standard deviation (ddof=0) across the five seeds.',styles['Small'])]
doc.build(story)
print('CSV',CSV,'rows',len(rows)); print('PDF',PDF,'bytes',PDF.stat().st_size); print('metrics',means); print('ratio',ratio,'stable',sum(x['heuristic_stable'] for x in rows),'/',len(rows)); print('ring drifts',min(x['ring_order_drift'] for x in rows),max(x['ring_order_drift'] for x in rows))
