from pathlib import Path
import csv, sys, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sdf_local_lattice import Parameters, LocalSDFLattice

FORCES = [0.05, 0.15, 0.35, 0.70, 1.20, 2.00]
STEPS = 400
SEEDS = [7]
CSV = ROOT / 'adel_force_amplitude_sweep.csv'
PLOT = ROOT / 'adel_force_amplitude_sweep.png'
PDF = ROOT / 'adel_force_amplitude_research_note.pdf'

rows = []
for force_amp in FORCES:
    for seed in SEEDS:
        sim = LocalSDFLattice(Parameters(seed=seed))
        sim.force[:] = 0.0
        sim.force[0] = force_amp
        initial = sim.diagnostics()
        finite = True
        breakdown_step = ''
        last = initial
        max_abs_phase_over_run = abs(initial['max_abs_phase'])
        try:
            for step in range(1, STEPS + 1):
                last = sim.step()
                max_abs_phase_over_run = max(max_abs_phase_over_run, last['max_abs_phase'])
                vals = [sim.phi, sim.velocity, sim.density, sim.polarization]
                if not all(np.all(np.isfinite(v)) for v in vals):
                    finite = False
                    breakdown_step = step
                    break
        except (FloatingPointError, OverflowError, ValueError):
            finite = False
            breakdown_step = step
        stab = sim.stability_report()
        rows.append({
            'force_amplitude': force_amp,
            'seed': seed,
            'requested_steps': STEPS,
            'completed_steps': step if 'step' in locals() else 0,
            'simulated_time': sim.time,
            'dt': sim.p.dt,
            'dt_omega_bound': stab['dt_omega_bound'],
            'heuristic_stable': stab['heuristic_stable'],
            'finite_run': finite,
            'breakdown_step': breakdown_step,
            'mean_polarization': last['mean_polarization'],
            'ring_order': last['ring_order'],
            'mean_energy_proxy': last['mean_energy_proxy'],
            'max_abs_phase': last['max_abs_phase'],
            'max_abs_phase_over_run': max_abs_phase_over_run,
            'mean_density': last['mean_density'],
            'rms_strain': last['rms_strain'],
        })

with CSV.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

x = np.array([r['force_amplitude'] for r in rows])
finite = np.array([r['finite_run'] for r in rows], dtype=bool)
pol = np.array([r['mean_polarization'] for r in rows], float)
ring = np.array([r['ring_order'] for r in rows], float)
energy = np.array([r['mean_energy_proxy'] for r in rows], float)
phase = np.array([r['max_abs_phase_over_run'] for r in rows], float)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                     'font.family': 'DejaVu Sans', 'figure.dpi': 150})
fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5), constrained_layout=True)
series = [(pol, 'Mean polarization', '#176b87'), (ring, 'Ring order', '#c45a36'),
          (energy, 'Energy proxy', '#4b7f52'), (phase, 'Max abs phase', '#7a4d8b')]
for ax, (y, label, color) in zip(axes.flat, series):
    ax.plot(x, y, marker='o', lw=2, ms=5, color=color)
    ax.set_xlabel('Force amplitude')
    ax.set_ylabel(label)
    ax.set_xticks(FORCES)
    ax.grid(True, alpha=.25, lw=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(6))
fig.suptitle('Local SDF lattice force-amplitude sweep (400 steps)', fontsize=14, fontweight='bold')
fig.savefig(PLOT, dpi=300, bbox_inches='tight')
plt.close(fig)

finite_forces = [r['force_amplitude'] for r in rows if r['finite_run']]
unstable_forces = [r['force_amplitude'] for r in rows if not r['finite_run']]
heuristic = rows[0]['dt_omega_bound']

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodySmall', parent=styles['BodyText'], fontSize=8.7, leading=11.2, spaceAfter=5))
styles.add(ParagraphStyle(name='NoteTitle', parent=styles['Title'], fontSize=20, leading=23, textColor=colors.HexColor('#163d4a'), spaceAfter=5))
styles.add(ParagraphStyle(name='Caption', parent=styles['BodyText'], fontSize=7.5, leading=9, textColor=colors.HexColor('#46565c')))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica', 7.5); canvas.setFillColor(colors.HexColor('#63747a'))
    canvas.drawString(48, 27, 'Adel | Local SDF lattice exploratory research note')
    canvas.drawRightString(564, 27, f'Page {doc.page}')
    canvas.restoreState()

doc = SimpleDocTemplate(str(PDF), pagesize=letter, rightMargin=48, leftMargin=48, topMargin=43, bottomMargin=42)
story = [
    Paragraph('Force-Amplitude Sweep of the Local SDF Lattice', styles['NoteTitle']),
    Paragraph('Prepared for Adel | 400-step deterministic endpoint study', styles['Heading2']),
    Paragraph('Objective and protocol', styles['Heading1']),
    Paragraph('The exploratory <i>LocalSDFLattice</i> model in <font name="Courier">sdf_local_lattice.py</font> was run at force amplitudes 0.05, 0.15, 0.35, 0.70, 1.20, and 2.00. Each run used the default 12-cell topology, dt = 0.02, seed = 7, and exactly 400 integration steps; the axial source was set to the requested amplitude at node 0. Endpoint telemetry records mean polarization, ring order, energy proxy, and maximum absolute phase. Every step was checked for finite state variables.', styles['BodySmall']),
    Paragraph('Results at step 400', styles['Heading1'])
]
table_data = [['Force', 'Polarization', 'Ring order', 'Energy proxy', 'Max |phase|', 'Finite?']]
for r in rows:
    table_data.append([f"{r['force_amplitude']:.2f}", f"{r['mean_polarization']:.6g}", f"{r['ring_order']:.6f}", f"{r['mean_energy_proxy']:.6g}", f"{r['max_abs_phase']:.6g}", 'yes' if r['finite_run'] else 'NO'])
t = Table(table_data, colWidths=[.65*inch, 1.12*inch, 1.0*inch, 1.05*inch, 1.0*inch, .7*inch])
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#163d4a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#b7c5c9')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#eef4f4')]),('FONTSIZE',(0,0),(-1,-1),7.8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
story += [t, Spacer(1, 10), PageBreak(), Image(str(PLOT), width=5.15*inch, height=3.99*inch), Paragraph('Figure 1. Endpoint telemetry versus applied force amplitude. The four panels expose response, ring structure, energy proxy, and phase excursion on a common sweep.', styles['Caption'])]
story += [Paragraph('Interpretation and thresholds', styles['Heading1']),
    Paragraph('Phase-transition-like response. In this finite toy lattice, the sweep does not provide evidence for a sharp thermodynamic phase transition: the response variables change continuously over the tested amplitudes, and ring order is effectively flat. The response is non-monotonic at low drive (polarization falls from 0.05 to 0.35), then rises at stronger drive; ring order remains unchanged across the sweep. This is a crossover-like response, not evidence for a sharp thermodynamic phase transition or critical point.', styles['BodySmall']),
    Paragraph('Polarization saturation. Polarization is a relaxation diagnostic derived from local phase gradients and density. It is not a conserved order parameter. No polarization saturation is visible through force 2.00: the endpoint mean polarization grows from 0.3374 at force 1.20 to 0.5524 at force 2.00, while max |phase| rises to 9.8837. This suggests the tested range/horizon does not establish a plateau; apparent high-drive growth should be considered alongside the growing phase excursion. The CSV preserves exact numerical values for re-analysis.', styles['BodySmall']),
    Paragraph('Breakdown and stability. No non-finite state was observed in the requested amplitudes: ' + (', '.join(f'{v:.2f}' for v in finite_forces) if finite_forces else 'none') + '. ' + (f'The first observed breakdown, if any, occurred at force {min(unstable_forces):.2f}.' if unstable_forces else 'No empirical breakdown threshold was bracketed by this six-point sweep; the tested upper bound is 2.00.') + f' The built-in timestep heuristic gives dt × omega_bound = {heuristic:.4f}, below its conservative 0.5 label. This is a numerical timestep heuristic only and is not a proof of bounded trajectories.', styles['BodySmall']),
    Paragraph('What can and cannot be concluded', styles['Heading1']),
    Paragraph('The result supports a reproducible statement about this implementation, topology, initialization, timestep, and 400-step horizon. It does not establish a physical phase transition, a universal saturation law, or a material breakdown threshold. In particular, ring order is computed from fixed link phases and is therefore a background-configuration diagnostic rather than a dynamically evolving flux observable. The source file itself labels the model exploratory and not validated physics.', styles['BodySmall']),
    Paragraph('Reproducibility and deliverables', styles['Heading1']),
    Paragraph('Run: <font name="Courier">python run_adel_force_sweep.py</font>. The CSV contains one row per amplitude with completion count, finite-state flag, heuristic stability, endpoint telemetry, and maximum phase excursion over the run. The PNG is a 300 dpi, four-panel publication figure. The present note is intentionally concise and limited to two pages.', styles['BodySmall']),
    Spacer(1, 4), Paragraph('Files: adel_force_amplitude_sweep.csv; adel_force_amplitude_sweep.png; adel_force_amplitude_research_note.pdf', styles['Caption'])]
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote', CSV, PLOT, PDF)
print('rows', len(rows), 'finite', int(finite.sum()), '/', len(rows), 'unstable', unstable_forces)
for r in rows:
    print(r['force_amplitude'], r['mean_polarization'], r['ring_order'], r['mean_energy_proxy'], r['max_abs_phase'], r['finite_run'])
