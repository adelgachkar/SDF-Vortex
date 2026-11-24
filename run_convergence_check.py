import sdf_local_lattice_dynamic_ring as m

base = {
    "seed": 7,
    "drive_amplitude": 0.35,
    "ring_dynamic_coupling": 0.35,
}

print("TIME_STEP_REFINEMENT")
print("dt,steps,time,ring_order_dynamic,dynamic_link_rms")

for dt, steps in [(0.02, 400), (0.01, 800), (0.005, 1600)]:
    sim = m.LocalSDFLattice(m.Parameters(dt=dt, **base))
    result = sim.run(steps)[-1]
    print("{:.4f},{},{:.6f},{:.9f},{:.9f}".format(
        dt,
        steps,
        result["time"],
        result["ring_order_dynamic"],
        result["dynamic_link_rms"],
    ))

print("SIZE_SENSITIVITY")
print("n_cells,n_nodes,ring_order_dynamic,rms_strain")

for cells in [8, 12, 16]:
    sim = m.LocalSDFLattice(m.Parameters(n_cells=cells, **base))
    result = sim.run(400)[-1]
    print("{},{},{:.9f},{:.9f}".format(
        cells,
        cells * 4,
        result["ring_order_dynamic"],
        result["rms_strain"],
    ))
