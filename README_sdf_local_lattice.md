# Local SDF Lattice: exploratory simulation

**Status:** exploratory phenomenological model, not validated physics. The available `Lattice Nodes` notes motivate tetrahedral local structure, phase steps, asymmetry/leakage, strain, density and polarization as concepts. They do not provide a uniquely derived microscopic Hamiltonian or calibrated parameter values. This package therefore makes its assumptions explicit and must not be read as evidence for SDF, galactic rotation claims, or a physical vacuum lattice.

## Files and running

- `sdf_local_lattice.py`: self-contained simulator; requires Python 3.9+ and NumPy.
- Run a demo: `python sdf_local_lattice.py --demo --steps 400 --seed 7`
- Run built-in checks: `python sdf_local_lattice.py --self-test`

The initial axial excitation is deterministic, and the seed controls the small initial phase perturbations. The demo prints a stability heuristic and initial/final diagnostics. The module does not require plotting libraries.

## Discrete model

There are `n_cells` local tetrahedra, each with four phase nodes and all six intra-tetrahedron links. Consecutive cells are connected by one axial link. Every link `(i,j)` carries a fixed link phase `A_ij`; the link response is

`I_ij = K sin(phi_j - phi_i - A_ij)`.

This depends on the gauge-invariant link-phase difference. The oriented sum of `A_ij` around each triangular face is reported modulo `2 pi` as a ring/flux diagnostic. The link phases are imposed background parameters, not a dynamically solved electromagnetic field. No `J^mu A_mu eta` term is used. Such a direct potential coupling would require an explicit gauge-covariant formulation and should not be asserted as established physics.

The node phase-like variable `phi` and its velocity evolve with damped second-order dynamics: the net link currents, density feedback, and an axial source determine acceleration. A phenomenological density variable relaxes toward a local edge-strain response plus a spatially decaying asymmetry/leakage source. A three-component polarization-like variable relaxes toward a response built from phase-gradient proxy and density. These are chosen constitutive rules, not consequences derived from tetrahedral geometry or known field theory.

Reported diagnostics are mean density, RMS link phase strain, mean polarization magnitude, mean cosine of face flux (`ring_order`), energy proxy, and maximum phase magnitude. The energy is a diagnostic proxy; damping, drive, density relaxation, and imposed link flux mean it is not generally conserved.

## Numerical stability and limitations

`dt * sqrt(2 K d_max / inertia)` is a conservative linearized graph-frequency heuristic. Values below 0.5 are flagged as a cautious operating region for this explicit velocity-Verlet update, not a proof of stability once nonlinear couplings are included. Reduce the timestep if trajectories grow or become non-finite. The code checks finiteness and refuses invalid parameters.

There is deliberately **no Nyquist condition presented as a physical law**. Sampling and integration resolution are numerical concerns; any signal-processing interpretation would require a separately specified observable, sampling protocol and frequency range. There is no imposed global speed limit, no inference from proton/electron speeds, and no coupling to cosmological inflation. No astrophysical calibration, uncertainty quantification, continuum limit, or empirical validation is supplied.

For research use, specify a target observable, derive or justify constitutive equations, perform timestep/mesh convergence studies, compare against null models and data, and report parameter identifiability before making physical claims.


---

# Phenomenology of Constraint, Law, and Substrate Debt

## 1. Scope and epistemic status

This chapter provides a shared interpretive layer for the SDF-Vortex local-lattice experiments and its relationship to Effective Distinction (ED). It is a phenomenological framework: it organizes observables, model assumptions, and failure conditions without claiming that the proposed substrate is an established physical entity. The words *constraint*, *law*, and *substrate* therefore have different roles and must not be silently exchanged.

- A **constraint** is an imposed, inferred, or dynamically maintained relation that restricts admissible states.
- A **law** is a reproducible rule, ideally expressed independently of a particular implementation, that maps specified conditions to consequences within a stated domain.
- A **substrate** is the representational carrier used by a model: a graph, lattice, field, state space, or update mechanism. Its presence in code does not establish a corresponding material medium in nature.
- A **phenomenon** is an operationally defined pattern in measured or simulated observables. A phenomenon is not yet an ontology.

The framework is useful only when these levels remain explicit. A simulation can demonstrate that a rule produces a pattern; it cannot, by that fact alone, demonstrate that the rule is a law of nature or that the implementation substrate exists physically.

## 2. Constraint as the primitive operational object

A constraint is represented by a compatibility condition over local or relational variables. In a graph formulation, node states, edge phase differences, holonomies, curvature-like residuals, and defect counts are candidate observables. In a lattice formulation, local phase steps, asymmetry, leakage, strain, density, polarization, and circulation are candidate observables. The model should state for every quantity:

1. its domain and units (or dimensionless normalization);
2. whether it is imposed, inferred, or emergent;
3. the update rule that changes it;
4. the measurement protocol used to report it; and
5. the perturbation or null test that could make the result disappear.

A useful generic residual is

$$
R_C(x) = C(x) - C_\mathrm{target},
$$

with an aggregate incompatibility or cost

$$
\Phi_C(x) = \sum_i w_i\,\rho_i(R_{C_i}(x)),
$$

where the weights, penalty functions, boundary conditions, and update schedule are part of the model specification. A nonzero residual is evidence of mismatch relative to the chosen constraints; it is not, without independent validation, evidence of force, energy, matter, or a new physical field.

## 3. From constraint to phenomenology

The minimal causal chain is:

> specified constraints -> state mismatch -> allowed update rule -> observable response -> reproducibility and falsification

The chain is broken if an observable is introduced after the fact, if the boundary conditions supply the claimed effect, or if the update rule contains an unreported source term. For this reason, ED and SDF-Vortex results should report the baseline, initial state, topology, boundary conditions, numerical resolution, random seed or ensemble protocol, conservation checks, and sensitivity to parameter changes.

A robust phenomenological claim has the form: under protocol P, observable O changes by amount D relative to control Q, with uncertainty U, across a stated parameter range. It should not be phrased as “the substrate creates” an effect unless the substrate is independently justified and the causal mechanism is tested.

## 4. Law, closure, and model choice

A law-like statement requires more than a visually coherent trajectory. At minimum it requires a declared state space, a closed evolution rule, invariance or covariance properties appropriate to the claim, reproducibility under independent implementations, and a domain in which predictions can fail. Conservation-like behavior must be checked rather than assumed; numerical stability must be separated from physical conservation; and fitted parameters must be distinguished from parameters fixed before observing the target result.

The following hierarchy keeps claims proportional to evidence:

1. **Implementation fact:** a particular program or derivation produces a specified output.
2. **Model regularity:** the output persists under seeds, resolution, perturbations, and independent reimplementation.
3. **Phenomenological relation:** an observable relation is predictive within a defined regime.
4. **Effective law:** the relation has a stable domain, stated symmetries, and known breakdown conditions.
5. **Physical law:** the relation survives comparison with independent empirical evidence and competing explanations.

The present repositories primarily support levels 1 and 2, and may motivate level-3 tests. They do not by themselves establish levels 4 or 5.

## 5. Substrate debt

**Substrate debt** is the accumulated explanatory and technical burden created when a model relies on an implicit carrier, hidden state, untracked boundary, or uncalibrated update mechanism. It is analogous to technical debt, but concerns interpretation as well as implementation. Substrate debt grows when:

- a graph or lattice is treated as physically privileged without a comparison class;
- boundary conditions inject order, momentum, circulation, or energy attributed to the interior;
- dissipation, noise, projection, clipping, or normalization is omitted from the accounting;
- a phenomenological parameter is presented as a derived constant;
- an observable depends on coordinate choice, mesh orientation, or discretization;
- a solver artifact is interpreted as a stable excitation; or
- metaphorical language replaces an operational definition.

Substrate debt is reduced by publishing the full state and update rule, exposing all source and sink terms, comparing alternative discretizations and null models, performing ablations, and recording failed regimes. It is not reduced by adding more interpretive vocabulary.

A practical debt register should list each hidden assumption, its observable consequence, the test that could reveal it, and the status of that test. Debt is acceptable during exploration only when it is named, bounded, and scheduled for resolution.

## 6. Epistemic integration across ED and SDF-Vortex

Effective Distinction supplies a relational language for distinctions, compatibility, holonomy, curvature-like residuals, and constrained dynamics. SDF-Vortex supplies exploratory local-lattice simulations involving phase structure, circulation, relaxation, polarization, and defect-like behavior. Their concepts can be compared at the level of operational observables, but one should not infer identity between their substrates or equations merely because the vocabulary overlaps.

The integration protocol is therefore: (a) define the observable independently of the narrative; (b) implement ED and SDF-Vortex controls with matched initial and boundary conditions where possible; (c) compare dimensionless response curves and uncertainty, not only snapshots; (d) identify which behavior is topology-dependent, discretization-dependent, or rule-dependent; and (e) record incompatibilities rather than forcing a unifying interpretation. A shared term such as “vortex,” “defect,” or “strain” is a hypothesis about correspondence, not evidence of correspondence.

## 7. Minimum validation and falsification suite

For any claimed emergent effect, the minimum suite is:

- a zero-distinction or uniform-state control;
- a boundary/source ablation;
- a resolution and timestep convergence check;
- a topology or connectivity comparison;
- a seed/ensemble reproducibility analysis;
- a conservation and budget audit for every claimed conserved quantity;
- a parameter-scramble or placebo test; and
- a predeclared failure criterion.

A result should be downgraded when it vanishes under a reasonable control, changes qualitatively with discretization, or depends on a parameter selected after inspecting the outcome. Negative results are epistemically productive: they identify the domain in which the proposed constraint-response mechanism is not sufficient.

## 8. Commitments for future work

Future documents and experiments should label every statement as one of: **definition**, **assumption**, **implementation fact**, **empirical observation**, **model prediction**, or **interpretation**. Each figure should identify controls and normalization. Each reported invariant should include the numerical tolerance and the operations under which it is expected to hold. Each physical analogy should be accompanied by a sentence stating what would count as disconfirmation.

The central commitment is modest but strict: constraints may organize observable behavior, and a substrate may be a productive computational representation. Neither claim becomes a law or an ontology until the intervening assumptions, substrate debt, and empirical tests have been made explicit.
