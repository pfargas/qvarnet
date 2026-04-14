# Quick Reference Checklist

## Most Critical (Do First)

### Testing & Validation
- [ ] Create `tests/` directory with `conftest.py`
- [ ] Test: Harmonic oscillator recovers exact GS energy
- [ ] Test: Kinetic energy methods match (autodiff vs divergence vs log-space)
- [ ] Test: DeepSet is permutation-invariant
- [ ] Test: Sampler produces expected distribution

### Numerical Stability
- [ ] Consolidate epsilon values → `config/numerical_constants.py`
- [ ] Choose one kinetic energy method (recommend: log-space autodiff)
- [ ] Document why (comments + paper references)

### Train Function Refactoring (See TRAIN_REFACTORING_GUIDE.md)
- [ ] Phase 1: Extract setup & config
- [ ] Phase 2: Extract sampling
- [ ] Phase 3: Extract training step
- [ ] Phase 4: Extract state management
- [ ] Phase 5: Simplify main loop

---

## Important (Do Next)

### Code Quality
- [ ] Remove all commented code
- [ ] Remove `laplacian_OLD()`
- [ ] Resolve all `# FIXME` comments
- [ ] Remove global `stop_requested` state
- [ ] Add shape assertions to forward passes

### Configuration
- [ ] Convert dict configs to typed dataclasses
- [ ] Add validation in config loader
- [ ] Document each config parameter with type & default

### Documentation
- [ ] Document epsilon values and why (NUMERICAL_STABILITY.md)
- [ ] Document known limitations (KNOWN_ISSUES.md)
- [ ] Add docstrings to all public functions
- [ ] Add architecture diagram to docs

---

## Nice-to-Have (Do Later)

- [ ] Convergence diagnostics (R-hat, autocorrelation)
- [ ] Visualization suite (energy convergence, acceptance rate)
- [ ] Benchmark suite (known systems, compare literature)
- [ ] Performance profiling
- [ ] CI/CD pipeline with GitHub Actions

---

## File Structure After Refactoring

```
src/qvarnet/
├── config/
│   ├── numerical_constants.py    # ✨ NEW
│   ├── training_setup.py         # ✨ NEW
│   └── __init__.py
├── probability.py                # ✨ NEW
├── sampling_step.py              # ✨ NEW
├── training_step.py              # ✨ NEW
├── training_state.py             # ✨ NEW
├── train.py                       # REFACTORED (~80 lines)
├── models/
├── hamiltonian/
├── samplers/
├── utils/
└── ...

tests/
├── conftest.py                   # ✨ NEW (fixtures)
├── test_kinetic_energy.py        # ✨ NEW
├── test_hamiltonians.py          # ✨ NEW
├── test_models.py                # ✨ NEW
├── test_sampling_step.py         # ✨ NEW
├── test_training_step.py         # ✨ NEW
├── test_training_integration.py  # ✨ NEW
└── __init__.py
```

---

## Success Metrics

- [x] Read REFACTORING_PLAN.md
- [x] Read TRAIN_REFACTORING_GUIDE.md
- [ ] 50+ tests written
- [ ] All tests passing
- [ ] Harmonic oscillator passes ground-truth test
- [ ] DeepSet permutation-invariance verified
- [ ] train() split into 5 modules
- [ ] Zero global mutable state
- [ ] Zero commented code
- [ ] All epsilons centralized
- [ ] Full reproducibility test passing
- [ ] Numerical stability doc written

---

## Red Flags to Watch For

🚩 If results change after refactoring, something is wrong
- Solution: Compare energies at each epoch with original code
- Use git to diff old vs new train.py

🚩 If tests pass but physics is wrong
- Solution: Add more validation tests (compare vs analytical solutions)

🚩 If JAX complains about JIT issues after refactoring
- Solution: Check that functions don't capture outer scope
- All dependencies should be explicit arguments

🚩 If refactoring takes >20 hours
- Solution: You're overcomplicating it. Break into smaller PRs.

---

## Git Strategy

```bash
# Work on a feature branch
git checkout -b refactor/train-function

# After each phase:
git add src/qvarnet/config/
git commit -m "refactor: extract training setup to config module

- Parse sampler/training params into typed dataclasses
- Add validation for config parameters
- Tests: all existing tests still pass"

# Then:
git push origin refactor/train-function
# Open PR for review/testing

# After all phases:
git checkout main
git merge refactor/train-function
```

---

## Estimated Timeline

| Task | Hours | Difficulty |
|------|-------|-----------|
| Write validation tests | 8 | ⭐⭐ |
| Kinetic energy unification | 4 | ⭐ |
| Extract train components | 8 | ⭐⭐⭐ |
| Code cleanup | 6 | ⭐ |
| Documentation | 4 | ⭐ |
| **Total** | **30** | |

About 1-2 weeks of focused work, 4-6 hours/day.

---

## Questions to Ask Yourself Before Implementing

**Before touching code**:
1. Is there a test that would catch this bug?
2. Can I write that test first?
3. Will this change break user-facing API?

**While refactoring**:
1. Does this module have a single responsibility?
2. Are all dependencies explicit arguments (not global state)?
3. Can I test this function in isolation?
4. Does this need JAX JIT? (If yes, ensure it's pure)

**After refactoring**:
1. Do results match the original code?
2. Do tests pass?
3. Is the code more readable?
4. Could someone else understand this?

---

## Resources

- **JAX best practices**: https://jax.readthedocs.io/en/latest/design.html
- **Flax best practices**: https://flax.readthedocs.io/en/latest/
- **VMC reviews**: Look up "Variational Monte Carlo review" for physics context
- **pytest**: https://docs.pytest.org/

---

## Contact/Questions

If stuck on:

**JAX JIT issues**: Check if function has side effects or captures outer scope
**Shape bugs**: Add assertions at function entry/exit
**Test design**: Look at similar physics codebases (e.g., NetKet)
**Architecture**: Step back, ask "what is this module's single job?"

---

**Remember**: The goal is not perfect code. It's **correct code that you can trust**.

Start small. Test often. Refactor incrementally.

Good luck! 🚀
