# qvarnet tutorials

Start here if you're new to the package.

| Notebook | Read it for |
|---|---|
| **`qvarnet_tour.ipynb`** | The big picture — a guided tour of **everything the package offers**: every model, Hamiltonian, sampler, the training API, results, observables, diagnostics, and the advanced subsystems (SR/QGT, TDVP, callbacks, discrete/spin, sharding, CLI). Runnable core + copy-paste patterns. |
| **`tutorial.ipynb`** | A single, fully-annotated end-to-end run — **every API call explained**, and every way to extract results from `TrainResult`. The "minimal complete example". |

Suggested path: skim `qvarnet_tour.ipynb` to see the whole surface, then run `tutorial.ipynb`
for one concrete training, then look at the real studies in `../calogero-sutherland/` and
`../soft_sphere_gas/`.

Both notebooks save figures to PNG (so plots appear even on a headless/`Agg` matplotlib backend)
and use a small, fast system so they run in well under a minute on CPU.
