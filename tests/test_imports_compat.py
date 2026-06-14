"""Step 1 guard: old top-level import paths still resolve to the moved objects."""


def test_compat_shims_resolve_to_moved_objects():
    # Canonical documented import path (unambiguous: module → function).
    from qvarnet.geometry.qgt import QGTConfig as QGTConfig_new
    from qvarnet.probability import build_prob_fn as build_prob_fn_shim
    from qvarnet.qgt import DEFAULT_QGT_CONFIG
    from qvarnet.qgt import QGTConfig as QGTConfig_shim
    from qvarnet.train import train as train_shim
    from qvarnet.train_result import TrainResult as TrainResult_shim
    from qvarnet.training_step import compute_step as compute_step_shim
    from qvarnet.vmc import train as train_new
    from qvarnet.vmc.probability import build_prob_fn as build_prob_fn_new
    from qvarnet.vmc.train_result import TrainResult as TrainResult_new
    from qvarnet.vmc.training_step import compute_step as compute_step_new
    from qvarnet.vmc.vmc_state import VMCState as VMCState_new
    from qvarnet.vmc_state import VMCState as VMCState_shim

    # Each old top-level path re-exports the exact moved object.
    assert train_shim is train_new
    assert TrainResult_shim is TrainResult_new
    assert compute_step_shim is compute_step_new
    assert VMCState_shim is VMCState_new
    assert build_prob_fn_shim is build_prob_fn_new
    assert QGTConfig_shim is QGTConfig_new
    assert DEFAULT_QGT_CONFIG is not None

    # The re-exported objects are the expected kinds.
    assert callable(train_shim)
    assert isinstance(TrainResult_shim, type)
    assert callable(compute_step_shim)


def test_new_subpackages_importable():
    import qvarnet.diagnostics  # noqa: F401
    import qvarnet.geometry  # noqa: F401
    import qvarnet.observables  # noqa: F401
    import qvarnet.vmc  # noqa: F401
