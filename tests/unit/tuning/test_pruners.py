"""Unit tests for CostPruner, RuntimePruner, and ParetoPruner."""
import pytest

from ragtune.tuning.pruners import CostPruner, RuntimePruner, ParetoPruner, _pareto_front


# ── CostPruner ────────────────────────────────────────────────────────────────

class TestCostPruner:
    def _report(self, pruner, step, cost):
        pruner.report(step=step, n_total=100, obj1=0.5, obj2=cost, elapsed_ms=1000.0)

    def test_no_prune_before_warmup(self):
        p = CostPruner(max_mean_rerank_docs=10.0, warmup_steps=3)
        # Two steps of very high cost — still in warmup
        for s in range(1, 3):
            self._report(p, s, 1000.0)
            assert not p.should_prune()

    def test_prunes_when_mean_exceeds_limit(self):
        p = CostPruner(max_mean_rerank_docs=10.0, warmup_steps=3)
        for s in range(1, 6):
            self._report(p, s, 50.0)  # mean = 50 >> 10
        assert p.should_prune()

    def test_no_prune_within_limit(self):
        p = CostPruner(max_mean_rerank_docs=10.0, warmup_steps=3)
        for s in range(1, 6):
            self._report(p, s, 5.0)   # mean = 5 < 10
        assert not p.should_prune()

    def test_reset_clears_state(self):
        p = CostPruner(max_mean_rerank_docs=10.0, warmup_steps=1)
        self._report(p, 1, 1000.0)
        assert p.should_prune()
        p.reset()
        assert not p.should_prune()


# ── RuntimePruner ─────────────────────────────────────────────────────────────

class TestRuntimePruner:
    def _report(self, pruner, step, n_total, elapsed_ms):
        pruner.report(step=step, n_total=n_total, obj1=0.5, obj2=1.0, elapsed_ms=elapsed_ms)

    def test_no_prune_before_warmup(self):
        p = RuntimePruner(max_trial_seconds=10.0, warmup_steps=3)
        # 2 steps at 9 seconds total — warmup not complete
        self._report(p, 2, 100, 9_000.0)
        assert not p.should_prune()

    def test_prunes_when_projection_exceeds_limit(self):
        p = RuntimePruner(max_trial_seconds=10.0, warmup_steps=3)
        # 5 queries in 60 seconds; 100 queries projected = 1200s >> 10s
        self._report(p, 5, 100, 60_000.0)
        assert p.should_prune()

    def test_no_prune_within_projection(self):
        p = RuntimePruner(max_trial_seconds=120.0, warmup_steps=3)
        # 5 queries in 5 seconds; 100 queries projected = 100s < 120s
        self._report(p, 5, 100, 5_000.0)
        assert not p.should_prune()

    def test_reset_clears_state(self):
        p = RuntimePruner(max_trial_seconds=1.0, warmup_steps=1)
        self._report(p, 5, 100, 100_000.0)
        assert p.should_prune()
        p.reset()
        assert not p.should_prune()


# ── ParetoPruner ──────────────────────────────────────────────────────────────

class _FakeTrialState:
    COMPLETE = "COMPLETE"


class _FakeTrial:
    def __init__(self, values, state="COMPLETE"):
        self.values = values
        self.state = type("S", (), {"name": state})()


class _FakeStudy:
    def __init__(self, trials):
        self.trials = trials


class TestParetoPruner:
    def _make_study(self, pareto_points):
        """Create a fake study whose completed trials define the given Pareto front."""
        import optuna

        trials = []
        for obj1, obj2 in pareto_points:
            t = type("T", (), {
                "state": optuna.trial.TrialState.COMPLETE,
                "values": (obj1, obj2),
            })()
            trials.append(t)
        return type("S", (), {"trials": trials})()

    def _report_n(self, pruner, n, obj1_val=0.3, obj2_val=5.0):
        for s in range(1, n + 1):
            pruner.report(step=s, n_total=100, obj1=obj1_val, obj2=obj2_val, elapsed_ms=float(s * 1000))

    def test_no_prune_before_warmup_trials(self):
        # Only 5 completed trials but warmup_trials=30
        study = self._make_study([(0.9, 1.0), (0.8, 2.0)])
        p = ParetoPruner(study=study, warmup_trials=30, zscore=0.0)
        self._report_n(p, 10)
        assert not p.should_prune()

    def test_no_prune_before_min_steps(self):
        import optuna
        study = self._make_study([(0.95, 0.5)] * 31)
        p = ParetoPruner(study=study, warmup_trials=5, zscore=0.0, min_steps_before_prune=10)
        # Only report 3 steps
        self._report_n(p, 3, obj1_val=0.1, obj2_val=100.0)
        assert not p.should_prune()

    def test_prunes_clearly_dominated_trial(self):
        """A trial that clearly can't beat the Pareto front should be pruned."""
        import optuna

        # Pareto front: (0.95, 1.0) — high quality, very low cost
        study = self._make_study([(0.95, 1.0)] * 35)
        p = ParetoPruner(study=study, warmup_trials=30, zscore=0.0, min_steps_before_prune=5)
        # Trial running at NDCG 0.1 with cost 100 — clearly dominated
        self._report_n(p, 10, obj1_val=0.1, obj2_val=100.0)
        assert p.should_prune()

    def test_no_prune_for_promising_trial(self):
        import optuna

        # Pareto front: (0.7, 5.0)
        study = self._make_study([(0.7, 5.0)] * 35)
        p = ParetoPruner(study=study, warmup_trials=30, zscore=0.0, min_steps_before_prune=5)
        # Trial running at NDCG 0.8 with cost 4.0 — strictly dominates the front
        self._report_n(p, 10, obj1_val=0.8, obj2_val=4.0)
        assert not p.should_prune()

    def test_latch_stays_true_after_firing(self):
        import optuna

        study = self._make_study([(0.95, 1.0)] * 35)
        p = ParetoPruner(study=study, warmup_trials=30, zscore=0.0, min_steps_before_prune=5)
        self._report_n(p, 10, obj1_val=0.1, obj2_val=100.0)
        assert p.should_prune()
        # Even after many more good reports, it stays pruned
        self._report_n(p, 10, obj1_val=0.99, obj2_val=0.01)
        assert p.should_prune()

    def test_reset_clears_latch(self):
        import optuna

        study = self._make_study([(0.95, 1.0)] * 35)
        p = ParetoPruner(study=study, warmup_trials=30, zscore=0.0, min_steps_before_prune=5)
        self._report_n(p, 10, obj1_val=0.1, obj2_val=100.0)
        assert p.should_prune()
        p.reset()
        assert not p.should_prune()


# ── _pareto_front helper ──────────────────────────────────────────────────────

class TestParetoFrontHelper:
    def _trial(self, obj1, obj2):
        return type("T", (), {"values": (obj1, obj2)})()

    def test_single_point_is_pareto(self):
        trials = [self._trial(0.8, 5.0)]
        front = _pareto_front(trials)
        assert len(front) == 1

    def test_dominated_point_excluded(self):
        # (0.6, 10.0) is dominated by (0.8, 5.0)
        trials = [self._trial(0.8, 5.0), self._trial(0.6, 10.0)]
        front = _pareto_front(trials)
        assert (0.8, 5.0) in front
        assert (0.6, 10.0) not in front

    def test_trade_off_points_both_on_front(self):
        # Neither dominates the other
        trials = [self._trial(0.9, 10.0), self._trial(0.7, 2.0)]
        front = _pareto_front(trials)
        assert len(front) == 2
