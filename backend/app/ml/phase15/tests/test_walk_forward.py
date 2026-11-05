import unittest

import pandas as pd

from backend.app.ml.phase15.validation.walk_forward import (
    WalkForwardError,
    build_walk_forward_splits,
    combine_training_frames,
)


INSTRUMENTS = [
    f"NSE_EQ|TEST{i:04d}"
    for i in range(20)
]


def make_training_frame(rows=100):
    dates = pd.bdate_range("2025-01-01", periods=rows)

    result = []

    for i, session in enumerate(dates):
        target_position = i + 7

        target_date = (
            dates[target_position].strftime("%Y-%m-%d")
            if target_position < rows
            else None
        )

        eligible = (
            i >= 5
            and target_position < rows
        )

        result.append({
            "session_date": session.strftime("%Y-%m-%d"),
            "max_horizon_target_session": target_date,
            "training_eligible": eligible,
            "feature_ready": i >= 5,
            "targets_ready": target_position < rows,
        })

    return pd.DataFrame(result)


def make_panel(instrument_keys=INSTRUMENTS, rows=100):
    frames = {
        instrument_key: make_training_frame(rows)
        for instrument_key in instrument_keys
    }

    return combine_training_frames(frames)


def make_plan():
    panel = make_panel()

    return build_walk_forward_splits(
        panel,
        n_splits=4,
        initial_train_size=30,
        final_holdout_size=15,
        max_horizon=7,
    )


class TestWalkForwardSplitter(unittest.TestCase):

    def setUp(self):
        self.panel = make_panel()
        self.plan = build_walk_forward_splits(
            self.panel,
            n_splits=4,
            initial_train_size=30,
            final_holdout_size=15,
            max_horizon=7,
        )

    def test_combiner_adds_instrument_keys(self):
        self.assertIn("instrument_key", self.panel.columns)
        self.assertEqual(
            self.panel["instrument_key"].nunique(),
            20,
        )

    def test_creates_requested_number_of_folds(self):
        self.assertEqual(len(self.plan.folds), 4)

    def test_all_twenty_instruments_are_included(self):
        self.assertEqual(len(self.plan.instrument_keys), 20)

        self.assertEqual(
            set(self.plan.instrument_keys),
            set(INSTRUMENTS),
        )

    def test_every_validation_date_is_shared_by_all_instruments(self):
        for fold in self.plan.folds:
            validation = self.plan.panel.iloc[
                list(fold.validation_indices)
            ]

            date_sets = []

            for _, group in validation.groupby("instrument_key"):
                date_sets.append(set(group["session_date"]))

            self.assertEqual(len(date_sets), 20)

            for dates in date_sets[1:]:
                self.assertEqual(dates, date_sets[0])

            self.assertEqual(
                len(date_sets[0]),
                fold.validation_session_count,
            )

    def test_validation_windows_are_chronological(self):
        previous_end = None

        for fold in self.plan.folds:
            start = pd.Timestamp(fold.validation_start_date)
            end = pd.Timestamp(fold.validation_end_date)

            self.assertLessEqual(start, end)

            if previous_end is not None:
                self.assertGreater(start, previous_end)

            previous_end = end

    def test_training_windows_expand(self):
        train_session_counts = [
            fold.train_session_count
            for fold in self.plan.folds
        ]

        for earlier, later in zip(
            train_session_counts,
            train_session_counts[1:],
        ):
            self.assertGreater(later, earlier)

    def test_every_training_target_ends_before_validation_start(self):
        for fold in self.plan.folds:
            training = self.plan.panel.iloc[
                list(fold.train_indices)
            ]

            target_end = pd.to_datetime(
                training["max_horizon_target_session"]
            )

            validation_start = pd.Timestamp(
                fold.validation_start_date
            )

            self.assertTrue(
                (target_end < validation_start).all()
            )

    def test_exact_seven_session_purge_is_applied(self):
        for fold in self.plan.folds:
            self.assertEqual(
                fold.purged_session_count,
                7,
            )

            self.assertEqual(
                fold.purged_sample_count,
                7 * 20,
            )

    def test_final_training_is_before_holdout(self):
        training = self.plan.panel.iloc[
            list(self.plan.final_train_indices)
        ]

        holdout_start = pd.Timestamp(
            self.plan.final_holdout_start_date
        )

        training_dates = pd.to_datetime(training["session_date"])

        self.assertTrue((training_dates < holdout_start).all())

    def test_final_training_targets_end_before_holdout(self):
        training = self.plan.panel.iloc[
            list(self.plan.final_train_indices)
        ]

        target_dates = pd.to_datetime(
            training["max_horizon_target_session"]
        )

        holdout_start = pd.Timestamp(
            self.plan.final_holdout_start_date
        )

        self.assertTrue((target_dates < holdout_start).all())

    def test_final_training_has_seven_session_purge(self):
        self.assertEqual(
            self.plan.final_purged_session_count,
            7,
        )

        self.assertEqual(
            self.plan.final_purged_sample_count,
            7 * 20,
        )

    def test_final_training_and_holdout_are_disjoint(self):
        self.assertFalse(
            set(self.plan.final_train_indices).intersection(
                self.plan.final_holdout_indices
            )
        )

    def test_final_training_metadata_matches_indices(self):
        training = self.plan.panel.iloc[
            list(self.plan.final_train_indices)
        ]

        self.assertEqual(
            training["session_date"].nunique(),
            self.plan.final_train_session_count,
        )

        self.assertEqual(
            pd.Timestamp(training["session_date"].min()).strftime("%Y-%m-%d"),
            self.plan.final_train_start_date,
        )

        self.assertEqual(
            pd.Timestamp(training["session_date"].max()).strftime("%Y-%m-%d"),
            self.plan.final_train_end_date,
        )

    def test_final_holdout_has_requested_size(self):
        self.assertEqual(
            self.plan.final_holdout_session_count,
            15,
        )

        self.assertEqual(
            len(self.plan.final_holdout_indices),
            15 * 20,
        )

    def test_validation_targets_end_before_final_holdout(self):
        holdout_start = pd.Timestamp(
            self.plan.final_holdout_start_date
        )

        for fold in self.plan.folds:
            validation = self.plan.panel.iloc[
                list(fold.validation_indices)
            ]

            target_dates = pd.to_datetime(
                validation["max_horizon_target_session"]
            )

            self.assertTrue(target_dates.notna().all())
            self.assertTrue((target_dates < holdout_start).all())

    def test_seven_sessions_before_holdout_are_embargoed(self):
        calendar = pd.bdate_range(
            "2025-01-01",
            periods=100,
        )

        holdout_start = pd.Timestamp(
            self.plan.final_holdout_start_date
        )

        holdout_position = calendar.get_loc(holdout_start)

        embargo_dates = set(
            calendar[
                holdout_position - 7:holdout_position
            ].strftime("%Y-%m-%d")
        )

        validation_dates = set()

        for fold in self.plan.folds:
            validation = self.plan.panel.iloc[
                list(fold.validation_indices)
            ]

            validation_dates.update(
                validation["session_date"].tolist()
            )

        self.assertTrue(
            embargo_dates.isdisjoint(validation_dates)
        )

    def test_holdout_embargo_metadata(self):
        self.assertEqual(
            self.plan.holdout_embargo_session_count,
            7,
        )

    def test_final_holdout_is_after_all_validation_folds(self):
        holdout_start = pd.Timestamp(
            self.plan.final_holdout_start_date
        )

        for fold in self.plan.folds:
            validation_end = pd.Timestamp(
                fold.validation_end_date
            )

            self.assertLess(validation_end, holdout_start)

    def test_final_holdout_never_enters_a_fold(self):
        holdout_indices = set(
            self.plan.final_holdout_indices
        )

        for fold in self.plan.folds:
            self.assertFalse(
                holdout_indices.intersection(fold.train_indices)
            )

            self.assertFalse(
                holdout_indices.intersection(
                    fold.validation_indices
                )
            )

    def test_validation_and_training_indices_do_not_overlap(self):
        for fold in self.plan.folds:
            self.assertFalse(
                set(fold.train_indices).intersection(
                    fold.validation_indices
                )
            )

    def test_repeated_runs_are_deterministic(self):
        another_plan = make_plan()

        self.assertEqual(
            self.plan.final_holdout_indices,
            another_plan.final_holdout_indices,
        )

        self.assertEqual(
            len(self.plan.folds),
            len(another_plan.folds),
        )

        for first, second in zip(
            self.plan.folds,
            another_plan.folds,
        ):
            self.assertEqual(
                first.train_indices,
                second.train_indices,
            )

            self.assertEqual(
                first.validation_indices,
                second.validation_indices,
            )

    def test_calendar_mismatch_is_rejected(self):
        frames = {
            key: make_training_frame()
            for key in INSTRUMENTS
        }

        frames[INSTRUMENTS[0]] = (
            frames[INSTRUMENTS[0]].iloc[1:].reset_index(drop=True)
        )

        panel = combine_training_frames(frames)

        with self.assertRaises(WalkForwardError):
            build_walk_forward_splits(
                panel,
                n_splits=4,
                initial_train_size=30,
                final_holdout_size=15,
            )

    def test_eligibility_mismatch_across_instruments_is_rejected(self):
        frames = {
            key: make_training_frame()
            for key in INSTRUMENTS
        }

        frames[INSTRUMENTS[0]].loc[20, "training_eligible"] = False

        panel = combine_training_frames(frames)

        with self.assertRaises(WalkForwardError):
            build_walk_forward_splits(
                panel,
                n_splits=4,
                initial_train_size=30,
                final_holdout_size=15,
            )

    def test_duplicate_instrument_session_is_rejected(self):
        duplicate = self.panel.iloc[[0]].copy()

        bad_panel = pd.concat(
            [self.panel, duplicate],
            ignore_index=True,
        )

        with self.assertRaises(WalkForwardError):
            build_walk_forward_splits(
                bad_panel,
                n_splits=4,
                initial_train_size=30,
                final_holdout_size=15,
            )

    def test_wrong_target_horizon_is_rejected(self):
        bad_panel = self.panel.copy()

        eligible_index = bad_panel.index[
            (bad_panel["instrument_key"] == INSTRUMENTS[0])
            & bad_panel["training_eligible"]
        ][0]

        feature_date = pd.Timestamp(
            bad_panel.loc[eligible_index, "session_date"]
        )

        dates = pd.bdate_range("2025-01-01", periods=100)
        feature_position = dates.get_loc(feature_date)

        bad_panel.loc[
            eligible_index,
            "max_horizon_target_session",
        ] = dates[feature_position + 6].strftime("%Y-%m-%d")

        with self.assertRaises(WalkForwardError):
            build_walk_forward_splits(
                bad_panel,
                n_splits=4,
                initial_train_size=30,
                final_holdout_size=15,
            )

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(WalkForwardError):
            build_walk_forward_splits(
                self.panel,
                n_splits=0,
                initial_train_size=30,
                final_holdout_size=15,
            )

        with self.assertRaises(WalkForwardError):
            build_walk_forward_splits(
                self.panel,
                n_splits=4,
                initial_train_size=30,
                final_holdout_size=15,
                max_horizon=0,
            )

    def test_insufficient_history_is_rejected(self):
        short_panel = make_panel(rows=40)

        with self.assertRaises(WalkForwardError):
            build_walk_forward_splits(
                short_panel,
                n_splits=4,
                initial_train_size=30,
                final_holdout_size=15,
            )


if __name__ == "__main__":
    unittest.main()
