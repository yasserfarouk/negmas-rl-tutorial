from pathlib import Path
from typing import Annotated

from typer import Argument, Typer

from negmas_rl.cli.constants import BaselineMethods, Methods
from negmas_rl.cli.experiments.anac2024 import ANAC2024Experiment
from negmas_rl.cli.experiments.scml import SCMLExperiment
from negmas_rl.cli.experiments.scml_dynamic import SCMLDynamicExperiment
from negmas_rl.cli.experiments.scml_single import SCMLSingleScenarioExperiment
from negmas_rl.experiment import Experiment

app = Typer(name="negmasrl")

BASE_PATH = Path(Path.home() / "negmas" / "negmasrl" / "experiments")


EXPERIMENTS: dict[str, type[Experiment]] = dict(
    scml=SCMLExperiment,
    scmlsingle=SCMLSingleScenarioExperiment,
    scmldynamic=SCMLDynamicExperiment,
    anac2024=ANAC2024Experiment,
)
"""Maps command-line experiment name to the correct predefined experiment in run"""


@app.command()
def run(
    experiment: Annotated[
        str,
        Argument(
            help=f"A predefined experiment of the following: {list(EXPERIMENTS.keys())}",
            case_sensitive=False,
        ),
    ],
    condition: list[Methods] | None = None,
    ignore: list[Methods] | None = None,
    baseline_condition: list[BaselineMethods] | None = None,
    baseline_ignore: list[BaselineMethods] | None = None,
    ntrain: int | None = None,
    rand_negs: int = 0,
    test_negs: int = 50,
    on_train: bool = False,
    eval_freq: float = 0.005,
    test_eval_freq: float = 0.01,
    save_freq: float = 0.05,  # type: ignore
    plot: int = 0,
    always_starts: bool = None,  # type: ignore
    verbosity: int = 1,
    raise_exceptions: bool = False,
    raise_exceptions_in_depoyment: bool = False,
    same_train_test_partners: bool = False,
    save_outcome_optimality: bool = True,
    save_scenario_stats_per_neg: bool = True,
    save_outcome_optimality_per_neg: bool = True,
    path: Path = BASE_PATH,
    progress_bar: bool = None,  # type: ignore
):
    """Runs predefined experiments

    Args:
        experiment: The experiment to run. Must be one of the `PredefinedExperiment` members: SCML, SCMLSingleScenario
        condition: Must be either left alone or passed as one or more of the conditions in the experiment. Other conditions will not be run
        ignore: Conditions not to run
        baseline_condition: Must be either left alone or passed as one or more of the baseline conditions in the experiment. Other conditions will not be run
        baseline_ignore: Baseline conditions not to run
        ntrain: Number of training samples
        rand_negs: Number of random negotiations to run before training to get the baseline performance of each condition
        test_negs: Number of test negotiations to run at the end
        on_train: Report test on train results
        plot: save plots every given number of negotiations during final testing of deployment in negotiations
        always_starts: Always make the learner start if True
        verbosity: Verbosity level
        same_train_test_partners: If given and True, it forces the train and test partners to be the same
        raise_exceptions: Raise any exception during any run and stop execution
        path: Path to save logs to
    """
    exp = EXPERIMENTS[experiment]()  # type: ignore
    exp.path = path
    if same_train_test_partners:
        exp.testing.partners = exp.training.partners
    if progress_bar is not None:
        exp.progress_bar = progress_bar
    if ntrain is not None:
        for v in exp.conditions:
            v.n_training_steps = ntrain
    if condition is not None:
        for c in condition:
            exp.name += f"-{c.value}"
        exp.conditions = tuple(exp._condition_map[c.value] for c in condition)
    if baseline_condition is not None:
        for c in baseline_condition:
            exp.name += f"-{c.value}"
        exp.baseline_conditions = tuple(
            exp._baselines_map[c.value] for c in baseline_condition
        )
    if ignore is not None:
        for c in ignore:
            exp.name += f"-_{c.value}"
        names = set([_.name for _ in exp.conditions]).difference(set(ignore))
        exp.conditions = tuple(_ for _ in exp.conditions if _.name in names)
    if baseline_ignore is not None:
        for c in baseline_ignore:
            exp.name += f"-_{c.value}"
        names = set([_.name for _ in exp.baseline_conditions]).difference(
            set(baseline_ignore)
        )
        exp.baseline_conditions = tuple(
            _ for _ in exp.baseline_conditions if _.name in names
        )
    if always_starts is not None:
        exp.always_starts = always_starts
    if verbosity is not None:
        exp.verbosity = verbosity
    exp.run(
        n_rand_negs=rand_negs,
        n_test_negs=test_negs,
        on_train=on_train,
        plot_every=plot,
        eval_freq=eval_freq,
        test_eval_freq=test_eval_freq,
        save_freq=save_freq,
        raise_exceptions=raise_exceptions,
        raise_exceptions_in_depoyment=raise_exceptions_in_depoyment,
        save_outcome_optimality=save_outcome_optimality,
        save_scenario_stats_per_neg=save_scenario_stats_per_neg,
        save_outcome_optimality_per_neg=save_outcome_optimality_per_neg,
    )


if __name__ == "__main__":
    app()
