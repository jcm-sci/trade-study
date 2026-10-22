"""Qualify the host screen with actual compiler, solver and information artifacts.

This optional example requires reviewed op_system/op_engine/op_infer sources.
It uses only synthetic NumPy CPU inputs and performs no statistical fit.
"""

from __future__ import annotations

import json
from importlib.metadata import version

import numpy as np
from op_engine.core_solver import RunConfig
from op_engine.sensitivities import (
    SensitivityInitial,
    SensitivityOptions,
    SensitivitySystem,
    solve_sensitivities,
)
from op_infer.adapters.op import structural_model_reference
from op_infer.adapters.op_identify import (
    information_from_sensitivities,
    sensitivity_execution_reference,
)
from op_infer.identify import InformationContext
from op_infer.observe import IIDObservation
from op_infer.posterior import ArtifactReference
from op_system import compile_spec, symbolic_derivatives

from trade_study import FisherContext, FisherScreenPolicy, screen_fisher


def reference(label: str) -> ArtifactReference:
    """Identify a retained synthetic fixture declaration.

    Returns:
        Content identity of the supplied fixture label.
    """
    return ArtifactReference.from_bytes(label.encode(), kind="synthetic-fisher-v1")


def main() -> None:  # ruff: ignore[too-many-locals] -- complete qualification fixture
    """Check measured sensitivities and both nominal/targeted screen outcomes.

    Raises:
        ValueError: If actual artifact screens disagree with the reference.
    """
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "-rate*x", "z": "-rate*z"},
        "observables": {"measurement": {"expr": "gain*(x+2*z)"}},
    })
    system = SensitivitySystem(
        compiled.eval_fn,
        symbolic_derivatives(compiled),
        initial_parameters=("amplitude",),
    )
    initial = SensitivityInitial(
        np.array([2.0, 4.0]), np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    )
    parameters = {
        name: np.asarray(value)
        for name, value in zip(system.parameter_names, (0.7, 1.5, 2.0), strict=True)
    }
    times = np.array([0.0, 0.5, 1.0])
    solution = solve_sensitivities(
        system,
        parameters,
        initial,
        times,
        options=SensitivityOptions(
            store_history=True,
            solver=RunConfig(method="rk4", fixed_max_step=0.0625),
        ),
    )
    observation = IIDObservation(family="normal", scale=0.4, slope=2.0, offset=1.0)
    native_context = InformationContext(
        source_model=structural_model_reference(compiled),
        design=reference("times=0,.5,1; all observed"),
        parameter_point=reference("rate=.7,gain=1.5,amplitude=2"),
        initialization=reference("y0=amplitude*(1,2)"),
        observation=observation.reference,
        numerical=sensitivity_execution_reference(solution),
        qualification=reference("independent exponential measured derivative"),
        parameter_names=solution.parameter_names,
        parameter_units=("1/time", "measurement/state", "state"),
        observable_names=solution.observable_names,
    )
    information = information_from_sensitivities(
        solution, observation, context=native_context
    )
    # y_measured = 2*gain*5*amplitude*exp(-rate*t)+1; divide by sigma=.4.
    decay = np.exp(-0.7 * times)
    jacobian = np.stack((-15 * times * decay, 10 * decay, 7.5 * decay), axis=1)
    expected = 25 * jacobian.T @ jacobian
    np.testing.assert_allclose(information.matrix, expected, rtol=3e-7, atol=1e-6)
    ctx = FisherContext(
        parameter_names=solution.parameter_names,
        parameter_coordinates="physical rate,gain,amplitude; units in native context",
        parameter_scales=(1.0, 1.0, 1.0),
        observation_model=observation.reference.model_dump_json(),
        design=native_context.design.model_dump_json(),
        reference_regime=native_context.parameter_point.model_dump_json(),
        nuisance_treatment="complement fixed; gain/amplitude product direction",
        sensitivity_reference=native_context.source_model.model_dump_json(),
        numerical_reference=native_context.numerical.model_dump_json(),
        qualification_reference=native_context.qualification.model_dump_json(),
        information_reference=reference(
            "retained actual information from measured sensitivities"
        ).model_dump_json(),
    )
    policy = FisherScreenPolicy(1.0)
    nominal = screen_fisher(information.matrix, context=ctx, policy=policy)
    basis = np.array([[1.0, 0.0], [0.0, 0.8], [0.0, 0.6]])
    targeted = screen_fisher(
        information.matrix,
        context=ctx,
        policy=FisherScreenPolicy(
            1.0, directions=tuple(tuple(float(x) for x in row) for row in basis)
        ),
    )
    np.testing.assert_allclose(
        targeted.eigenvalues, np.linalg.eigvalsh(basis.T @ expected @ basis), rtol=3e-7
    )
    if nominal.status != "infeasible" or targeted.status != "feasible":
        msg = "Actual information artifacts failed the declared screen reference"
        raise ValueError(msg)
    print(
        json.dumps(
            {
                "packages": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "op-system",
                        "op-engine",
                        "op-infer",
                        "trade-study",
                    )
                },
                "nominal": nominal.to_metadata(),
                "targeted": targeted.to_metadata(),
                "maximum_information_error": float(
                    np.max(np.abs(information.matrix - expected))
                ),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
