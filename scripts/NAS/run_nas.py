"""
import logging
import sys
import io
import time

class StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

def patch_logger_adapter(logger_adapter, logger, level=logging.INFO):
    original_info = logger_adapter.info
    def patched_info(message, *args, **kwargs):
        logger.log(level, message)
        original_info(message, *args, **kwargs)
    logger_adapter.info = patched_info

# Configure the logging module
logging.basicConfig(filename=f'outputs/logs/run_nas_output_{time.strftime("%Y%m%d-%H%M%S")}.log', level=logging.INFO)

# Redirect stdout to the logger
stdout_logger = logging.getLogger('STDOUT')
sys.stdout = StreamToLogger(stdout_logger, logging.INFO)

# Redirect stderr to the logger
stderr_logger = logging.getLogger('STDERR')
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
#"""

from pathlib import Path

import torchx

from torchx import specs
from torchx.components import utils
import time
import os
import tempfile
import subprocess

from ax.runners.torchx import TorchXRunner
from ax.core import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    FixedParameter,
)
from ax import OrderConstraint
from ax.core.search_space import HierarchicalSearchSpace
from ax.metrics.tensorboard import TensorboardMetric
from ax.core import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core import Experiment
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.service.utils.report_utils import exp_to_df

curDir = os.getcwd()


def trainer(trial_idx: int = -1, *args, **kwargs) -> specs.AppDef:
    update_log_path_with_trial_idx(kwargs, trial_idx)
    command_line_arguments = prepare_command_line_arguments_for_trial(kwargs)
    output = utils.python(
        *command_line_arguments,
        name="trainer",
        script=os.path.join(curDir, "model_script_for_nas.py"),
        image=torchx.version.TORCHX_IMAGE,
    )
    print_command_line(output)
    return output


def update_log_path_with_trial_idx(kwargs, trial_idx):
    if trial_idx >= 0:
        kwargs["log_path"] = (
            Path(kwargs["log_path"]).joinpath(str(trial_idx)).absolute().as_posix()
        )


def prepare_command_line_arguments_for_trial(kwargs):
    values = []
    for key, value in kwargs.items():
        if key in ["root"]:
            continue
        if key in ["sinusoidal_position", "rotary_position", "alibi_position", "learned_position"]:
            if value==True: values += [f"--{key}"]
            continue
        values += [f"--{key}", str(value)]
    return values


def print_command_line(output):
    outputStr = ["python"] + output.roles[0].args
    print(" ".join(outputStr))
    #subprocess.run(['python']+output.roles[0].args) # for debugging purposes.


# Make a temporary dir to log our results into
log_dir = tempfile.mkdtemp(
    prefix=f'log_{time.strftime("%Y%m%d-%H%M%S")}_',
    dir=os.path.join(curDir, "outputs", "logs"),
)

ax_runner = TorchXRunner(
    tracker_base="/tmp/",
    component=trainer,
    scheduler="local_cwd",
    component_const_params={"log_path": log_dir},
    cfg={},
)

parameters = [
    FixedParameter(
        name="root",
        value="true",
        parameter_type=ParameterType.STRING,
        dependents={
            "true": [
                "batch_size",
                "sinusoidal_position",
                "learned_position",
                "rotary_position",
                "alibi_position",
                "embedding_dimension",
                "feedforward_dimension",
                "number_of_heads",
                "number_of_layers",
                "number_of_warmup_steps",
                "weight_decay",
                "learning_rate",
                "dropout",
                "activation",
            ]
        },
    ),
    FixedParameter(
        name="batch_size",
        value=64,
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(
        name="sinusoidal_position",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="learned_position",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="rotary_position",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="alibi_position",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="embedding_dimension",
        values=[128, 256, 512, 1024],
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(
        name="feedforward_dimension",
        values=[128, 256, 512, 1024, 2048],
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(
        name="number_of_heads",
        values=[4, 8, 16, 32],
        parameter_type=ParameterType.INT,
    ),
    RangeParameter(
        name="number_of_layers",
        lower=4,
        upper=12,
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(
        name="activation",
        values=["leaky_relu_steep", "leaky_relu_slight", "sigmoid", "tanh", "selu", "relu"],
        parameter_type=ParameterType.STRING,
        is_ordered=False,
        sort_values=False,
    ),
    RangeParameter(
        name="dropout",
        lower=0.0,
        upper=0.9,
        parameter_type=ParameterType.FLOAT,
    ),
    FixedParameter(
        name="number_of_warmup_steps",
        value=10_000,
        parameter_type=ParameterType.INT,
    ),
    FixedParameter(
        name="weight_decay",
        value=1e-3,
        parameter_type=ParameterType.FLOAT,
    ),
    FixedParameter(
        name="learning_rate",
        value=1e-3,
        parameter_type=ParameterType.FLOAT,
    ),
]

search_space = HierarchicalSearchSpace(
    parameters=parameters,
    parameter_constraints=[
        #OrderConstraint(
        #    lower_parameter=num_layers_that_pass_directly_into_latent_space_parameter,
        #    upper_parameter=encoder_max_number_of_layers_parameter,
        #),
        #OrderConstraint(
        #    lower_parameter=num_layers_that_receive_direct_output_from_latent_space_parameter,
        #    upper_parameter=decoder_max_number_of_layers_parameter,
        #),
    ],
)


class MyTensorboardMetric(TensorboardMetric):
    @classmethod
    def get_ids_from_trials(cls, trials):
        return {
            trial.index: Path(log_dir).joinpath(str(trial.index)).as_posix()
            for trial in trials
        }

    @classmethod
    def is_available_while_running(cls):
        return False


val_loss = MyTensorboardMetric(
    name="val_loss",
    tag="val_loss",
    lower_is_better=True,
)


opt_config = OptimizationConfig(objective=Objective(metric=val_loss, minimize=True))

experiment = Experiment(
    name="torchx_cptac",
    search_space=search_space,
    optimization_config=opt_config,
    runner=ax_runner,
)

total_trials = 40


gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=total_trials,
)


scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(total_trials=total_trials, max_pending_trials=1),
)

scheduler.run_all_trials()

df = exp_to_df(experiment).sort_values("val_loss", ascending=True)
df.to_csv(
    os.path.join(
        curDir,
        "outputs",
        f'sample_dataset_nas_output_input_residual{time.strftime("%Y%m%d-%H%M%S")}.csv',
    ),
    index=False,
)
