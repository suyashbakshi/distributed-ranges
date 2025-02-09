# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import resource
import subprocess
import uuid
from collections import namedtuple

from drbench.common import Device, Model, Runtime

AnalysisCase = namedtuple("AnalysisCase", "target size ranks")
AnalysisConfig = namedtuple(
    "AnalysisConfig",
    (
        "prefix benchmark_filter reps dry_run mhp_bench shp_bench "
        "weak_scaling different_devices ranks_per_node"
    ),
)


class Runner:
    def __init__(self, analysis_config: AnalysisConfig):
        self.analysis_config = analysis_config

    def __execute(self, command: str):
        logging.info(f"running command: {command}")
        usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        if not self.analysis_config.dry_run:
            subprocess.run(command, shell=True, check=True)
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        logging.info(
            f"command execution time, "
            f"user:{usage_end.ru_utime - usage_start.ru_utime:.2f}, "
            f"system:{usage_end.ru_stime - usage_start.ru_stime:.2f}, "
            f"command: {command}"
        )

    def __run_mhp_analysis(self, params, ranks, ranks_per_node, target):
        if target.runtime == Runtime.SYCL:
            params.append("--sycl")
            if self.analysis_config.different_devices:
                params.append("--different-devices")
            if target.device == Device.CPU:
                env = "ONEAPI_DEVICE_SELECTOR=opencl:cpu"
            else:
                env = (
                    "ONEAPI_DEVICE_SELECTOR="
                    "'level_zero:gpu;ext_oneapi_cuda:gpu'"
                )
        else:
            env = (
                "I_MPI_PIN_DOMAIN=core "
                "I_MPI_PIN_ORDER=compact "
                "I_MPI_PIN_CELL=unit"
            )

        mpirun_params = []
        mpirun_params.append(f"-n {str(ranks)}")
        if ranks_per_node:
            mpirun_params.append(f"-ppn {str(ranks_per_node)}")
        self.__execute(
            env
            + " mpirun "
            + " ".join(mpirun_params)
            + " "
            + self.analysis_config.mhp_bench
            + " "
            + " ".join(params)
        )

    def __run_shp_analysis(self, params, ranks, target):
        if target.device == Device.CPU:
            env = "ONEAPI_DEVICE_SELECTOR=opencl:cpu"
        else:
            env = (
                "ONEAPI_DEVICE_SELECTOR="
                "'level_zero:gpu;ext_oneapi_cuda:gpu'"
            )
        env += " KMP_AFFINITY=compact"
        params.append(f"--num-devices {ranks}")
        self.__execute(
            f'{env} {self.analysis_config.shp_bench} {" ".join(params)}'
        )

    def run_one_analysis(self, analysis_case: AnalysisCase):
        params = []
        target = analysis_case.target
        params.append(f"--vector-size {str(analysis_case.size)}")
        params.append(f"--reps {str(self.analysis_config.reps)}")
        params.append("--benchmark_out_format=json")
        params.append(f"--context device:{target.device.name}")
        params.append(f"--context model:{target.model.name}")
        params.append(f"--context runtime:{target.runtime.name}")
        params.append(f"--context target:{target}")
        params.append("--v=3")  # verbosity

        prefix = self.analysis_config.prefix
        params.append(f"--benchmark_out={prefix}-{uuid.uuid4().hex}.json")

        if self.analysis_config.benchmark_filter:
            params.append(
                f"--benchmark_filter={self.analysis_config.benchmark_filter}"
            )

        if self.analysis_config.weak_scaling:
            params.append("--weak-scaling")

        if analysis_case.target.model == Model.SHP:
            self.__run_shp_analysis(
                params, analysis_case.ranks, analysis_case.target
            )
        else:
            self.__run_mhp_analysis(
                params,
                analysis_case.ranks,
                self.analysis_config.ranks_per_node,
                analysis_case.target,
            )
