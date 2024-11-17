import logging
import os
from typing import NamedTuple

from ray import serve

from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.gain_models.gain_model import GainModel

logger = logging.getLogger('ray')


class SystemGainSimulatorParams(SerialisableBaseModel):
    run_params: ForwardModellingRunParams


class SystemGainSimulatorResponse(NamedTuple):
    gain_model: GainModel


@serve.deployment
class SystemGainSimulator:
    def __init__(self, params: ForwardModellingRunParams):
        self.params = params
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'system_gain_simulator')
        os.makedirs(self.params.plot_folder, exist_ok=True)

    async def __call__(self, time_idx: int, freq_idx: int) -> SystemGainSimulatorResponse:
        return SystemGainSimulatorResponse(

        )
