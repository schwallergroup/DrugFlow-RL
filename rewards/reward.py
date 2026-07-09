import os
import numpy as np
from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig, OmegaConf

from rdkit import Chem

def linear_scaling(values, minv=0.0, maxv=6.0):
    ss = (values - minv) / (maxv - minv)
    ss[ss > 1.0] = 1.0
    ss[ss < 0.0] = 0.0
    return ss


def average_props(prop_dict):
    prop_num = len(prop_dict)
    prop_list = list(prop_dict.values())
    prop_sum = prop_list[0]
    for _prop in prop_list[1:]:
        prop_sum += _prop
    prop_mean = prop_sum / prop_num

    return prop_mean


def min_props(prop_dict):
    prop_list = list(prop_dict.values())
    prop_arr = np.array(prop_list)
    prop_min = prop_arr.min(axis=0)
    return prop_min


class Reward():
    def __init__(
        self,
        root_dir: str,
        prop_cfg: DictConfig,
        reward_threshold: float,
        reduce: str = 'mean',
        **kwargs,
    ) -> None:
        assert reduce in ['mean', 'min']
        self.root_dir = root_dir
        self.prop_cfg = prop_cfg
        self.threshold = reward_threshold
        self.cfg = OmegaConf.create(kwargs)
        self.reduce = reduce
        self.gate_metric = self.cfg.get('gate_metric', None)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # self.calculators = {}
        # for _cfg in prop_cfg:
        #     _root_dir = os.path.join(root_dir, _cfg.name)
        #     _calc = hydra.utils.instantiate(_cfg.calculator, root_dir=_root_dir)
        #     self.calculators[_cfg.name] = _calc

    def calc_props(
        self,
        samples: List[Chem.Mol],
        label: str = 'tmp'
    ):
        prop_dict, prop_list = {}, []

        # Extract pre-calculated metrics if available
        if isinstance(samples, dict) and 'metrics' in samples and self.gate_metric:
             metrics_list = samples['metrics']
             # Only extract the gate metric if it exists in the pre-calculated metrics
             if metrics_list and self.gate_metric in metrics_list[0]:
                 k = self.gate_metric
                 values = [m.get(k, 0.0) for m in metrics_list]
                 # Convert booleans to floats (1.0/0.0)
                 if values and isinstance(values[0], bool):
                     values = [1.0 if v else 0.0 for v in values]
                 prop_dict[k] = np.array(values, dtype=float)

        for _cfg in self.prop_cfg:
            _prop1 = _cfg.calculator.calc(samples, label)
            prop_list.append(_prop1)
            _prop2 = np.nan_to_num(_prop1, nan=0.0)
            prop_dict[_cfg.name] = _prop2.astype(float)

        prop_list = np.array(prop_list)
        none_ids = np.isnan(prop_list).any(axis=0)

        return prop_dict, none_ids

    def scoring(
        self,
        samples: List[Chem.Mol],
        label: str = 'tmp'
    ):

        prop_dict, failed_mask = self.calc_props(samples, label)

        scaled_prop_dict = {}
        for _cfg in self.prop_cfg:
            if _cfg.target == 'ascending':
                _sprop = linear_scaling(
                    values = prop_dict[_cfg.name],
                    minv = _cfg.minv,
                    maxv = _cfg.maxv,
                )
            elif _cfg.target == 'descending':
                _sprop = linear_scaling(
                    values = -prop_dict[_cfg.name],
                    minv = -_cfg.maxv,
                    maxv = -_cfg.minv,
                )
            elif isinstance(_cfg.target, float):
                diff = np.abs(prop_dict[_cfg.name] - _cfg.target)
                _sprop = linear_scaling(
                    values = -diff,
                    minv = -_cfg.maxv,
                    maxv = -_cfg.minv,
                )
            else:
                raise TypeError("prop cfg.target must be a float or descending or ascending")

            scaled_prop_dict[_cfg.name] = _sprop

        # Gating logic
        gate_score = None
        if self.gate_metric:
            if self.gate_metric not in prop_dict:
                raise ValueError(f"Gate metric '{self.gate_metric}' not found in calculated properties. Available: {list(prop_dict.keys())}")
            
            gate_values = prop_dict[self.gate_metric]
            
            # Mask out samples where gate metric is 0
            gate_mask = gate_values == 0.0
            failed_mask = failed_mask | gate_mask
            
            # Add raw gate values to scaled_prop_dict for logging purposes (optional, or just rely on prop_dict)
            # scaled_prop_dict[f'gate_{self.gate_metric}'] = gate_values
            
            gate_score = gate_values

        if self.reduce == 'mean':
            rewards = average_props(scaled_prop_dict)
        elif self.reduce == 'min':
            rewards = min_props(scaled_prop_dict)
        
        # Add gate score to rewards if it exists
        if gate_score is not None:
             rewards = rewards + gate_score

        rewards[failed_mask] = 0.0

        return rewards, prop_dict, failed_mask
