# -*- coding: utf-8 -*-

import numpy as np
import torch

import gntp

from gntp.lookup.base import BaseLookupIndex

from typing import List, Union, Optional

import logging

logger = logging.getLogger(__name__)


def reshape(tensor: Union[torch.Tensor, str],
            embedding_size: int):
    return torch.reshape(tensor, [-1, embedding_size]) if gntp.is_tensor(tensor) else tensor


def find_best_heads(index: BaseLookupIndex,
                    atoms: List[Union[torch.Tensor, str]],
                    goals: List[Union[torch.Tensor, str]],
                    goal_shape,
                    k: int = 10,
                    is_training: bool = False,
                    goal_indices: Optional[List[Union[np.ndarray, str]]] = None,
                    position: int = None):

    if isinstance(index, gntp.lookup.SymbolLookupIndex) and goal_indices is not None:
        assert position is not None
        atom_indices = index.query_sym(data_indices=goal_indices,
                                       k=k,
                                       is_training=is_training,
                                       position=position)

        actual_k = atom_indices.shape[-1]
        new_shp = (actual_k, *goal_shape[:-1])
        atom_indices = torch.as_tensor(atom_indices, dtype=torch.int64)
        atom_indices = torch.reshape(torch.transpose(atom_indices, 0, 1), new_shp)
    else:
        embedding_size = goal_shape[-1]
        new_goals = [reshape(ge, embedding_size) for ge in goals]

        ground_goals = [ge for fe, ge in zip(atoms, new_goals) if gntp.is_tensor(fe) and gntp.is_tensor(ge)]

        max_dim = max([gg.shape[0] for gg in ground_goals])

        ground_goals = [torch.tile(goal, [max_dim // goal.shape[0], 1]) for goal in ground_goals]

        # [G, 3 E], or e.g. [K, 2 3] if facts or goals contains a variable
        goals_2d = torch.cat(ground_goals, dim=1)

        # Facts in 'facts_2d' most relevant to the query in 'goals_2d'
        query_data = goals_2d if isinstance(index, gntp.lookup.FAISSLookupIndex) else goals_2d.detach().cpu().numpy()
        atom_indices = index.query(query_data,
                                   k=k,
                                   is_training=is_training)

        actual_k = atom_indices.shape[-1]
        new_shp = (actual_k, *goal_shape[:-1])
        atom_indices = torch.as_tensor(atom_indices, dtype=torch.int64)
        atom_indices = torch.reshape(torch.transpose(atom_indices, 0, 1), new_shp)
    return atom_indices
