# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:13:44 2023

@author: anonymous
"""

from oodesign import Node
import numpy as np


def get_node_by_id(node_id: str, nodes: list[Node]) -> Node:
    for node in nodes:
        if node.id == node_id:
            return node
    raise Exception(f"node_id:{node_id} not found in nodes:{nodes}")


def remove_zero_rows_cols(matrix: np.ndarray) -> np.ndarray:
    """remove zero rows and columns from a matrix"""
    return matrix[~np.all(matrix == 0, axis=1)][:, ~np.all(matrix == 0, axis=0)]


def remove_zero_val_dict(d: dict) -> dict:
    """remove zero values from a dictionary"""
    return {k: v for k, v in d.items() if v != 0}


def find_phases(d: dict) -> list[str]:
    """find the phases from a dictionary"""
    phases = d.keys()
    if len(next(iter(phases))) == 2:  # Check if it's delta connection
        return [phase[0] for phase in phases]
    else:
        return list(phases)


# function to get the sequence components of a 3-phase matrix
def get_seq_components(Va, Vb, Vc, pha, phb, phc) -> tuple[float, float, float]:
    """get the sequence components given phase components"""
    pha = np.deg2rad(pha)
    phb = np.deg2rad(phb)
    phc = np.deg2rad(phc)
    Va = Va * np.exp(1j * pha)
    Vb = Vb * np.exp(1j * phb)
    Vc = Vc * np.exp(1j * phc)

    a = np.exp(1j * 2 * np.pi / 3)
    V0 = (Va + Vb + Vc) / 3
    V1 = (Va + a * Vb + a**2 * Vc) / 3
    V2 = (Va + a**2 * Vb + a * Vc) / 3
    return V0, V1, V2
