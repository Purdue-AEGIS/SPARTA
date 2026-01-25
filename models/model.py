from const import NodeSide
import scipy.sparse as sps
from oodesign import Node, get_node_on_side
from enum import StrEnum
import numpy as np


class ValType(StrEnum):
    REAL = "real"
    IMAG = "imag"
    # TIME_DOMAIN = "time_domain"


class Model:
    def get_id(self):
        raise NotImplementedError

    def get_num_term(self):
        return self.obj.terminal.get_num_term()

    # returns: ["A", "B", "N"]
    def get_phases(self) -> list[str]:
        result = []
        for ph in self.obj.phases:
            if self.obj.phases[ph] == 1:
                result.append(ph)
        return result

    # returns (M_re, M_im)
    def get_M_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        M = self.get_M_powerflow_inner()
        M_re = M.tocsc()
        M_im = M_re[self.num_eqns_real :, self.num_vars_real :]
        assert M_im.shape[0] == self.num_eqns_complex
        assert M_im.shape[1] == self.num_vars_complex, f"M_im.shape[1]:{M_im.shape[1]}, self.num_vars_complex:{self.num_vars_complex}"
        return (M_re, M_im)

    def get_M_powerflow_inner(self) -> np.ndarray:
        raise NotImplementedError
    
    def get_local_idx_dynamic(self,
        var: str,
        ph: str | None = None,
        side: NodeSide | None = None,
    ) -> int:
        vars = list(self.var_offset_dynamic.keys())
        assert (
            var in self.var_offset_dynamic.keys()
        ), f"key {var} not found in self.var_offset_dynamic {vars}"

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset


    def get_local_idx(
        self,
        var: str,
        val_type: ValType,
        ph: str | None = None,
        side: NodeSide | None = None,
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        assert var not in ["v", "i"], "to be implemented if required"

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset
        

    def get_node_side(self, node_id) -> NodeSide:
        if self.num_term == 1:
            if self.obj.terminal.at_node.id == node_id:
                return NodeSide.AT
            else:
                raise ValueError("node_id:{node_id} is not adjacent to comp:{self}")

        elif self.num_term == 2:
            if self.obj.terminal.from_node.id == node_id:
                return NodeSide.FROM
            elif self.obj.terminal.to_node.id == node_id:
                return NodeSide.TO
            else:
                raise ValueError("node_id:{node_id} is not adjacent to comp:{self}")

        else:
            raise NotImplementedError

    def get_node_on_side(self, side: NodeSide) -> Node:
        return get_node_on_side(self.obj, side)
        # if side == NodeSide.FROM:
        #     return self.obj.terminal.from_node
        # elif side == NodeSide.TO:
        #     return self.obj.terminal.to_node
        # elif side == NodeSide.AT:
        #     return self.obj.at_node
        # else:
        #     raise ValueError("uknown side: {side}")

    def get_fy_powerflow(self, y: sps.coo_array) -> sps.coo_array:
        raise NotImplementedError

    def get_hy_powerflow(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array] | None:
        raise NotImplementedError
        return None

    def get_pd_hy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array] | None:
        raise NotImplementedError
        return None

    def get_pd_pd_hy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        mu_real: sps.coo_array,
        mu_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array] | None:
        raise NotImplementedError
        return None

    def get_pd_fy(self, fy: sps.coo_array, y: sps.coo_array) -> sps.coo_array:
        raise NotImplementedError

    def initial_guess(self, vals: dict) -> sps.coo_array:
        """This would be the complex value (not the split one)"""
        raise NotImplementedError
    
    def initial_guess_dynamic(self, y_comp: np.ndarray, w_nom: float) -> np.ndarray:
        print(f"self = {self}")
        raise NotImplementedError

    def get_basetype(self) -> str:
        raise NotImplementedError

    def get_objective(self, y_real: sps.coo_array, y_imag: sps.coo_array) -> float:
        return 0

    def get_pd_objective_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        raise NotImplementedError
        re = sps.lil_matrix((self.num_vars, 1), dtype=float)
        im = sps.lil_matrix((self.num_vars_complex, 1), dtype=float)

        return re, im

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        raise NotImplementedError
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_pd_objective_split(
        self,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        raise NotImplementedError
        pd_pd_obj_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        rr = pd_pd_obj_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_pd_obj_split[0 : self.num_vars, self.num_vars :]
        ir = pd_pd_obj_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_pd_obj_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_num_ineq_split(self) -> int:
        return 0

    def get_num_ineq_real(self) -> int:
        return 0

    def get_num_ineq_complex(self) -> int:
        return 0

    def get_K_dynamic(self) -> sps.coo_array:
        raise NotImplementedError

    def get_u_dynamic(self, t: float, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_algebraic_idx(self) -> list[int]:
        raise NotImplementedError

    def get_fy_dynamic(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError
