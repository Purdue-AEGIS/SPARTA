## transformer_model

from oodesign import Transformer
import numpy as np
import scipy.sparse as sps
from const import NodeSide, StudyType
from models.model import Model
from models.model import Model, ValType
from models.component_models.equipment_model import EquipmentModel
import utils
from utils import phasor_to_timedomain
import itertools
import matplotlib.pyplot as plt

"""
1) class TransformerModel represent the transformer model for different transformer types
2) The different transformer types shall be subclasses of TransformerModel
3) Transformer object passed to the init function here comes from the object oriented representation of the network.
4) class TransformerModel shall have the following methods:
 a) get_M_powerflow
 b) get_u_powerflow
 c) get_fy_powerflow
Note: xmer is used as shortform  for transformer in the code
 TODO: M, K, fy, u for dynamic simulation
"""


class TransformerModel(EquipmentModel):
    def __init__(self, xmer_obj: Transformer):
        super().__init__(xmer_obj)

        self.R_act = None
        self.R = self.obj.resistance_mat

        self.L_act = None
        self.L = self.obj.inductance_mat

    def get_basetype(self):
        return "xfmr"


############################################################################################################
# Step down transformer models
# 3ph delta-star transformer model
class ThreePhaseDYStepDownTransformerModel(TransformerModel):
    def __init__(self, xmer_obj: Transformer):
        super().__init__(xmer_obj)

        self.num_term = self.obj.terminal.get_num_term()

        # book keeping of the equations
        # fmt: off
        self.num_eqns_real = 0
        self.num_eqns_complex = (
            3 * self.n_ph   # 1)KCL:-I + Ai = 0 )
            + 3 * self.n_ph # 2)KVL: ATV - v = 0
            + self.n_ph     # 3) v_s - v_p / nt = 0
            + self.n_ph     # 4) i_s - nt * i_p = 0
            + self.n_ph     # 5) Li - lamda = 0
            + self.n_ph     # 6) Ri + jw * lamda - v = 0
            + self.n_ph     # 7) I_mid = 0
        )
        # fmt: on
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book keeping of the variables
        # fmt: off
        # y = [w, V, I, v, i, lamda]

        self.num_vars_real = (1)
        self.num_vars_complex = (
            3 * self.n_ph       # V
            + 3 * self.n_ph     # I           
            + 3 * self.n_ph     # v
            + 3 * self.n_ph     # i
            + self.n_ph         # lamda
        )
        # fmt: on
        self.num_vars = self.num_vars_real + self.num_vars_complex

        # create dictionaries to store the offset of each variable in the y vector
        # fmt: off
        # 1. dictionary of var offset for real variables
        self.var_offset_real ={"w" :0}
        # 2. dictionary of var offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": 3 * self.n_ph,            
            "v": 3 * self.n_ph + 3 * self.n_ph,
            "i": 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
            "lamda": 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph
        }
        # 3. dictionary of var offset for all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + 3 * self.n_ph,
            "v": 1 + 3 * self.n_ph + 3 * self.n_ph,
            "i": 1 + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
            "lamda": 1 + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
        }
        # fmt: on
        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["lamda"] + self.n_ph

        #dynamic
        # this model has same eqns and variable for powerlflow and dynamic
        self.num_eqns_dynamic = self.num_eqns
        self.num_vars_dynamic = self.num_vars
        self.var_offset_dynamic = self.var_offset.copy()

        self.nt = xmer_obj.turns_ratio


####################################Powerflow functions#####################################       

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # y = [w, V, I, v, i , lamda]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        phases_without_n = [ph for ph in self.get_phases() if ph != "N"]

        from_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.pri_volt)))
        v_phasors_dict = {
            k: v for (k, v) in utils.get_vector_phasors(from_ph_dict).items()
        }
        v_phasors_from = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

        to_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.sec_volt)))
        v_phasors_dict = {
            k: v for (k, v) in utils.get_vector_phasors(to_ph_dict).items()
        }
        v_phasors_to = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

        # v_phasors = np.vstack((v_phasors_from, v_phasors_to))

        idx_Vfrom_start = self.var_offset["V"]
        idx_Vfrom_end = idx_Vfrom_start + self.n_ph
        idx_Vto_start = idx_Vfrom_end + self.n_ph
        idx_Vto_end = idx_Vto_start + self.n_ph

        y_0[idx_Vfrom_start:idx_Vfrom_end, 0] = v_phasors_from
        y_0[idx_Vto_start:idx_Vto_end, 0] = v_phasors_to

        return y_0

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        assert var not in [
            "v",
            "i",
        ], f"to be implemented for var={var}"

        # the NodeSide.TO is only valid for "V" and "I"
        if side == NodeSide.TO:
            assert var in ["V", "I"]

        side_offset = 0
        if side == NodeSide.TO:
            side_offset = 2 * self.n_ph

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_id(self):
        return self.obj.id

    def get_M_powerflow_inner(self, stage=None):
        """
        1) This function returns the M matrix for power flow simulation
        2) First we create the node-incidence A, identity, zero and other coefficient matrices required for each eqn
            and then stack them to form the M matrix
        3) ID_* are identity matrices
        4) Z_* are zero matrices
        """

        # create the node-incidence matrix
        # fmt: off
        A = np.array([
            # ip_ca ip_ab ip_bc is_a is_b is_c iz_a iz_b iz_c
            [ -1,   1,    0,    0,   0,   0,   0,   0,   0   ], # KCL at primary phase a
            [  0,  -1,    1,    0,   0,   0,   0,   0,   0   ], # KCL at primary phase b
            [  1,   0,   -1,    0,   0,   0,   0,   0,   0   ], # KCL at primary phase c
            [  0,   0,    0,    1,   0,   0,   1,   0,   0   ], # KCL at sec phase a
            [  0,   0,    0,    0,   1,   0,   0,   1,   0   ], # KCL at sec phase b
            [  0,   0,    0,    0,   0,   1,   0,   0,   1   ], # KCL at sec phase c
            [  0,   0,    0,    0,   0,   0,  -1,   0,   0   ], # KCL at leakage branch phase a
            [  0,   0,    0,    0,   0,   0,   0,  -1,   0   ], # KCL at leakage branch phase b
            [  0,   0,    0,    0,   0,   0,   0,   0,  -1   ], # KCL at leakage branch phase c

        ])
        # fmt: on

        # identity matrices for the equations
        Id_I = sps.identity(3 * self.n_ph, format="coo").tocsc()
        # I ->[I1a, I1b, I1c, I2a, I2b, I2c, I3a, I3b, I3c]

        Id_v = sps.identity(3 * self.n_ph, format="coo").tocsc()
        # v ->[vp_ca, vp_ab, vp_bc, vs_a, vs_b, vs_c, vz_a, vz_b, vz_c]

        # Identity matrix based on no. of phases
        Id_ph = sps.identity(self.n_ph, format="coo").tocsc()

        # create induction matrix for currents (Li - lamda = 0)
        # i ->[ip_ca, ip_ab, ip_bc, is_a, is_b, is_c, iz_a, iz_b, iz_c]
        Z_nph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        L_mat = sps.bmat([[Z_nph, Z_nph, self.L]], format="coo").tocsc()

        # create resistance matrix for currents (Ri + jw*lamda - v = 0)
        R_mat = sps.bmat([[Z_nph, Z_nph, self.R]], format="coo").tocsc()

        # create matrix for relation between primary and secondary side currents
        # i_s - nt * i_p (relation between primary and secondary currents)
        nt = self.nt * np.sqrt(3)
        # print(f">> nt: {nt}")
        # fmt: off
        N_i = np.array([
            # ip_ca    ip_ab  ip_bc  is_a  is_b  is_c  iz_a  iz_b  iz_c
            [  -nt,    0,     0,     1,    0,    0,    0,    0,    0   ], # is_a-nt*ip_ca = 0
            [   0,    -nt,    0,     0,    1,    0,    0,    0,    0   ], # is_b-nt*ip_ab = 0
            [   0,     0,    -nt,    0,    0,    1,    0,    0,    0   ], # is_c-nt*ip_bc = 0
        ])


        nt_r = 1 / nt  # reciprocal of turns ratio
        # create matrix for relation between primary and secondary side voltages
        # v_s - nt_r * v_p (relation between primary and secondary voltages)

        N_v = np.array([
            # vp_ca    vp_ab  vp_bc  vs_a  vs_b  vs_c  vz_a  vz_b  vz_c
            [  nt_r,    0,     0,     1,   0,   0,    0,    0,    0   ], # vs_a-nt_r * vp_ca = 0
            [   0,      nt_r,  0,     0,   1,   0,    0,    0,    0   ], # vs_b-nt_r * vp_ab = 0
            [   0,      0,     nt_r,  0,   0,   1,    0,    0,    0   ], # vs_c- nt_r * vp_bc = 0
        ])

        # fmt: on

        # create coeefficient matrix for branch voltage v for eqn 6
        # v ->[vp_ca, vp_ab, vp_bc, vs_a, vs_b, vs_c, vz_a, vz_b, vz_c]
        eqn6_v = sps.bmat([[Z_nph, Z_nph, Id_ph]], format="coo").tocsc()

        # zero vectors for w for each equation
        Z_w1 = sps.lil_matrix((A.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((A.T.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((N_v.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((N_i.shape[0], 1), dtype=float)
        Z_w5 = sps.lil_matrix((self.L.shape[0], 1), dtype=float)
        Z_w6 = sps.lil_matrix((self.R.shape[0], 1), dtype=float)

        # I_mid = 0 injection on internal node
        eqn7_i = sps.bmat([[Z_nph, -Id_ph, Z_nph]], dtype=float)
        Z_w7 = sps.lil_matrix((self.n_ph, 1), dtype=float)

        # fmt:off
        M = sps.bmat([
            #[w         V         I        v          i       lamda]
            [ Z_w1,    None,     -Id_I,     None,   A,      None],      # 1)KCL: -I + Ai = 0
            [ Z_w2,    A.T,       None,     -Id_v,  None,   None],      # 2)KVL: A'V - v = 0
            [ Z_w3,    None,      None,     N_v,    None,   None],      # 3) v_s - v_p / nt = 0
            [ Z_w4,    None,      None,     None,   N_i,    None],      # 4) i_s - nt * i_p = 0
            [ Z_w5,    None,      None,     None,   L_mat,  -Id_ph],    # 5) Li - lamda = 0
            [ Z_w6,    None,      None,     -eqn6_v,R_mat,  None],      # 6) Ri + jw * lamda - v = 0
            [ Z_w7,    None,      eqn7_i,   None,   None,   None],      # 7) I_mid = 0 injection on internal node
        ])
        # fmt:on

        return M

    def get_u_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        u = sps.lil_matrix((self.num_eqns, 1), dtype=complex)
        return u.real, u[self.num_eqns_real :].imag

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. In this transformer model, eqn 6 has non-linearity due to product of w and lamda
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im
        # create a zero matric for fy
        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # fy update for eqn 6
        # start stop index of w and lamda in y vector
        idx_w = self.var_offset["w"]
        w = y[idx_w, 0]
        idx_lamda_start = self.var_offset["lamda"]
        idx_lamda_end = idx_lamda_start + self.n_ph

        # start stop index of eqn 6
        # fmt: off
        idx_eq6_start = (
            3 * self.n_ph   # 1) KCL:-I + Ai = 0
            + 3 * self.n_ph # 2) KVL: ATv - V = 0
            + self.n_ph     # 3) v_s-v_p/nt = 0
            + self.n_ph     # 4) i_s-nt*i_p = 0
            + self.n_ph     # 5) Li-lamda = 0
        )
        # fmt: on
        idx_eq6_end = idx_eq6_start + self.n_ph

        fy[idx_eq6_start:idx_eq6_end] = 0 + 1j * (w * y[idx_lamda_start:idx_lamda_end])

        return fy.real, fy[self.num_eqns_real :].imag

    # return four quadrants: [rr, ri, ir, ii]
    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        # input("continue?")
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        # get the index of w
        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]

        # get the index of lamda
        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.var_offset_complex["lamda"]

        # eq6_re : real part of the pd of eqn 6 nonlinear term
        eq6_re_start_row = (
            3 * self.n_ph  # 1)KCL:-I + Ai = 0
            + 3 * self.n_ph  # 2)KVL: ATV - v = 0
            + self.n_ph  # 3)v_s-v_p/nt = 0
            + self.n_ph  # 4)i_s-nt*i_p = 0
            + self.n_ph  # 5)Li-lamda = 0
        )
        for offset in range(self.n_ph):
            row = eq6_re_start_row + offset

            # get the index of lamda in the y vector
            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = -lamda_im

            # lamda_re : derivative is zero

            # lamda_im
            pd_fy_split[row, self.num_vars + lamda_im_col_offset] = -w

        # eq6_im : imaginary part of the pd of eqn 6 nonlinear term
        eq6_im_start_row = (
            self.num_eqns
            + 3 * self.n_ph  # 1)KCL:-I + Ai = 0
            + 3 * self.n_ph  # 2)KVL: ATV - v = 0
            + self.n_ph  # 3)v_s-v_p/nt = 0
            + self.n_ph  # 4)i_s-nt*i_p = 0
            + self.n_ph
        )  # 5)Li-lamda = 0

        for offset in range(self.n_ph):
            row = eq6_im_start_row + offset

            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w_re
            pd_fy_split[row, w_col_offset] = lamda_re

            # lamda_re
            pd_fy_split[row, lamda_re_col_offset] = w
            # lamda_im: derivative is zero

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        # input("continue?")
        # plt.spy(pd_fy_split)
        # plt.show()

        return (rr, ri, ir, ii)
    
###########################Dynamic Simulation functions#########################

    def get_local_idx_dynamic(
            self, var : str, ph : str | None, side: NodeSide | None) -> int:
        
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None
        assert var not in[
            "v",
            "i",
        ], f"to be implemented for var = {var}"

        # the NodeSide.TO is only for V and I
        if side == NodeSide.TO:
            assert var in ["V", "I"]
        if side == NodeSide.FROM or side == NodeSide.AT or side is None:
            side_offset =0
        if side == NodeSide.TO:
            side_offset = 2 * self.n_ph # transformer has an internal node
        
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset
    
    def initial_guess_dynamic(self, y_comp: np.ndarray, wnom: float) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        # variables : [w, V, I, v, i, lamda]

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype = float)

        # initialize from powerflow directly
        vars_counts_real = [
            ("w", 1),
        ]
        for var, count in vars_counts_real:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count

            y0_dyn[idx_var_dyn_start : idx_var_dyn_end] = y_comp[idx_var_pf_start : idx_var_pf_end]

        vars_counts_complex = [
            ("V", 3 * self.n_ph),
            ("I", 3 * self.n_ph),
            ("v", 3 * self.n_ph),
            ("i", 3 * self.n_ph),
            ("lamda", self.n_ph)
        ]

        # to be converted from phasor to time domain
        for var, count in vars_counts_complex:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_end + count

            y0_dyn[idx_var_dyn_start : idx_var_dyn_end] = np.sqrt(2) * phasor_to_timedomain(
                y_comp[idx_var_pf_start : idx_var_pf_end])
            
        return y0_dyn
    
    def initial_yp_dynamic(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn
    
    def initial_guess_dynamic_zero(self, y_comp: list, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        # y0_dyn[0] = y_comp[0].real

        # # we only init voltage:
        # pf_idx_V_start, pf_idx_V_end = get_start_end_idx(self.var_offset, "V", self.n_ph)
        # V = y_comp[pf_idx_V_start:pf_idx_V_end]
        # dyn_idx_V_start, dyn_idx_V_end = get_start_end_idx(self.var_offset_dynamic, "V", self.n_ph)
        # y0_dyn[dyn_idx_V_start:dyn_idx_V_end] = [
        #     np.sqrt(2) * phasor_to_timedomain(val) for val in V
        # ]

        return y0_dyn
    
    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        return yp0_dyn
    
    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        # for this model the dynamic M is exactly same as that of powerflow
        M = self.get_M_powerflow_inner(stage)

        return M
    
    def get_K_dynamic(self, stage=None) -> sps.coo_array:
        # fmt:off
        Z_A = sps.lil_matrix((3 * self.n_ph, 3 * self.n_ph), dtype=float)
        Z_Nv = sps.lil_matrix((self.n_ph, 3 * self.n_ph), dtype=float)
        Z_Ni = sps.lil_matrix((self.n_ph, 3 * self.n_ph), dtype=float)


        Z_w1 = sps.lil_matrix((3 * self.n_ph, 1), dtype=float)
        Z_w2 = sps.lil_matrix((3 * self.n_ph, 1), dtype=float)
        Z_w3 = sps.lil_matrix(( self.n_ph, 1), dtype=float)

        # Identity matrix based on no. of phases
        Id_ph = sps.identity(self.n_ph, format="coo")
       

        K = sps.bmat([
                    # w,      V,         I,         v,          i,       lamda
                    [Z_w1,   None,      Z_A,       None,      Z_A,       None], # 1) KCL:A*i-I=0
                    [Z_w2,   Z_A,       None,      None,      None,      None], # 2) KVL:A'*V-v=0
                    [Z_w3,   None,      None,      Z_Nv,      None,      None], # 3) v_s - v_p / nt = 0
                    [Z_w3,   None,      None,      None,      None,      None], # 4) i_s + nt * i_p = 0
                    [Z_w3,   None,      None,      None,      None,      None], # 5) Li-lamda=0                    
                    [Z_w3,   None,      None,      None,      None,      Id_ph],# 6) -v+Ri+d(lamda)/dt =0
                    [Z_w3,   None,      None,      None,      None,      None], # 7) I_mid = 0 injection on internal node

                ]
            )
        # fmt: on

        return K
    
    def get_fy_dynamic(self, t : float, y: np.ndarray, stage=None) -> np.ndarray:
          """
        1. This function returns the non-linear terms of every equation for dynamic simulation
        2. This dynamic model has no non-linearity.

        """
          fy = np.zeros(self.num_eqns_dynamic, dtype = float)

          return fy
    
    def get_u_dynamic(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        1. This function returns the u vector for dynamic simulation
        2. The dynamic model has zero u vector.
        """
         
        u = np.zeros(self.num_eqns_dynamic, dtype=float)
        return u
    







        #init from powerflow
    



############################################################################################################
# 3ph Y-Y transformer model


class ThreePhaseYYStepDownTransformerModel(TransformerModel):
    def __init__(self, xmer_obj: Transformer):
        super().__init__(xmer_obj)

        # declare num of terminals
        self.num_term = self.obj.terminal.get_num_term()

        # book keeping for equations
        # fmt: off
        self.num_eqns_real = 0
        self.num_eqns_complex = (
            3 * self.n_ph   # 1)KCL:-I + Ai = 0 )
            + 3 * self.n_ph # 2)KVL: ATV - v = 0
            + self.n_ph     # 3) v_s - v_p / nt = 0
            + self.n_ph     # 4) i_s - nt * i_p = 0
            + self.n_ph     # 5) Li - lamda = 0
            + self.n_ph     # 6) Ri + jw * lamda - v = 0
            + self.n_ph     # 7) I_mid = 0 inection on internal node
        )
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # fmt: on

        # book keeping for variables
        # fmt: off
        # y = [w, V, I, v, i, lamda]
        # Create dictionaries to store the offset of each variable in the y vector
        # 1. dictionary of var offset for real variables
        self.num_vars_real = (1)
        
        self.num_vars_complex = (
            3*self.n_ph       # V
            + 3*self.n_ph     # I            
            + 3*self.n_ph     # v
            + 3*self.n_ph     # i
            + self.n_ph       # lamda
        )        
        # 3. dictionary of var offset for all variables         
        self.num_vars = self.num_vars_real + self.num_vars_complex

        # fmt: on

        # Create dictionaries to store the offset of each variable in the y vector
        # 1. dictionary of var offset for real variables
        self.var_offset_real = {"w": 0}
        # 2. dictionary of var offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": 3 * self.n_ph,
            "v": 3 * self.n_ph + 3 * self.n_ph,
            "i": 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
            "lamda": 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
        }

        # fmt: off
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + 3 * self.n_ph,            
            "v": 1 + 3 * self.n_ph + 3 * self.n_ph, 
            "i": 1 + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
            "lamda": 1 + 3 * self.n_ph + 3 * self.n_ph  + 3 * self.n_ph + 3 * self.n_ph
        }
        # fmt: on
        
        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["lamda"] + self.n_ph

        # dynamic
        # this model has same eqns and variables for powerflow and dynamic
        self.num_eqns_dynamic = self.num_eqns 

        self.num_vars_dynamic = self.num_vars
        self.var_offset_dynamic = self.var_offset.copy()
        self.nt = xmer_obj.turns_ratio

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # y = [w, V, I, v, i, lamda]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        phases_without_n = [ph for ph in self.get_phases() if ph != "N"]

        from_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.pri_volt)))
        v_phasors_dict = {
            k: v for (k, v) in utils.get_vector_phasors(from_ph_dict).items()
        }
        v_phasors_from = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

        to_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.sec_volt)))
        v_phasors_dict = {
            k: v for (k, v) in utils.get_vector_phasors(to_ph_dict).items()
        }
        v_phasors_to = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

        # v_phasors = np.vstack((v_phasors_from, v_phasors_to))

        idx_Vfrom_start = self.var_offset["V"]
        idx_Vfrom_end = idx_Vfrom_start + self.n_ph
        idx_Vto_start = idx_Vfrom_end + self.n_ph
        idx_Vto_end = idx_Vto_start + self.n_ph

        y_0[idx_Vfrom_start:idx_Vfrom_end, 0] = v_phasors_from
        y_0[idx_Vto_start:idx_Vto_end, 0] = v_phasors_to

        return y_0

    # This function gives the local index of the variable in the local y vector.
    # In general it is assumed that the internal branch currents and voltage shall not be required
    # hence not implemented presently and therefore and assertion is raised for the same.
    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        assert var not in [
            "v",
            "i",
        ], f"to be implemented for var={var}"

        # the NodeSide.TO is only valid for "V" and "I"
        if side == NodeSide.TO:
            assert var in ["V", "I"]
        if side == NodeSide.FROM or side == NodeSide.AT or side is None:
            side_offset = 0
        if side == NodeSide.TO:
            side_offset = 2 * self.n_ph  # transformer also has an internal node

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_id(self):
        return self.obj.id

    def get_M_powerflow_inner(self, stage=None):
        """
        1) This function returns the M matrix for power flow simulation
        2) First we create the node-incidence A, identity, zero and other coefficient matrices required for each eqn
            and then stack them to form the M matrix
        3) ID_* are identity matrices
        4) Z_* are zero matrices
        """

        # create the node-incidence matrix for the transformer node and branches
        # fmt: off
        A = np.array([
            #[ip_a, ip_b, ip_c, is_a, is_b, is_c, iz_a, iz_b, iz_c]
             [1,    0,    0,    0,    0,    0,    0,    0,    0   ], # KCL at primary phase a
             [0,    1,    0,    0,    0,    0,    0,    0,    0   ], # KCL at primary phase b
             [0,    0,    1,    0,    0,    0,    0,    0,    0   ], # KCL at primary phase c
             [0,    0,    0,    1,    0,    0,    1,    0,    0   ], # KCL at sec phase a
             [0,    0,    0,    0,    1,    0,    0,    1,    0   ], # KCL at sec phase b
             [0,    0,    0,    0,    0,    1,    0,    0,    1   ], # KCL at sec phase c
             [0,    0,    0,    0,    0,    0,   -1,    0,    0   ], # KCL at leakage branch phase a
             [0,    0,    0,    0,    0,    0,    0,   -1,    0   ], # KCL at leakage branch phase b
             [0,    0,    0,    0,    0,    0,    0,    0,   -1   ], # KCL at leakage branch phase c

        ])
        # fmt: on

        # identity matrices for the equations
        Id_I = sps.identity(3 * self.n_ph, format="coo")
        # I ->[I1a, I1b, I1c, I2a, I2b, I2c, I3a, I3b, I3c]

        # identity matrices for the branch voltages
        Id_v = sps.identity(3 * self.n_ph, format="coo")
        # v ->[vp_a, vp_b, vp_c, vs_a, vs_b, vs_c, vz_a, vz_b, vz_c]

        # Identity matrix based on no. of phases
        Id_ph = sps.identity(self.n_ph, format="coo")

        # create induction matrix for currents (Li - lamda = 0)
        # i ->[ip_a, ip_b, ip_c, is_a, is_b, is_c, iz_a, iz_b, iz_c]
        Z_nph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        L_mat = sps.bmat([[Z_nph, Z_nph, self.L]], format="coo")
        print(f">> L_mat: {L_mat}")

        # create resistance matrix for currents (Ri + jw*lamda - v = 0)
        R_mat = sps.bmat([[Z_nph, Z_nph, self.R]], format="coo")
        print(f">> R_mat: {R_mat}")

        # create matrix for relation between primary and secondary side currents
        # i_s +  nt * i_p = 0 (relation between primary and secondary currents)

        nt = self.nt
        nt_r = 1 / nt  # reciprocal of turns ratio
        # fmt: off
        N_i = np.array([
            # ip_a    ip_b  ip_c  is_a  is_b  is_c  iz_a  iz_b  iz_c
            [  nt,    0,    0,    1,    0,    0,    0,    0,    0   ], # is_a+nt*ip_a = 0
            [  0,     nt,   0,    0,    1,    0,    0,    0,    0   ], # is_b+nt*ip_b = 0
            [  0,     0,    nt,   0,    0,    1,    0,    0,    0   ], # is_c+nt*ip_c = 0
        ])

        # create matrix for relation between primary and secondary side voltages
        # v_s - nt_r * v_p (relation between primary and secondary voltages)
        N_v = np.array([
            # vp_a    vp_b  vp_c  vs_a  vs_b  vs_c  vz_a  vz_b  vz_c
            [  nt_r,    0,    0,   -1,    0,    0,    0,    0,    0   ], # vs_a- nt_r* vp_a = 0
            [  0,     nt_r,   0,    0,   -1,    0,    0,    0,    0   ], # vs_b-nt_r * vp_b = 0
            [  0,     0,    nt_r,   0,    0,   -1,    0,    0,    0   ], # vs_c-nt_r * vp_c = 0
        ])
        # fmt: on

        # zero vectors for w for each equation
        Z_w1 = sps.lil_matrix((A.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((A.T.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((N_v.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((N_i.shape[0], 1), dtype=float)
        Z_w5 = sps.lil_matrix((self.L.shape[0], 1), dtype=float)
        Z_w6 = sps.lil_matrix((self.R.shape[0], 1), dtype=float)
        Z_w7 = sps.lil_matrix((self.n_ph, 1), dtype=float)

        # I_mid = 0 injection on internal node
        eqn7_i = sps.bmat([[Z_nph, -Id_ph, Z_nph]], dtype=float)
        eqn6_v = sps.bmat([[Z_nph, Z_nph, Id_ph]], format="coo")

        # fmt:off
        M = sps.bmat([
            #[w        V         I         v           i       lamda]
            [ Z_w1,   None,     -Id_I,     None,       A,      None],      # 1)KCL: -I + Ai = 0
            [ Z_w2,   A.T,      None,      -Id_v,      None,   None],      # 2)KVL: ATV - v = 0
            [ Z_w3,   None,     None,      N_v,        None,   None],      # 3) v_s - v_p / nt = 0
            [ Z_w4,   None,     None,      None,       N_i,    None],      # 4) i_s + nt * i_p = 0
            [ Z_w5,   None,     None,      None,       L_mat,  -Id_ph],    # 5) Li - lamda = 0
            [ Z_w6,   None,     None,      -eqn6_v,    R_mat,  None],      # 6) Ri + jw * lamda - v = 0
            [ Z_w7,   None,     eqn7_i,    None,       None,   None],      # 7) I_mid = 0 injection on internal node
        ])
        # fmt:on

        return M

    def get_u_powerflow(self):
        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=float)
        return u[: self.num_eqns], u[self.num_eqns :]

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. In this transformer model, eqn6 has non-linearity due to product of w and lamda
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # fy update for eqn 6

        # index of w in  y vector
        idx_w = self.var_offset["w"]
        w = y[idx_w, 0]

        # start stop index of lamda in y vector
        idx_lamda_start = self.var_offset["lamda"]
        idx_lamda_end = idx_lamda_start + self.n_ph

        print(
            f">> get_fy_powerflow(): idx_lamda_start: {idx_lamda_start}, idx_lamda_end: {idx_lamda_end}"
        )

        # start stop index of eqn 6
        # fmt: off
        idx_eq6_start = (
            3 * self.n_ph   # 1) KCL:-I + Ai = 0
            + 3 * self.n_ph # 2) KVL: ATv - V = 0
            + self.n_ph     # 3) v_s-v_p/nt = 0
            + self.n_ph     # 4) i_s-nt*i_p = 0
            + self.n_ph     # 5) Li-lamda = 0
        )
        # fmt: on
        idx_eq6_end = idx_eq6_start + self.n_ph

        print(
            f">> get_fy_powerflow(): idx_eq6_start: {idx_eq6_start}, idx_eq6_end: {idx_eq6_end}"
        )

        fy[idx_eq6_start:idx_eq6_end] = 0 + 1j * (w * y[idx_lamda_start:idx_lamda_end])

        return fy.real, fy[self.num_eqns_real :].imag

    # return four quadrants: [rr, ri, ir, ii]
    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]

        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.var_offset_complex["lamda"]

        # fmt: off
        # eq6_re
        eq6_re_start_row = (
            3 * self.n_ph    # 1)KCL:-I + Ai = 0
            + 3 * self.n_ph  # 2)KVL: ATV - v = 0
            + self.n_ph      # 3)v_s-v_p/nt = 0
            + self.n_ph      # 4)i_s-nt*i_p = 0
            + self.n_ph      # 5)Li-lamda = 0
        )
        # fmt: on

        # update the pd_fy_split for eqn 6 real part
        for offset in range(self.n_ph):
            row = eq6_re_start_row + offset

            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = -lamda_im

            # lamda_re : derivative is zero

            # lamda_im
            pd_fy_split[row, self.num_vars + lamda_im_col_offset] = -w

        # update the pd_fy_split for eqn 6 imaginary part
        eq6_im_start_row = self.num_eqns + eq6_re_start_row
        for offset in range(self.n_ph):
            row = eq6_im_start_row + offset

            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = lamda_re

            # lamda_re
            pd_fy_split[row, lamda_re_col_offset] = w
            # lamda_im : derivative is zero

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        """
        1. This function is to be called from Optimal Powerflow
        2. This function finds the Jacobian for the new non-linear vector formed by the multiplication of the
        Lagrange multiplier and the transposed Jacobian of the no-linear vector of the powerflow eqns.
        3. g(y) = Jf^T *lagrangian.
        4. This function returns the partial derivative of g(y) with respect to y for this component to be stacked appropriately
        in the overall Jacobian"""

        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.lil_matrix(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        )

        # update Jacobian for eqn 6

        idx_eq6 = (
            3 * self.n_ph  # 1) KCL:-I + Ai = 0
            + 3 * self.n_ph  # 2) KVL: ATv - V = 0
            + self.n_ph  # 3) v_s-v_p/nt = 0
            + self.n_ph  # 4) i_s-nt*i_p = 0
            + self.n_ph  # 5) Li-lamda = 0)
        )

        w_row_offset = self.var_offset["w"]
        w_col_offset = w_row_offset
        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.var_offset_complex["lamda"]

        # eq6_re: (-w * lamda_im)
        for offset in range(self.n_ph):
            lagm = lagm_real[idx_eq6 + offset, 0]
            col = self.num_vars + lamda_im_start_offset + offset
            pd_gy_split[w_row_offset, col] = -1 * lagm

        lamda_im_row_offset_start = self.num_vars + self.var_offset_complex["lamda"]

        for offset in range(self.n_ph):
            lagm = lagm_real[idx_eq6 + offset, 0]
            row = lamda_im_row_offset_start + offset
            pd_gy_split[row, w_col_offset] = -1 * lagm

        # eq6_im: (w * lamda_re)
        for offset in range(self.n_ph):
            lagm = lagm_imag[idx_eq6 + offset, 0]
            col = lamda_re_start_offset + offset
            pd_gy_split[w_row_offset, col] = 1 * lagm

        for offset in range(self.n_ph):
            lagm = lagm_imag[idx_eq6 + offset, 0]
            row = lamda_re_start_offset + offset
            pd_gy_split[row, w_col_offset] = 1 * lagm

        rr = pd_gy_split[: self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)
    
##########################dynamic simulation block##############################
    def get_local_idx_dynamic(
            self, var : str, ph: str | None, side: NodeSide | None) -> int:
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None
        assert var not in[
            "v",
            "i",
        ], f"to be implemented for var={var}"

        # the NodeSide.TO is only valid for "V" and "I"
        if side == NodeSide.TO:
            assert var in ["V", "I"]
        if side == NodeSide.FROM or side == NodeSide.AT or side is None:
            side_offset = 0
        if side == NodeSide.TO:
            side_offset = 2 * self.n_ph # transformer also has an internal node

        phase_offset  = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset
    

    def initial_guess_dynamic(self, y_comp: np.ndarray, w_nom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        # variables : [w, V, I, v, i, lamda]
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype = float)

        # initialization from powerflow directly
        vars_counts_real = [
            ("w", 1),
        ]
        for var, count in vars_counts_real:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count

            y0_dyn[idx_var_dyn_start : idx_var_dyn_end] = (y_comp[idx_var_pf_start : idx_var_pf_end])

        vars_counts_complex = [
            ("V", 3 * self.n_ph),
            ("I", 3 * self.n_ph),
            ("v", 3 * self.n_ph),
            ("i", 3 * self.n_ph),
            ("lamda", self.n_ph)
        ]

        for var, count in vars_counts_complex:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count

            y0_dyn[idx_var_dyn_start : idx_var_dyn_end] = np.sqrt(2) * phasor_to_timedomain(y_comp[idx_var_pf_start : idx_var_pf_end])

        return y0_dyn

    def initial_yp_dynamic(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn
    
    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        # for this model the dynamic M is exactly same as that of powerflow
        M = self.get_M_powerflow_inner(stage)

        return M
    

    def get_K_dynamic(self, stage=None) -> sps.coo_array:
        # fmt: off
        Z_A = sps.lil_matrix((3 * self.n_ph, 3 * self.n_ph), dtype=float)
        Z_Nv = sps.lil_matrix((self.n_ph, 3 * self.n_ph), dtype=float)
        Z_Ni = sps.lil_matrix((self.n_ph, 3 * self.n_ph), dtype=float)


        Z_w1 = sps.lil_matrix((3 * self.n_ph, 1), dtype=float)
        Z_w2 = sps.lil_matrix((3 * self.n_ph, 1), dtype=float)
        Z_w3 = sps.lil_matrix(( self.n_ph, 1), dtype=float)

        # Identity matrix based on no. of phases
        Id_ph = sps.identity(self.n_ph, format="coo")
       

        K = sps.bmat([
                    # w,      V,         I,         v,          i,       lamda
                    [Z_w1,   None,      Z_A,       None,      Z_A,       None], # 1) KCL:A*i-I=0
                    [Z_w2,   Z_A,       None,      None,      None,      None], # 2) KVL:A'*V-v=0
                    [Z_w3,   None,      None,      Z_Nv,      None,      None], # 3) v_s - v_p / nt = 0
                    [Z_w3,   None,      None,      None,      None,      None], # 4) i_s + nt * i_p = 0
                    [Z_w3,   None,      None,      None,      None,      None], # 5) Li-lamda=0                    
                    [Z_w3,   None,      None,      None,      None,      Id_ph],# 6) -v+Ri+d(lamda)/dt =0
                    [Z_w3,   None,      None,      None,      None,      None], # 7) I_mid = 0 injection on internal node

                ]
            )
        # fmt: on

        return K
    
    def get_fy_dynamic(self, t: float, y: np.ndarray, yp: np.ndarray, stage=None) -> np.ndarray:
        """
        1. This function returns the non-linear terms of every equation for dynamic simulation
        2. This dynamic model has no non-linearity.

        """
        fy = np.zeros(self.num_eqns_dynamic, dtype = float)

        return fy
    
    def get_u_dynamic(self, t: float, y) -> np.ndarray:
        """
        1. This function returns the u vector for dynamic simulation
        2. The dynamic model has zero u vector.
        """
        u = np.zeros(self.num_eqns_dynamic, dtype=float)
        return u
    
    def initial_guess_dynamic_zero(self, y_comp: list, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        # y0_dyn[0] = y_comp[0].real

        # # we only init voltage:
        # pf_idx_V_start, pf_idx_V_end = get_start_end_idx(self.var_offset, "V", self.n_ph)
        # V = y_comp[pf_idx_V_start:pf_idx_V_end]
        # dyn_idx_V_start, dyn_idx_V_end = get_start_end_idx(self.var_offset_dynamic, "V", self.n_ph)
        # y0_dyn[dyn_idx_V_start:dyn_idx_V_end] = [
        #     np.sqrt(2) * phasor_to_timedomain(val) for val in V
        # ]

        return y0_dyn
    
    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        return yp0_dyn
    


# ############################################################################################################
# # 3ph delta-delta transformer model
# class ThreePhaseDDStepDownTransformerModel2(TransformerModel):
#     def __init__(self, xmer_obj: Transformer):
#         super().__init__(xmer_obj)

#         # declare num of terminals
#         self.num_term = self.obj.terminal.get_num_term()

#         # book keeping of the equations
#         # fmt: off
#         self.num_eqns_real = 0
#         self.num_eqns_complex = (
#             2 * self.n_ph + 2   # 1)KCL
#             + 3 * self.n_ph # 2)KVL
#             + self.n_ph    # 3) v_s - v_p / nt =0
#             + self.n_ph    # 4) i_s - nt * i_p = 0
#             + self.n_ph    # 5) L*i_z - lamda =0
#             + self.n_ph    # 6) R*i_z + jw*lamda - v_z = 0
#             + self.n_ph    # 7) I_mid = 0 injection on internal node
#             + 1             # 8) v_za + v_zb + v_zc = 0
#         )
#         # fmt: on
#         self.num_eqns = self.num_eqns_real + self.num_eqns_complex

#         # book keeping for variables
#         # fmt: off
#         # y = [w, V, I, v, i, lamda]
#         # V = [V1a, V1b, V1c, V2a, V2b, V2c, V3a, V3b, V3c]
#         # I = [I1a, I1b, I1c, I2a, I2b, I2c, I3a, I3b, I3c]
#         # v = [vp_ca, vp_ab, vp_bc, vs_ca, vs_ab, vs_bc, vz_a, vz_b, vz_c]
#         # i = [ip_ca, ip_ab, ip_bc, is_ca, is_ab, is_bc, iz_a, iz_b, iz_c]

#         self.num_vars_real = (1) # w
#         self.num_vars_complex = (
#             3 * self.n_ph    #V
#             + 3 * self.n_ph  #I
#             + 3 * self.n_ph  #v
#             + 3 * self.n_ph  #i
#             + self.n_ph      #lamda
#         )
#         # fmt: on
#         self.num_vars = self.num_vars_real + self.num_vars_complex

#         # create dictionaries to store the offset of each variable in the y vector
#         # fmt:off
#         # 1. dictionary of var offset for real variables
#         self.var_offset_real = {"w" : 0}
#         # 2. dictionary pf var offset for complex variables
#         self.var_offset_complex = {
#             "V" : 0,
#             "I" : 3 * self.n_ph,
#             "v" : 3 * self.n_ph + 3 * self.n_ph,
#             "i" : 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
#             "lamda" : 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph + 3*self.n_ph
#         }
#         # 3. dictionary of var offset for all variables
#         self.var_offset = {
#             "w" : 0,
#             "V" : 1,
#             "I" : 1 + 3 * self.n_ph,
#             "v" : 1 + 3 * self.n_ph + 3 * self.n_ph,
#             "i" : 1 + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
#             "lamda" : 1 + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph
#         }
#         # fmt: on

#         assert len(self.var_offset_real.keys()) + len(
#             self.var_offset_complex.keys()
#         ) == len(self.var_offset.keys())
#         assert self.num_vars == self.var_offset["lamda"] + self.n_ph

#         self.nt = xmer_obj.turns_ratio

#     def initial_guess(self, vals: dict) -> sps.coo_array:
#         # y = [w, V, I, v, i, lamda]
#         y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

#         idx_w = self.var_offset["w"]
#         y_0[idx_w, 0] = vals["w"]

#         phases_without_n = [ph for ph in self.get_phases() if ph != "N"]

#         from_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.pri_volt)))
#         v_phasors_dict = {
#             k: 1e3 * v for (k, v) in utils.get_vector_phasors(from_ph_dict).items()
#         }
#         v_phasors_from = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

#         to_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.sec_volt)))
#         v_phasors_dict = {
#             k: 1e3 * v for (k, v) in utils.get_vector_phasors(to_ph_dict).items()
#         }
#         v_phasors_to = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

#         idx_vfrom_start = self.var_offset["v"]
#         idx_vfrom_end = idx_vfrom_start + self.n_ph
#         idx_vto_start = idx_vfrom_end
#         idx_vto_end = idx_vto_start + self.n_ph

#         # y_0[idx_vfrom_start:idx_vfrom_end, 0] = v_phasors_from
#         # y_0[idx_vto_start:idx_vto_end, 0] = v_phasors_to

#         idx_Vfrom_start = self.var_offset["V"]
#         idx_Vfrom_end = idx_Vfrom_start + self.n_ph
#         idx_Vto_start = idx_Vfrom_end + self.n_ph
#         idx_Vto_end = idx_Vto_start + self.n_ph
#         y_0[idx_Vfrom_start:idx_Vfrom_end, 0] = v_phasors_from / np.sqrt(3)
#         # y_0[idx_Vto_start:idx_Vto_end, 0] = v_phasors_to / np.sqrt(3)
#         y_0[idx_Vto_start:idx_Vto_end, 0] = [
#             -403.038 + 263.2706j,
#             423.904 + 224.2715j,
#             -20.8661 - 487.542j,
#         ]

#         # S = 1e3 * self.obj.kVA
#         # i_phasors_from = (S / v_phasors_from).conjugate()
#         # i_phasors_to = (S / v_phasors_to).conjugate()
#         # idx_i_pri_start = self.var_offset["i"]
#         # idx_i_pri_end = idx_i_pri_start + self.n_ph
#         # # idx_i_sec_start = idx_i_pri_end
#         # # idx_i_sec_end = idx_i_sec_start + self.n_ph
#         # y_0[idx_i_pri_start:idx_i_pri_end, 0] = i_phasors_from
#         # # y_0[idx_i_sec_start:idx_i_sec_end, 0] = i_phasors_to

#         # idx_i_z_start = self.var_offset["i"] + 2 * self.n_ph
#         # idx_i_z_end = idx_i_z_start + self.n_ph
#         # y_0[idx_i_z_start:idx_i_z_end, 0] = i_phasors_to

#         # # idx_lamda_start = self.var_offset["lamda"]
#         # # idx_lamda_end = idx_lamda_start + self.n_ph
#         # # print(f">> self.L: {self.L}")
#         # # print(f">> i_phasors_to: {i_phasors_to}")
#         # # val = np.matmul(self.L, (i_phasors_to.reshape(-1, 1)))
#         # # print(f">> val: {val}")
#         # # y_0[idx_lamda_start:idx_lamda_end, 0] = val

#         return y_0

#     def get_local_idx(
#         self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
#     ) -> int:
#         assert var in self.var_offset.keys()

#         if var == "w":
#             assert ph is None
#             assert side is None

#         assert var not in [
#             "v",
#             "i",
#         ], f"to be implemented for var={var}"

#         # the NodeSide.TO is only valid for "V" and "I"
#         if side == NodeSide.TO:
#             assert var in ["V", "I"]

#         side_offset = 0
#         if side == NodeSide.TO:
#             side_offset = 2 * self.n_ph

#         phase_offset = 0 if ph is None else self.get_phases().index(ph)

#         if val_type == ValType.REAL:
#             return self.var_offset[var] + side_offset + phase_offset
#         elif val_type == ValType.IMAG:
#             return self.var_offset_complex[var] + side_offset + phase_offset

#     def get_id(self):
#         return self.obj.id

#     def get_M_powerflow_inner(self):
#         """
#         1) This function returns the M matrix for power flow simulation
#         2) First we create the node-incidence A, identity, zero and other coefficient matrices required for each eqn
#         and then stack them to form the M matrix
#         3) ID_* are identity matrices
#         4) Z_* are zero matrices
#         """

#         # identity matrices for the equations
#         # I ->[I1a, I1b, I1c, I2a, I2b, I2c, I3a, I3b, I3c]
#         # fmt: off
#         I_coeff = np.array([
#             # I1a,  I1b,  I1c,  I2a,  I2b,  I2c,  I3a,  I3b,  I3c
#             [-1,    0,    0,    0,    0,    0,    0,    0,    0],  #
#             [0,    -1,    0,    0,    0,    0,    0,    0,    0],  #
#             [0,     0,   -1,    0,    0,    0,    0,    0,    0],  #
#             [0,     0,    0,   -1,    0,    0,    0,    0,    0],  #
#             [0,     0,    0,    0,   -1,    0,    0,    0,    0],  #
#             [0,     0,    0,    0,    0,    0,   -1,    0,    0],  #
#             [0,     0,    0,    0,    0,    0,    0,   -1,    0],  #
#             [0,     0,    0,    0,    0,    0,    0,    0,   -1],  #
#         ])
#         # fmt: on
#         # Id_I = sps.identity(3 * self.n_ph, format="coo").tocsc()

#         # create the node-incidence matrix
#         # fmt:off
#         A = np.array([
#             #[ip_ca, ip_ab, ip_bc, is_ca, is_ab, is_bc, iz_a, iz_b, iz_c]
#             [-1,       1,     0,     0,     0,     0,     0,    0,    0], # KCL at primary phase a
#             [0,       -1,     1,     0,     0,     0,     0,    0,    0], # KCL at primary phase b
#             [1,        0,     -1,    0,     0,     0,     0,    0,    0], # KCL at primary phase c
#             [0,        0,     0,    -1,     1,     0,     1,    0,    0], # KCL at sec phase a (modified)
#             [0,        0,     0,     0,    -1,     1,     0,    1,    0], # KCL at sec phase b (modified)
#             [0,        0,     0,     0,     0,     0,    -1,    0,    0], # KCL at leakage branch phase a
#             [0,        0,     0,     0,     0,     0,     0,   -1,    0], # KCL at leakage branch phase b
#             [0,        0,     0,     0,     0,     0,     0,    0,   -1], # KCL at leakage branch phase c
#         ])

#         # V_coeff = A.T
#         V_coeff = np.array([
#             # V1_a, V1_b, V1_c, V2_a, V2_b, V2_c, V3_a, V3_b, V3_c
#             [-1,    0,    1,    0,    0,    0,    0,    0,    0],  #
#             [1,    -1,    0,    0,    0,    0,    0,    0,    0],  #
#             [0,     0,    0,    0,    0,    0,    0,    0,    0],  #
#             [0,     0,    0,   -1,    0,    0,    0,    0,    0],  #
#             [0,     0,    0,    0,   -1,    0,    0,    0,    0],  #
#             [0,     0,    0,    0,    0,   -1,    0,    0,    0],  #
#             [0,     0,    0,    1,    0,    0,   -1,    0,    0],  #
#             [0,     0,    0,    0,    1,    0,    0,   -1,    0],  #
#             [0,     0,    0,    0,    0,    1,    0,    0,   -1],  #
#         ])
#         # fmt:on

#         # v ->[vp_ca, vp_ab, vp_bc, vs_ca, vs_ab, vs_bc, vz_ca, vz_ab, vz_bc]
#         # fmt: off
#         v_coeff = [
#             # vp_ca, vp_ab, vp_bc, vs_ca, vs_ab, vs_bc, vz_ca, vz_ab, vz_bc
#             [-1,     0,     0,     0,     0,     0,     0,    0,    0],  #
#             [0,     -1,     0,     0,     0,     0,     0,    0,    0],  #
#             [1,      1,     1,     0,     0,     0,     0,    0,    0],  #
#             [0,      0,     0,     0,     2/3,   1/3,   0,   -2/3, -1/3],  #
#             [0,      0,     0,     1/3,   0,     2/3,  -1/3,  0,   -2/3],  #
#             [0,      0,     0,     2/3,   1/3,   0,    -2/3, -1/3,  0],  #
#             [0,      0,     0,     0,     0,     0,     0,    0,    0],  #
#             [0,      0,     0,     0,     0,     0,     0,    0,    0],  #
#             [0,      0,     0,     0,     0,     0,     0,    0,    0],  #
#         ]
#         # fmt: on

#         # Identity matrix based on no. of phases
#         Id_ph = sps.identity(self.n_ph, format="coo").tocsc()

#         # create coefficient matrix for leakage branch eqns
#         # print shape of R_mat
#         print(f"shape of self.R: {self.R.shape}")
#         Z_nph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
#         print(f"shape of Z_nph: {Z_nph.shape}")
#         L_mat = sps.bmat([[Z_nph, self.L, Z_nph]], format="coo")
#         R_mat = sps.bmat([[Z_nph, self.R, Z_nph]], format="coo")

#         print(f">> L_mat: {L_mat}")
#         print(f">> R_mat: {R_mat}")

#         # Voltage and current transformation equations
#         nt = self.nt
#         nt_r = 1 / nt  # reciprocal of turns ratio
#         print(f">> nt: {nt}")
#         print(f">> nt_r: {nt_r}")

#         # fmt:off
#         N_i = np.array([
#             # ip_ca    ip_ab   ip_bc   is_ca    is_ab    is_bc   iz_a   iz_b   iz_c
#             [  1,       0,     0,     nt_r,    0,      0,       0,     0,     0   ], # is_ac - nt_r*ip_ac = 0
#             [  0,       1,     0,      0,       nt_r,   0,       0,     0,     0   ], # is_ab - nt_r*ip_b = 0
#             [  0,       0,     1,      0,        0,      nt_r,   0,     0,     0   ], # is_bc - nt_r*ip_bc = 0
#         ])

#         N_v = np.array([
#             # vp_ca    vp_ab   vp_bc   vs_ca    vs_ab    vs_bc   vz_ca   vz_ab   vz_bc
#             [  1,       0,     0,       -nt,       0,      0,      0,     0,     0   ], # vs_ca - vp_ca = 0
#             [  0,       1,     0,       0,       -nt,      0,      0,     0,     0   ], # vs_ab - vp_ab = 0
#             [  0,       0,     1,       0,       0,       -nt,      0,     0,     0   ], # vs_bc - vp_bc = 0
#         ])

#         # fmt:on

#         # zero vectors for w for each equation
#         # zero vectors for w for each equation
#         Z_w1 = sps.lil_matrix((A.shape[0], 1), dtype=float)
#         Z_w2 = sps.lil_matrix((V_coeff.shape[0], 1), dtype=float)
#         Z_w3 = sps.lil_matrix((N_v.shape[0], 1), dtype=float)
#         Z_w4 = sps.lil_matrix((N_i.shape[0], 1), dtype=float)
#         Z_w5 = sps.lil_matrix((self.L.shape[0], 1), dtype=float)
#         Z_w6 = sps.lil_matrix((self.R.shape[0], 1), dtype=float)
#         Z_w7 = sps.lil_matrix((self.n_ph, 1), dtype=float)

#         eqn6_v = sps.bmat([[Z_nph, Z_nph, Id_ph]], format="coo")
#         eqn7_I = sps.bmat([[Z_nph, Id_ph, Z_nph]], format="coo")

#         # Z_w8 = sps.lil_matrix((1, 1), dtype=float)

#         # v_za + v_zb + v_zc = 0
#         v_z_coeff = sps.lil_matrix((1, 3 * self.n_ph), dtype=float)
#         v_z_coeff[0, 6] = 1
#         v_z_coeff[0, 7] = 1
#         v_z_coeff[0, 8] = 1

#         # fmt:off
#         M = sps.bmat([
#             #[w        V         I         v           i       lamda]
#             [ Z_w1,   None,     I_coeff,   None,       A,      None],      # 1) KCL: I_coeff * I + Ai = 0
#             [ Z_w2,   V_coeff,  None,      v_coeff,    None,   None],      # 2) KVL: ATV - v = 0
#             [ Z_w3,   None,     None,      N_v,        None,   None],      # 3) v_s - v_p / nt = 0
#             [ Z_w4,   None,     None,      None,       N_i,    None],      # 4) i_s - nt * i_p = 0
#             [ Z_w5,   None,     None,      None,       L_mat,  Id_ph],    # 5) Li - lamda = 0
#             [ Z_w6,   None,     None,      eqn6_v,     R_mat,  None],      # 6) Ri + jw * lamda + v = 0
#             [ Z_w7,   None,     eqn7_I,    None,       None,   None],      # 7) I_mid = 0 injection on internal node
#             [ None,   None,     None,      v_z_coeff,  None,   None],      # 8) v_za + v_zb + v_zc = 0
#         ])
#         # fmt:on

#         return M

#     def get_u_powerflow(self):
#         u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=float)
#         return u[: self.num_eqns], u[self.num_eqns :]

#     def get_fy_powerflow(
#         self, y_re: sps.coo_array, y_im: sps.coo_array
#     ) -> tuple[sps.coo_array, sps.coo_array]:
#         """
#         1. This function returns the non-linear terms of every equation
#         2. In this transformer model, eqn6 has non-linearity due to product of w and lamda
#         3. This function is to be called from the newton-raphson method on every iteration.
#         4. 'y' is the part of overall-y vector that pertains to this line.
#         """

#         y = y_re.astype(complex)
#         y[self.num_vars_real :] += 1j * y_im

#         fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

#         # fy update for eqn 6

#         # index of w in y vector
#         idx_w = self.var_offset["w"]
#         w = y[idx_w, 0]

#         # start stop index of lamda in y vector
#         idx_lamda_start = self.var_offset["lamda"]
#         idx_lamda_end = idx_lamda_start + self.n_ph

#         # start stop index of eqn 6
#         # fmt: off
#         idx_eq6_start = (
#             (2 * self.n_ph + 2)           # 1)KCL:-I + Ai = 0
#             + 3 * self.n_ph         # 2)KVL: ATv - V = 0
#             + self.n_ph             # 3)v_s-v_p/nt = 0
#             + self.n_ph             # 4)i_s-nt*i_p = 0
#             + self.n_ph             # 5)Li-lamda = 0
#         )
#         # fmt: on

#         idx_eq6_end = idx_eq6_start + self.n_ph

#         fy[idx_eq6_start:idx_eq6_end] = 0 + 1j * (w * y[idx_lamda_start:idx_lamda_end])

#         return fy.real, fy[self.num_eqns_real :].imag

#     def get_pd_fy_split(
#         self, y_real: sps.coo_array, y_imag: sps.coo_array
#     ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:

#         assert self.num_vars == y_real.shape[0]
#         assert self.num_vars_complex == y_imag.shape[0]

#         pd_fy_split = sps.coo_array(
#             (
#                 self.num_eqns + self.num_eqns_complex,
#                 self.num_vars + self.num_vars_complex,
#             ),
#             dtype=float,
#         ).tocsc()

#         w_col_offset = self.var_offset["w"]
#         w = y_real[w_col_offset, 0]

#         lamda_re_start_offset = self.var_offset["lamda"]
#         lamda_im_start_offset = self.var_offset_complex["lamda"]

#         # fmt:off
#         # eq6_re
#         eq6_re_start_row = (
#             (2 * self.n_ph + 2) # 1)KCL
#             + 3 * self.n_ph     # 2)KVL
#             + self.n_ph     # 3)v_p - nt*v_s = 0
#             + self.n_ph     # 4)i_p - nt_r * i_s = 0
#             + self.n_ph     # 5)L*i - lamda = 0
#         )
#         # fmt:on

#         # update the pd_fy_split for eqn 6 real part
#         for offset in range(self.n_ph):
#             row = eq6_re_start_row + offset

#             lamda_re_col_offset = lamda_re_start_offset + offset
#             lamda_im_col_offset = lamda_im_start_offset + offset
#             lamda_re = y_real[lamda_re_col_offset, 0]
#             lamda_im = y_imag[lamda_im_col_offset, 0]

#             # print(
#             #     f">> get_pd_fy_split(): row: {row}"
#             # )
#             # print(
#             #     f">> get_pd_fy_split(): lamda_re_col_offset: {lamda_re_col_offset}, lamda_im_col_offset: {lamda_im_col_offset}"
#             # )
#             # print(f">> get_pd_fy_split(): self.num_vars: {self.num_vars}")

#             # derivatives
#             # wrt w
#             pd_fy_split[row, w_col_offset] = -lamda_im

#             # wrt lamda_re : derivative is zero

#             # wrt lamda_im
#             pd_fy_split[row, self.num_vars + lamda_im_col_offset] = -w

#         # update the pd_fy_split for eqn 6 imaginary part
#         eq6_im_start_row = self.num_eqns + eq6_re_start_row
#         for offset in range(self.n_ph):
#             row = eq6_im_start_row + offset

#             lamda_re_col_offset = lamda_re_start_offset + offset
#             lamda_im_col_offset = lamda_im_start_offset + offset
#             lamda_re = y_real[lamda_re_col_offset, 0]
#             lamda_im = y_imag[lamda_im_col_offset, 0]

#             # print(
#             #     f">> get_pd_fy_split(): row: {row}"
#             # )
#             # print(
#             #     f">> get_pd_fy_split(): lamda_re_col_offset: {lamda_re_col_offset}, lamda_im_col_offset: {lamda_im_col_offset}"
#             # )
#             # print(f">> get_pd_fy_split(): self.num_vars: {self.num_vars}")

#             # wrt w
#             pd_fy_split[row, w_col_offset] = lamda_re

#             # wrt lamda_re
#             pd_fy_split[row, lamda_re_col_offset] = w

#             # wrt lamda_im : derivative is zero

#         rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
#         ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
#         ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
#         ii = pd_fy_split[self.num_eqns :, self.num_vars :]

#         # input("continue?")
#         # plt.spy(pd_fy_split)
#         # plt.show()

#         return (rr, ri, ir, ii)


###############################################################################################################################################
# 3ph delta-delta transformer model
class ThreePhaseDDStepDownTransformerModel(TransformerModel):
    def __init__(self, xmer_obj: Transformer):
        super().__init__(xmer_obj)

        # declare num of terminals
        self.num_term = self.obj.terminal.get_num_term()

        # book keeping of the equations
        # fmt: off
        self.num_eqns_real = 0
        self.num_eqns_complex = (
            2 * self.n_ph + 2   # 1)KCL
            + 3 * self.n_ph # 2)KVL
            + self.n_ph    # 3) v_s - v_p / nt =0
            + self.n_ph    # 4) i_s - nt * i_p = 0
            + self.n_ph    # 5) L*i_z - lamda =0
            + self.n_ph    # 6) R*i_z + jw*lamda - v_z = 0
            + self.n_ph    # 7) I_mid = 0 injection on internal node
            + 1             # 8) v_za + v_zb + v_zc = 0
            # + 1             # 9) V2a + V2b + V2c - 3Vsn = 0
            + 1            # 10) V1a + V1b + V1c - 3Vpn = 0
            + 1            # 11) Vpn = nt * Vsn
        )
        # fmt: on
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book keeping for variables
        # fmt: off
        # y = [w, V, I, v, i, lamda]
        # V = [V1a, V1b, V1c, V2a', V2b', V2c', V_n, V2a, V2b, V2c, V3a, V3b, V3c]
        # I = [I1a, I1b, I1c, I2a, I2b, I2c, I3a, I3b, I3c]
        # v = [vp_ca, vp_ab, vp_bc, vs_ca, vs_ab, vs_bc, vz_a, vz_b, vz_c]
        # i = [ip_ca, ip_ab, ip_bc, is_ca, is_ab, is_bc, iz_a, iz_b, iz_c]

        self.num_vars_real = (1) # w
        self.num_vars_complex = (
            (3 * self.n_ph + 2)   #V
            + 3 * self.n_ph  #I
            + 3 * self.n_ph  #v
            + 3 * self.n_ph  #i
            + self.n_ph      #lamda
        )
        # fmt: on
        self.num_vars = self.num_vars_real + self.num_vars_complex

        # create dictionaries to store the offset of each variable in the y vector
        # fmt:off
        # 1. dictionary of var offset for real variables
        self.var_offset_real = {"w" : 0}
        # 2. dictionary pf var offset for complex variables
        self.var_offset_complex = {
            "V" : 0,
            "I" : 3 * self.n_ph + 2,
            "v" : 3 * self.n_ph + 2 + 3 * self.n_ph,
            "i" : 3 * self.n_ph + 2 + 3 * self.n_ph + 3 * self.n_ph,
            "lamda" : 3 * self.n_ph + 2 + 3 * self.n_ph + 3 * self.n_ph + 3*self.n_ph
        }
        # 3. dictionary of var offset for all variables
        self.var_offset = {
            "w" : 0,
            "V" : 1,
            "I" : 1 + 3 * self.n_ph + 2,
            "v" : 1 + 3 * self.n_ph + 2 + 3 * self.n_ph,
            "i" : 1 + 3 * self.n_ph + 2 + 3 * self.n_ph + 3 * self.n_ph,
            "lamda" : 1 + 3 * self.n_ph + 2 + 3 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph
        }
        # fmt: on

        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["lamda"] + self.n_ph

        # dynamic :
        # this model has same eqns and variable for powerflow and dynamic simulations
        self.num_eqns_dynamic = self.num_eqns
        self.num_vars_dynamic = self.num_vars
        self.var_offset_dynamic = self.var_offset.copy()

        self.nt = xmer_obj.turns_ratio

#######################################Powerflow functions#############################
    def initial_guess(self, vals: dict) -> sps.coo_array:
        # y = [w, V, I, v, i, lamda]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        phases_without_n = [ph for ph in self.get_phases() if ph != "N"]

        from_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.pri_volt)))
        v_phasors_dict = {
            k: v for (k, v) in utils.get_vector_phasors(from_ph_dict).items()
        }
        v_phasors_from = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

        to_ph_dict = dict(zip(phases_without_n, itertools.repeat(self.obj.sec_volt)))
        v_phasors_dict = {
            k: v for (k, v) in utils.get_vector_phasors(to_ph_dict).items()
        }
        v_phasors_to = np.array(list(v_phasors_dict.values())).reshape(-1, 1)

        idx_vfrom_start = self.var_offset["v"]
        idx_vfrom_end = idx_vfrom_start + self.n_ph
        # idx_vto_start = idx_vfrom_end
        # idx_vto_end = idx_vto_start + self.n_ph

        # y_0[idx_vfrom_start:idx_vfrom_end, 0] = v_phasors_from
        # y_0[idx_vto_start:idx_vto_end, 0] = v_phasors_to

        idx_Vfrom_start = self.var_offset["V"]
        idx_Vfrom_end = idx_Vfrom_start + self.n_ph
        # idx_Vto_start = idx_Vfrom_end + self.n_ph
        # idx_Vto_end = idx_Vto_start + self.n_ph
        y_0[idx_Vfrom_start:idx_Vfrom_end, 0] = v_phasors_from / np.sqrt(3)
        # y_0[idx_Vto_start:idx_Vto_end, 0] = v_phasors_to / np.sqrt(3)
        # y_0[idx_Vto_start:idx_Vto_end, 0] = [
        #     -403.038 + 263.2706j,
        #     423.904 + 224.2715j,
        #     -20.8661 - 487.542j,
        # ]

        # S = 1e3 * self.obj.kVA
        # i_phasors_from = (S / v_phasors_from).conjugate()
        # i_phasors_to = (S / v_phasors_to).conjugate()
        # idx_i_pri_start = self.var_offset["i"]
        # idx_i_pri_end = idx_i_pri_start + self.n_ph
        # # idx_i_sec_start = idx_i_pri_end
        # # idx_i_sec_end = idx_i_sec_start + self.n_ph
        # y_0[idx_i_pri_start:idx_i_pri_end, 0] = i_phasors_from
        # # y_0[idx_i_sec_start:idx_i_sec_end, 0] = i_phasors_to

        # idx_i_z_start = self.var_offset["i"] + 2 * self.n_ph
        # idx_i_z_end = idx_i_z_start + self.n_ph
        # y_0[idx_i_z_start:idx_i_z_end, 0] = i_phasors_to

        # # idx_lamda_start = self.var_offset["lamda"]
        # # idx_lamda_end = idx_lamda_start + self.n_ph
        # # print(f">> self.L: {self.L}")
        # # print(f">> i_phasors_to: {i_phasors_to}")
        # # val = np.matmul(self.L, (i_phasors_to.reshape(-1, 1)))
        # # print(f">> val: {val}")
        # # y_0[idx_lamda_start:idx_lamda_end, 0] = val

        return y_0

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        assert var not in [
            "v",
            "i",
        ], f"to be implemented for var={var}"

        # the NodeSide.TO is only valid for "V" and "I"
        if side == NodeSide.TO:
            assert var in ["V", "I"]

        side_offset = 0
        if side == NodeSide.TO:
            if var == "V":
                side_offset = 2 * self.n_ph + 2
            elif var == "I":
                side_offset = 2 * self.n_ph
            else:
                raise ValueError(f"Invalid side={side} for var={var}")

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_id(self):
        return self.obj.id

    def get_M_powerflow_inner(self):
        """
        1) This function returns the M matrix for power flow simulation
        2) First we create the node-incidence A, identity, zero and other coefficient matrices required for each eqn
        and then stack them to form the M matrix
        3) ID_* are identity matrices
        4) Z_* are zero matrices
        """

        # identity matrices for the equations
        # I ->[I1a, I1b, I1c, I2a, I2b, I2c, I3a, I3b, I3c]
        # fmt: off
        I_coeff = np.array([
            # I1a,  I1b,  I1c,  I2a,  I2b,  I2c,  I3a,  I3b,  I3c
            [-1,    0,    0,    0,    0,    0,    0,    0,    0],  #
            [0,    -1,    0,    0,    0,    0,    0,    0,    0],  #
            [0,     0,   -1,    0,    0,    0,    0,    0,    0],  #

            [0,     0,    0,   -1,    0,    0,    0,    0,    0],  #
            [0,     0,    0,    0,   -1,    0,    0,    0,    0],  # 
            
            [0,     0,    0,    0,    0,    0,   -1,    0,    0],  #
            [0,     0,    0,    0,    0,    0,    0,   -1,    0],  #
            [0,     0,    0,    0,    0,    0,    0,    0,   -1],  #
        ])
        # fmt: on
        # Id_I = sps.identity(3 * self.n_ph, format="coo").tocsc()

        # create the node-incidence matrix
        # fmt:off
        A = np.array([
            #[ip_ca, ip_ab, ip_bc, is_ca, is_ab, is_bc, iz_a, iz_b, iz_c]
            [-1,       1,     0,     0,     0,     0,     0,    0,    0], # KCL at primary phase a
            [0,       -1,     1,     0,     0,     0,     0,    0,    0], # KCL at primary phase b
            [1,        0,     -1,    0,     0,     0,     0,    0,    0], # KCL at primary phase c
            
            [0,        0,     0,    -1,     1,     0,     1,    0,    0], # KCL at sec phase a (modified)
            [0,        0,     0,     0,    -1,     1,     0,    1,    0], # KCL at sec phase b (modified)
            
            [0,        0,     0,     0,     0,     0,    -1,    0,    0], # KCL at leakage branch phase a
            [0,        0,     0,     0,     0,     0,     0,   -1,    0], # KCL at leakage branch phase b
            [0,        0,     0,     0,     0,     0,     0,    0,   -1], # KCL at leakage branch phase c
        ])

        # V_coeff = A.T
        V_coeff = np.array([
            # V1_a, V1_b, V1_c, V_pn, V_sn, V2_a, V2_b, V2_c, V3_a, V3_b, V3_c
            [-1,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],  # V1_a - V1_c = 0
            [1,    -1,    0,    0,    0,    0,    0,    0,    0,    0,    0],  # 
            [0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  # 

            [0,     0,    0,    0,    0,   -1,    1,    0,    0,    0,    0],  # 
            [0,     0,    0,    0,    0,    0,   -1,    1,    0,    0,    0],  # 
            [0,     0,    0,    0,   -3,    1,    1,    1,    0,    0,    0],  # 
            
            [0,     0,    0,    0,    0,    1,    0,    0,   -1,    0,    0],  # 
            [0,     0,    0,    0,    0,    0,    1,    0,    0,   -1,    0],  # 
            [0,     0,    0,    0,    0,    0,    0,    1,    0,    0,   -1],  # 
        ])
        # fmt:on

        # v ->[vp_ca, vp_ab, vp_bc, vs_ca, vs_ab, vs_bc, vz_ca, vz_ab, vz_bc]
        # fmt: off
        v_coeff = [
            # vp_ca, vp_ab, vp_bc, vs_ca, vs_ab, vs_bc, vz_ca, vz_ab, vz_bc
            [-1,     0,     0,     0,     0,     0,     0,      0,     0 ],  #
            [0,     -1,     0,     0,     0,     0,     0,      0,     0 ],  #
            [1,      1,     1,     0,     0,     0,     0,      0,     0 ],  #

            [0,      0,     0,     0,     1,     0,     0,     -1,     0 ],  #
            [0,      0,     0,     0,     0,     1,     0,      0,    -1 ],  #
            [0,      0,     0,     0,     0,     0,     0,      0,     0 ],  #
            
            [0,      0,     0,     0,     0,     0,     0,      0,     0 ],  #
            [0,      0,     0,     0,     0,     0,     0,      0,     0 ],  #
            [0,      0,     0,     0,     0,     0,     0,      0,     0 ],  #
        ] 
        # fmt: on

        # Identity matrix based on no. of phases
        Id_ph = sps.identity(self.n_ph, format="coo").tocsc()

        # create coefficient matrix for leakage branch eqns
        # print shape of R_mat
        print(f"shape of self.R: {self.R.shape}")
        Z_nph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        print(f"shape of Z_nph: {Z_nph.shape}")
        L_mat = sps.bmat([[Z_nph, self.L, Z_nph]], format="coo")
        R_mat = sps.bmat([[Z_nph, self.R, Z_nph]], format="coo")

        print(f">> L_mat: {L_mat}")
        print(f">> R_mat: {R_mat}")

        # Voltage and current transformation equations
        nt = self.nt
        nt_r = 1 / nt  # reciprocal of turns ratio
        print(f">> nt: {nt}")
        print(f">> nt_r: {nt_r}")

        # fmt:off
        N_i = np.array([
            # ip_ca    ip_ab   ip_bc   is_ca    is_ab    is_bc   iz_a   iz_b   iz_c
            [  1,       0,     0,     nt_r,    0,      0,       0,     0,     0   ], # is_ac - nt_r*ip_ac = 0
            [  0,       1,     0,      0,       nt_r,   0,       0,     0,     0   ], # is_ab - nt_r*ip_b = 0
            [  0,       0,     1,      0,        0,      nt_r,   0,     0,     0   ], # is_bc - nt_r*ip_bc = 0
        ])

        N_v = np.array([
            # vp_ca    vp_ab   vp_bc   vs_ca    vs_ab    vs_bc   vz_ca   vz_ab   vz_bc
            [  1,       0,     0,       -nt,       0,      0,      0,     0,     0   ], # vs_ca - vp_ca = 0
            [  0,       1,     0,       0,       -nt,      0,      0,     0,     0   ], # vs_ab - vp_ab = 0
            [  0,       0,     1,       0,       0,       -nt,      0,     0,     0   ], # vs_bc - vp_bc = 0
        ])

        # fmt:on

        # zero vectors for w for each equation
        # zero vectors for w for each equation
        Z_w1 = sps.lil_matrix((A.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((V_coeff.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((N_v.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((N_i.shape[0], 1), dtype=float)
        Z_w5 = sps.lil_matrix((self.L.shape[0], 1), dtype=float)
        Z_w6 = sps.lil_matrix((self.R.shape[0], 1), dtype=float)
        Z_w7 = sps.lil_matrix((self.n_ph, 1), dtype=float)

        eqn6_v = sps.bmat([[Z_nph, Z_nph, Id_ph]], format="coo")
        eqn7_I = sps.bmat([[Z_nph, Id_ph, Z_nph]], format="coo")

        # Z_w8 = sps.lil_matrix((1, 1), dtype=float)

        # v_za + v_zb + v_zc = 0
        v_z_coeff = sps.lil_matrix((1, 3 * self.n_ph), dtype=float)
        v_z_coeff[0, 6] = 1
        v_z_coeff[0, 7] = 1
        v_z_coeff[0, 8] = 1

        # fmt:off
        # V2a' + V2b' + V2c' - 3Vn = 0
        V_coeff2 = np.array([
            # V1_a, V1_b, V1_c, V_pn, V_sn, V2_a, V2_b, V2_c,  V3_a, V3_b, V3_c
            [1,     1,    1,    -3,    0,    0,    0,    0,    0,    0,    0],  # 
        ])

        V_coeff3 = np.array([
            # V1_a, V1_b, V1_c, V_pn, V_sn, V2_a, V2_b, V2_c,  V3_a, V3_b, V3_c
            [0,     0,    0,    1,    -nt,    0,    0,    0,    0,    0,    0],  # 
        ])
        # fmt:on

        # fmt:off
        M = sps.bmat([
            #[w        V         I         v            i       lamda]
            [ Z_w1,   None,     I_coeff,   None,        A,      None],      # 1) KCL: I_coeff * I + Ai = 0
            [ Z_w2,   V_coeff,  None,      v_coeff,     None,   None],      # 2) KVL: ATV - v = 0
            [ Z_w3,   None,     None,      N_v,         None,   None],      # 3) v_s - v_p / nt = 0
            [ Z_w4,   None,     None,      None,        N_i,    None],      # 4) i_s - nt * i_p = 0
            [ Z_w5,   None,     None,      None,        L_mat,  Id_ph],    # 5) Li - lamda = 0
            [ Z_w6,   None,     None,      eqn6_v,      R_mat,  None],      # 6) Ri + jw * lamda + v = 0
            [ Z_w7,   None,     eqn7_I,    None,        None,   None],      # 7) I_mid = 0 injection on internal node
            [ None,   None,     None,      v_z_coeff,   None,   None],      # 8) v_za + v_zb + v_zc = 0
            [ None,   V_coeff2, None,      None,        None,   None],      # 8) V1a + V1b + V1c - 3Vpn = 0
            [ None,   V_coeff3, None,      None,        None,   None],      # 9) Vpn - nt * Vsn = 0
        ])
        # fmt:on

        return M

    def get_u_powerflow(self):
        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=float)
        return u[: self.num_eqns], u[self.num_eqns :]

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. In this transformer model, eqn6 has non-linearity due to product of w and lamda
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """

        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # fy update for eqn 6

        # index of w in y vector
        idx_w = self.var_offset["w"]
        w = y[idx_w, 0]

        # start stop index of lamda in y vector
        idx_lamda_start = self.var_offset["lamda"]
        idx_lamda_end = idx_lamda_start + self.n_ph

        # start stop index of eqn 6
        # fmt: off
        idx_eq6_start = (
            (2 * self.n_ph + 2)     # 1)KCL:-I + Ai = 0
            + 3 * self.n_ph         # 2)KVL: ATv - V = 0
            + self.n_ph             # 3)v_s-v_p/nt = 0
            + self.n_ph             # 4)i_s-nt*i_p = 0
            + self.n_ph             # 5)Li-lamda = 0
        )
        # fmt: on

        idx_eq6_end = idx_eq6_start + self.n_ph

        fy[idx_eq6_start:idx_eq6_end] = 0 + 1j * (w * y[idx_lamda_start:idx_lamda_end])

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:

        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]

        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.var_offset_complex["lamda"]

        # fmt:off
        # eq6_re
        eq6_re_start_row = (
            (2 * self.n_ph + 2) # 1)KCL
            + 3 * self.n_ph     # 2)KVL
            + self.n_ph     # 3)v_p - nt*v_s = 0
            + self.n_ph     # 4)i_p - nt_r * i_s = 0
            + self.n_ph     # 5)L*i - lamda = 0  
        )
        # fmt:on

        # update the pd_fy_split for eqn 6 real part
        for offset in range(self.n_ph):
            row = eq6_re_start_row + offset

            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # print(
            #     f">> get_pd_fy_split(): row: {row}"
            # )
            # print(
            #     f">> get_pd_fy_split(): lamda_re_col_offset: {lamda_re_col_offset}, lamda_im_col_offset: {lamda_im_col_offset}"
            # )
            # print(f">> get_pd_fy_split(): self.num_vars: {self.num_vars}")

            # derivatives
            # wrt w
            pd_fy_split[row, w_col_offset] = -lamda_im

            # wrt lamda_re : derivative is zero

            # wrt lamda_im
            pd_fy_split[row, self.num_vars + lamda_im_col_offset] = -w

        # update the pd_fy_split for eqn 6 imaginary part
        eq6_im_start_row = self.num_eqns + eq6_re_start_row
        for offset in range(self.n_ph):
            row = eq6_im_start_row + offset

            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # print(
            #     f">> get_pd_fy_split(): row: {row}"
            # )
            # print(
            #     f">> get_pd_fy_split(): lamda_re_col_offset: {lamda_re_col_offset}, lamda_im_col_offset: {lamda_im_col_offset}"
            # )
            # print(f">> get_pd_fy_split(): self.num_vars: {self.num_vars}")

            # wrt w
            pd_fy_split[row, w_col_offset] = lamda_re

            # wrt lamda_re
            pd_fy_split[row, lamda_re_col_offset] = w

            # wrt lamda_im : derivative is zero

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        # input("continue?")
        # plt.spy(pd_fy_split)
        # plt.show()

        return (rr, ri, ir, ii)

###############################Dynamic Simulation Functions##################################

    def get_local_idx_dynamic(
            self, var : str, ph : str | None, side: NodeSide | None) -> int:
        
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None

        assert var not in [
            "v",
            "i"
        ], f"to be implemented for var = {var}"

        side_offset = 0
        # the NodeSide.TO is valid for "V" and "I"
        if side == NodeSide.TO:
            assert var in ["V", "I"]
            if var == "V":
                side_offset = 2 * self.n_ph + 2
            elif var == "I":
                side_offset = 2 * self.n_ph
            else:  
                ValueError(f"Invalid side = {side} for var = {var}")

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset
    
    def initial_guess_dynamic(self, y_comp: np.ndarray) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        # variables : [w, V, I, v, i, lamda]
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype = float)

        # initialization from powerflow directly
        vars_counts_real = [
            ("w", 1),
        ]

        for var, count in vars_counts_real:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count

            y0_dyn[idx_var_dyn_start : idx_var_dyn_end] = y_comp[idx_var_pf_start : idx_var_pf_end]

        vars_counts_complex = [
            ("V", 3 * self.n_ph + 2),
            ("I", 3 * self.n_ph),
            ("v", 3 * self.n_ph),
            ("i", 3 * self.n_ph),
            ("lamda", self.n_ph)
        ]

        # needs conversion from phasor to time domain
        for var, count in vars_counts_complex:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count

            y0_dyn[idx_var_dyn_start : idx_var_dyn_end] = np.sqrt(2) * phasor_to_timedomain(y_comp[idx_var_pf_start : idx_var_pf_end])

        return y0_dyn
    
    def get_M_dynamic(self) -> sps.coo_array:
        # for this model the dynamic M is exactly same as that of powerflow
        M = self.get_M_powerflow_inner()
    
        return M
    
    def get_K_dynamic(self) -> sps.coo_array:
        Z_A = sps.lil_matrix((2 * self.n_ph + 2, 3 * self.n_ph ), dtype = float)
        Z_w1 = sps.lil_matrix((Z_A.shape[0], 1), dtype = float)
        Z_Vcoeff = sps.lil_matrix((3 * self.n_ph, 3 * self.n_ph + 2), dtype = float)
        Z_w2 = sps.lil_matrix((Z_Vcoeff.shape[0], 1), dtype = float)
        Z_Nv = sps.lil_matrix((self.n_ph, 3 * self.n_ph), dtype = float)
        Z_w3 = sps.lil_matrix((Z_Nv.shape[0]), dtype = float)
        Z_Ieq7 = sps.lil_matrix((self.n_ph, 3 * self.n_ph), dtype = float)
        z_v = sps.lil_matrix((1, 3 * self.n_ph + 2 ), dtype = float)




        Id_ph = sps.identity(self.n_ph, format = "coo").tocsc()



         # fmt:off
        K = sps.bmat([
            #[w        V         I         v            i       lamda]
            [ Z_w1,   None,     None,      None,        Z_A,    None],      # 1) KCL: I_coeff * I + Ai = 0
            [ None,   Z_Vcoeff,  None,     None,        None,   None],      # 2) KVL: ATV - v = 0
            [ None,   None,     None,      None,        None,   None],      # 4) i_s - nt * i_p = 0
            [ None,   None,     None,      None,        None,  None],       # 5) Li - lamda = 0
            [ None,   None,     None,      Z_Nv,        None,   None],      # 3) v_s - v_p / nt = 0
            [ None,   None,     None,      None,        None,  Id_ph],      # 6) Ri + d(lamda)/dt  + v = 0
            [ None,   None,     Z_Ieq7,    None,        None,   None],      # 7) I_mid = 0 injection on internal node
            [ None,   z_v,      None,      None,        None,   None],      # 8) v_za + v_zb + v_zc = 0
            [ None,   z_v,      None,      None,        None,   None],      # 8) V1a + V1b + V1c - 3Vpn = 0
            [ None,   z_v,      None,      None,        None,   None],      # 9) Vpn - nt * Vsn = 0
        ])
        
        return K
    
    def get_fy_dynamic(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        1. This function returns the non-linear terms of every equation for dynamic simulation
        2. This dynamic model has no non-linearity.

        """
          
        fy = np.zeros(self.num_eqns_dynamic, dtype = float)

        return fy
    
    def get_u_dynamic(self, t: float, y) -> np.ndarray:
        """
        1. This function returns the u vector for dynamic simulation
        2. The dynamic model has zero u vector.
        """
        u = np.zeros(self.num_eqns_dynamic, dtype = float)
        return u

    








        

        
    

