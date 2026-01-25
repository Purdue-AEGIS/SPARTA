from oodesign import *
import utils
from models.model import Model, ValType
from oodesign import GFMInverter, GFMInverter3Ph, Source
import numpy as np
import scipy.sparse as sps
import const
from const import NodeSide, StudyType
from pprint import pformat
from utils import phasor_to_timedomain

# from models.component_models.source_model import SourceModel
from models.component_models.source_model__gfl_study import SourceModel

from models.component_models.inverters.inverter_model import *

#####################################################
# global coefficients block
n_ph = 3
n_qd = 2
Id_ph = sps.identity(n_ph, format="coo")
Id_qd = sps.identity(n_qd, format="coo")
I_qd = np.array([1, 1])
Id_one = sps.identity(1, format="coo")
Z_ph = sps.lil_matrix((n_ph, n_ph), dtype=float)
Z_qd = sps.lil_matrix((n_qd, n_qd), dtype=float)
Zr_ph = np.array([[0, 0, 0]])
Zr_qd = np.array([[0, 0]])
Zero = sps.lil_matrix((1, 1), dtype=float)
a_ph = np.array([[1, 0, 0]])
b_ph = np.array([[0, 1, 0]])
c_ph = np.array([[0, 0, 1]])
q = np.array([[1, 0]])
d = np.array([[0, 1]])
ab = np.array([[1, 0]])
bc = np.array([[0, 1]])
#####################################################


class GFLInverter3PhModel(GFLInverterModel):
    def __init__(self, inverter_obj: GFLInverter3Ph):
        super().__init__(inverter_obj)

        self.seq_mode = False
        self.n_ph = 3
        self.n_varqd = 2
        self.n_varll = 2

        # obtained from powerflow/optimal powerflow
        self.Pref = None
        self.Qref = None

        self.V_base = self.obj.V_base
        self.Pb = self.obj.Pb
        self.I_base = self.Pb / (np.sqrt(3) * self.V_base)
        self.Zb = self.V_base**2 / self.Pb

        self.Vqd_b = np.sqrt(2 / 3) * self.V_base
        self.Iqd_b = np.sqrt(2) * self.I_base
        self.w_b = self.wnom = const.w_nominal

        # Powerflow variables
        self.num_vars_real = 1  # w
        self.num_vars_complex = (
            3 * self.n_ph  # Vga, Vgb, Vgc
            + 3 * self.n_ph  # Iga, Igb, Igc
            + 4 * self.n_ph  # vg, vc, vcc, vin
            + 3 * self.n_ph  # ig, ic, iin
            + self.n_ph  # q
            + 2 * self.n_ph  # lamda_g, lamda_in (each 3-ph)
        )

        self.num_vars = self.num_vars_real + self.num_vars_complex

        self.var_offset_real = {"w": 0}

        self.var_offset_complex = {
            "V": 0,  # Vg, Vc, Vin
            "I": 3 * self.n_ph,  # Ig, Ic, Iin
            "v": 3 * self.n_ph + 3 * self.n_ph,  # vg, vc, vcc, vin
            "i": 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph,  # ig, ic, iin
            "q": 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph + 3 * self.n_ph,  # q
            "lamda": 3 * self.n_ph
            + 3 * self.n_ph
            + 4 * self.n_ph
            + 3 * self.n_ph
            + self.n_ph,  # lamda_g, lamda_in
        }

        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + 3 * self.n_ph,
            "v": 1 + 3 * self.n_ph + 3 * self.n_ph,
            "i": 1 + 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph,
            "q": 1 + 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph + 3 * self.n_ph,
            "lamda": 1
            + 3 * self.n_ph
            + 3 * self.n_ph
            + 4 * self.n_ph
            + 3 * self.n_ph
            + self.n_ph,
        }

        assert len(self.var_offset.keys()) == len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        )
        assert self.num_vars == self.var_offset["lamda"] + 2 * self.n_ph

        # to be written in detail with eqns in comments
        self.num_eqns_real = 0
        self.num_eqns_complex = 14 * self.n_ph + 3
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # Resistances, inductance and capacitance matrix associated with each branch
        # fmt:off
        self.r1_mat = np.array([
            [self.obj.raL1, 0,  0],
            [0, self.obj.rbL1, 0],
            [0, 0, self.obj.rcL1]
        ])
        self.r2_mat = np.array([
            [self.obj.raC, 0, 0],
            [0, self.obj.rbC, 0],
            [0, 0, self.obj.rcC]
        ])
        self.r3_mat = np.array([
            [self.obj.raL2, 0, 0],
            [0, self.obj.rbL2, 0],
            [0, 0, self.obj.rcL2]
        ])
        self.L1_mat = np.array([
            [self.obj.La1, 0 , 0],
            [0, self.obj.Lb1, 0],
            [0, 0, self.obj.Lc1]
        ]) 
        self.L2_mat = np.array([
            [self.obj.La2, 0 , 0],
            [0, self.obj.Lb2, 0],
            [0, 0, self.obj.Lc2]
        ])
        self.C_mat = np.array([
            [self.obj.Ca, 0, 0],
            [0, self.obj.Cb, 0],
            [0, 0, self.obj.Cc]
        ])
        # fmt:on

        self.init_dynamic_simulation()

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
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

    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # y_pf = [w, V, I, v, i , q , lamda]
        # V = [Vg, Vc, Vin]
        # I = [Ig, Ic, Iin]
        # v = [ vg, vc, vcc, vin]
        # i = [ig, ic, iin]
        # q = [q]
        # lamda = [lamda_g, lamda_in]
        # vin = [vin_a, vin_b, vin_c]  inverter leg voltage
        # din = [din_a, din_b, din_c]  duty cycle

        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_Vg_start = self.var_offset["V"]
        idx_Vg_end = idx_Vg_start + self.n_ph
        y_0[idx_Vg_start:idx_Vg_end, 0] = v_phasors

        idx_Vin_start = self.var_offset["V"] + 2 * self.n_ph
        idx_Vin_end = idx_Vin_start + self.n_ph
        y_0[idx_Vin_start:idx_Vin_end, 0] = v_phasors

        # idx_vin_start = self.var_offset["vin"]
        # idx_vin_end = idx_vin_start + self.n_ph
        # y_0[idx_vin_start:idx_vin_end] = v_phasors

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        return y_0

    def get_M_powerflow_inner(self) -> sps.coo_array:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
            and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros
        """

        # identity matrix
        Id_ph = sps.identity(self.n_ph, format="coo")

        # zero matrix
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)

        # voltages and currents for individual nodes in LCL filter
        # V = [Vg, Vc, Vin]
        Vg = sps.bmat([[Id_ph, Z_ph, Z_ph]])
        Vc = sps.bmat([[Z_ph, Id_ph, Z_ph]])
        Vin = sps.bmat([[Z_ph, Z_ph, Id_ph]])

        # I = [Ig, Ic, Iin]
        Ig = sps.bmat([[Id_ph, Z_ph, Z_ph]])
        Ic = sps.bmat([[Z_ph, Id_ph, Z_ph]])
        Iin = sps.bmat([[Z_ph, Z_ph, Id_ph]])

        # v = [ vg, vc, vcc, vin]
        vg = sps.bmat([[Id_ph, Z_ph, Z_ph, Z_ph]])
        vc = sps.bmat([[Z_ph, Id_ph, Z_ph, Z_ph]])
        vcc = sps.bmat([[Z_ph, Z_ph, Id_ph, Z_ph]])
        vin = sps.bmat([[Z_ph, Z_ph, Z_ph, Id_ph]])

        # i = [ig, ic, iin]
        ig = sps.bmat([[Id_ph, Z_ph, Z_ph]])
        ic = sps.bmat([[Z_ph, Id_ph, Z_ph]])
        iin = sps.bmat([[Z_ph, Z_ph, Id_ph]])
        ig_r = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]])
        ic_r = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0]])
        i3_r = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1]])
        V3_r = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1]])
        V1_r = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]])
        V2_r = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0]])

        # q = [q]
        q = Id_ph

        # lamda = [lamda1, lamda2]
        lamda_g = sps.bmat([[Id_ph, Z_ph]])
        lamda_in = sps.bmat([[Z_ph, Id_ph]])

        C = sps.bmat([[Z_ph, Z_ph, self.C_mat, Z_ph]])
        rg = sps.bmat([[self.r1_mat, Z_ph, Z_ph]])
        rc = sps.bmat([[Z_ph, self.r2_mat, Z_ph]])
        rin = sps.bmat([[Z_ph, Z_ph, self.r3_mat]])
        Lg = sps.bmat([[self.L1_mat, Z_ph, Z_ph]])
        Lin = sps.bmat([[Z_ph, Z_ph, self.L2_mat]])
        # fmt:on

        Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)
        Z_wr = sps.lil_matrix((1, 1), dtype=float)

        # coefficient matrices for eqns
        eq1_V = Vin - Vc
        eq6_i = ig + ic - iin
        eq7_v = -vc + vcc
        eq10_V = Vc - Vg

        # fmt:off
        M = sps.bmat(
            [
                # [w     V       I       v      i         q     lamda]
                [Z_w,    eq1_V,  None,   -vin,  None,     None, None],  # 1)Vin - Vc - vin = 0
                [Z_w,    None,   -Iin,   None,  iin,      None, None],  # 2)-Iin + iin
                [Z_w,    None,   None,   vin,   rin,      None, None],  # 3)vin + rin*iin + jw*lamda_in (fy)
                [Z_w,    None,   None,   None,  -Lin,     None, lamda_in],  # 4)lamda_in - Lin*iin =0
                [Z_w,    Vc,     None,   -vc,    None,    None, None],  # 5)Vc-vc=0
                [Z_w,    None,   -Ic,    None,   eq6_i,   None, None],  # 6)-Ic -iin+ic+ig =0
                [Z_w,    None,   None,   eq7_v,  rc,      None, None],  # 7)-vc + rc*ic + vcc
                [Z_w,    None,   None,   C,      None,    -q,   None],  # 8)Cvc - q = 0
                [Z_w,    None,   None,   None,   -ic,    None,  None],  # 9)-i2 + jw*q =0 (fy)
                [Z_w,    eq10_V, None,   -vg,    None,   None,  None],  # 10)V2-V1-v1 =0
                [Z_w,    None,   -Ig,    None,   -ig,    None,  None],  # 11)-I1-i1 =0
                [Z_w,    None,   None,    -vg,   rg,     None,  None],  # 12)-v1+r1*i1+jw*lamda1 =0 (fy)
                [Z_w,    None,   None,    None,  -Lg,    None,  lamda_g],  # 13)lamda1-L1*i1=0
                [Z_w,    -Vg,    None,    None,  None,   None,  None],  # 14)-V1 + [u] =0 (u)
                [Z_w,    None,   Ic,      None,  None,   None,  None],  # 18) Ic = 0
            ]
        )

        return M

        # fmt:on

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1) This function returns the non-linear terms of every equation.
        2)  For this model eqn 3, 9, 12 have a non-linear term jw*lamda, jw*q
        """

        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        # create an empty matrix for fy
        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)
        print(f"fy.shape : {fy.shape}")

        # idx of w in y vector
        idx_w = self.var_offset["w"]
        w = y[idx_w, 0]

        # fy update for eqn 3 : jw*lamda_in
        idx_lamda_in_start = self.var_offset["lamda"] + self.n_ph
        idx_lamda_in_end = idx_lamda_in_start + self.n_ph

        idx_eq3_start = (
            self.n_ph + self.n_ph
        )  # 1)Vin - Vc - vin = 0  # 2)-Iin + iin =  0
        idx_eq3_end = idx_eq3_start + self.n_ph

        fy[idx_eq3_start:idx_eq3_end] = 0 + 1j * (
            w * y[idx_lamda_in_start:idx_lamda_in_end]
        )

        # fy update for eqn 9 : jw*q
        idx_q_start = self.var_offset["q"]
        idx_q_end = idx_q_start + self.n_ph

        idx_eq9_start = 8 * self.n_ph
        idx_eq9_end = idx_eq9_start + self.n_ph

        fy[idx_eq9_start:idx_eq9_end] = 0 + 1j * (w * y[idx_q_start:idx_q_end])

        # fy update eqn 12 : jw*lamda_g
        idx_lamda_in_start = self.var_offset["lamda"]
        idx_lamda_in_end = idx_lamda_in_start + self.n_ph

        idx_eq12_start = 11 * self.n_ph
        idx_eq12_end = idx_eq12_start + self.n_ph

        fy[idx_eq12_start:idx_eq12_end] = 0 + 1j * (
            w * y[idx_lamda_in_start:idx_lamda_in_end]
        )

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

        # get col and row offsets of w, q, lamda1, lamda2
        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]

        q_re_start_offset = self.var_offset["q"]
        q_im_start_offset = self.var_offset_complex["q"]
        lamda_g_re_start_offset = self.var_offset["lamda"]
        lamda_g_im_start_offset = self.var_offset_complex["lamda"]
        lamda_in_re_start_offset = self.var_offset["lamda"] + self.n_ph
        lamda_in_im_start_offset = self.var_offset_complex["lamda"] + self.n_ph

        # eq3 real
        eq3_re_start_row = 2 * self.n_ph  # after eqn set 1, 2

        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset

            lamda_in_re_col_offset = lamda_in_re_start_offset + offset
            lamda_in_im_col_offset = lamda_in_im_start_offset + offset
            lamda_in_re = y_real[lamda_in_re_col_offset, 0]
            lamda_in_im = y_imag[lamda_in_im_col_offset, 0]

            # derivatrive wrt w
            pd_fy_split[row, w_col_offset] = -lamda_in_im

            # derivative wrt lamda_in_re =0

            # derivative wrt lamda_in_im
            pd_fy_split[row, self.num_vars + lamda_in_im_col_offset] = -w

        # eq3 imaginary part
        eq3_im_start_row = self.num_eqns + eq3_re_start_row

        for offset in range(self.n_ph):
            row = eq3_im_start_row + offset

            lamda_in_re_col_offset = lamda_in_re_start_offset + offset
            lamda_in_im_col_offset = lamda_in_im_start_offset + offset
            lamda_in_re = y_real[lamda_in_re_col_offset, 0]
            lamda_in_im = y_imag[lamda_in_im_col_offset, 0]

            # derivative wrt w
            pd_fy_split[row, w_col_offset] = lamda_in_re

            # derivative wrt lamda_in_re
            pd_fy_split[row, lamda_in_re_col_offset] = w

            # derivative wrt lamda2_im = 0

        # eq9 real part
        eq9_re_start_row = 8 * self.n_ph  # after 8 sets of eqns

        for offset in range(self.n_ph):
            row = eq9_re_start_row + offset

            q_re_col_offset = q_re_start_offset + offset
            q_im_col_offset = q_im_start_offset + offset
            q_re = y_real[q_re_col_offset, 0]
            q_im = y_imag[q_im_col_offset, 0]

            # derivative wrt w
            pd_fy_split[row, w_col_offset] = -q_im

            # derivative wrt q_re = 0

            # derivative wrt q_im
            pd_fy_split[row, q_im_col_offset] = -w

        # eq9 imaginary part
        eq9_im_start_row = self.num_eqns + eq9_re_start_row

        for offset in range(self.n_ph):
            row = eq9_im_start_row + offset

            q_re_col_offset = q_re_start_offset + offset
            q_im_col_offset = q_im_start_offset + offset
            q_re = y_real[q_re_col_offset, 0]
            q_im = y_imag[q_im_col_offset, 0]

            # derivative wrt w
            pd_fy_split[row, w_col_offset] = q_re

            # derivative wrt q_re
            pd_fy_split[row, q_re_col_offset] = w

            # derivative wrt q_im = 0

        # eq12 real part
        eq12_re_start_row = 11 * self.n_ph  # after 11 sets of eqns

        for offset in range(self.n_ph):
            row = eq12_re_start_row + offset

            lamda_g_re_col_offset = lamda_g_re_start_offset + offset
            lamda_g_im_col_offset = lamda_g_im_start_offset + offset
            lamda_g_re = y_real[lamda_g_re_col_offset, 0]
            lamda_g_im = y_imag[lamda_g_im_col_offset, 0]

            # derivatrive wrt w
            pd_fy_split[row, w_col_offset] = -lamda_g_im

            # derivative wrt lamda1_re =0

            # derivative wrt lamda1_im
            pd_fy_split[row, self.num_vars + lamda_g_im_col_offset] = -w

        # eq12 imaginary part
        eq12_im_start_row = self.num_eqns + eq12_re_start_row

        for offset in range(self.n_ph):
            row = eq12_im_start_row + offset

            lamda_g_re_col_offset = lamda_g_re_start_offset + offset
            lamda_g_im_col_offset = lamda_g_im_start_offset + offset
            lamda_g_re = y_real[lamda_g_re_col_offset, 0]
            lamda_g_im = y_imag[lamda_g_im_col_offset, 0]

            # derivative wrt w
            pd_fy_split[row, w_col_offset] = lamda_g_re

            # derivative wrt lamda1_re
            pd_fy_split[row, lamda_g_re_col_offset] = w

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_u_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1) This function returns the u vector for inverter model
        2) The u vector is the input to the inverter model
        3) The u vector is the voltage at the output of the inverter + o/p impedance i.e. nominal voltage
        """
        # create the empty u vector
        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=complex)

        # u vector for eqn 14
        idx_eq14_re_start = 13 * self.n_ph
        idx_eq14_re_end = idx_eq14_re_start + self.n_ph

        idx_eq14_im_start = self.num_eqns + idx_eq14_re_start
        idx_eq14_im_end = idx_eq14_im_start + self.n_ph

        V_phasors = utils.get_vector_phasors(self.nominal_voltage)
        V = np.array(list(V_phasors.values()))

        u[idx_eq14_re_start:idx_eq14_re_end] = V.real
        u[idx_eq14_im_start:idx_eq14_im_end] = V.imag

        return u[: self.num_eqns], u[self.num_eqns :]

    ###########################dynamic simulation functions#########################################
    def init_dynamic_simulation(self):
        var_offset_dynamic = {}

        self._lcl_filter()
        self._mnt()
        self._pll()
        self._current_control()
        self._svpwm()
        self._inverter_bridge()
        self._vbrf()
        self._ibrf()

        # create the combined var_offset dictionary

        # tmp:
        print(f"self.lcl_Mmat.shape: {self.lcl_Mmat.shape}")
        print(f"self.mnt_Mmat.shape: {self.mnt_Mmat.shape}")
        print(f"self.pll_Mmat.shape: {self.pll_Mmat.shape}")
        print(f"self.cc_Mmat.shape: {self.cc_Mmat.shape}")
        print(f"self.svpwm_Mmat.shape: {self.svpwm_Mmat.shape}")
        print(f"self.inv_br_Mmat.shape: {self.inv_br_Mmat.shape}")

        var_offset_dynamic = {}
        shift = 0
        for offsets, num_vars in [
            (self.lcl_var_offset, self.lcl_num_vars),
            (self.mnt_var_offset, self.mnt_num_vars),
            (self.pll_var_offset, self.pll_num_vars),
            (self.cc_var_offset, self.cc_num_vars),
            (self.svpwm_var_offset, self.svpwm_num_vars),
            (self.inv_br_var_offset, self.inv_br_num_vars),
            (self.var_offset_dynamic_Vbrf, self.Vbrf_num_vars),
            (self.var_offset_dynamic_Ibrf, self.Ibrf_num_vars),
        ]:
            for var in offsets:
                # sanity check for variable names
                assert var not in var_offset_dynamic
                # shift offset
                var_offset_dynamic[var] = offsets[var] + shift
            print(f"num_vars = {num_vars}")
            shift += num_vars

        assert len(var_offset_dynamic.keys()) == len(set(var_offset_dynamic.values()))

        # also store the final var offsets by block for better checking
        for offsets in [
            self.lcl_var_offset,
            self.mnt_var_offset,
            self.pll_var_offset,
            self.cc_var_offset,
            self.svpwm_var_offset,
            self.inv_br_var_offset,
            self.var_offset_dynamic_Vbrf,
            self.var_offset_dynamic_Ibrf,
        ]:
            for key in offsets:
                final_offset = var_offset_dynamic[key]
                offsets[key] = final_offset

        M_dynamic = sps.bmat(
            [
                [self.lcl_Mmat, None, None, None, None, None, None, None],
                [None, self.mnt_Mmat, None, None, None, None, None, None],
                [None, None, self.pll_Mmat, None, None, None, None, None],
                [None, None, None, self.cc_Mmat, None, None, None, None],
                [None, None, None, None, self.svpwm_Mmat, None, None, None],
                [None, None, None, None, None, self.inv_br_Mmat, None, None],
                [None, None, None, None, None, None, self.Vbrf_Mmat, None],
                [None, None, None, None, None, None, None, self.Ibrf_Mmat],
            ]
        )

        M_dynamic_shape_before_iface = M_dynamic.shape

        K_dynamic = sps.bmat(
            [
                [self.lcl_Kmat, None, None, None, None, None, None, None],
                [None, self.mnt_Kmat, None, None, None, None, None, None],
                [None, None, self.pll_Kmat, None, None, None, None, None],
                [None, None, None, self.cc_Kmat, None, None, None, None],
                [None, None, None, None, self.svpwm_Kmat, None, None, None],
                [None, None, None, None, None, self.inv_br_Kmat, None, None],
                [None, None, None, None, None, None, self.Vbrf_Kmat, None],
                [None, None, None, None, None, None, None, self.Ibrf_Kmat],
            ]
        )

        assert M_dynamic.shape[1] == (
            self.lcl_num_vars
            + self.mnt_num_vars
            + self.pll_num_vars
            + self.cc_num_vars
            + self.svpwm_num_vars
            + self.inv_br_num_vars
            + self.Vbrf_num_vars
            + self.Ibrf_num_vars
        )

        assert M_dynamic.shape == K_dynamic.shape

        def interface_vars(fr: str, to: str, count: int):
            nonlocal M_dynamic
            nonlocal K_dynamic
            new_rows = sps.lil_matrix((count, M_dynamic.shape[1]), dtype="float")
            fr_col_start = var_offset_dynamic[fr]
            fr_col_end = fr_col_start + count
            to_col_start = var_offset_dynamic[to]
            to_col_end = to_col_start + count
            new_rows[:, fr_col_start:fr_col_end] = -np.eye(count)
            new_rows[:, to_col_start:to_col_end] = np.eye(count)

            M_dynamic = sps.bmat([[M_dynamic], [new_rows]])

            new_rows = sps.lil_matrix((count, M_dynamic.shape[1]), dtype="float")
            K_dynamic = sps.bmat([[K_dynamic], [new_rows]])

        interface_list = [
            # lcl <-> inverter
            ("Vab", "Eab", 1),
            ("Vbc", "Ebc", 1),
            ("i_abc", "iinf", self.n_ph),
            # mnt <-> lcl
            ("Vg", "V", self.n_ph),
            ("ig", "igf", self.n_ph),
            # # pll <-> mnt
            # ("Vg_pll_qd", "Vg_qd", self.n_varqd),
            # pll < -> Vbrf
            ("Vg_pll_qd", "V_qd_dc", self.n_varqd),
            # # cc <-> mnt
            # ("Vg_cc_qd", "Vg_qd", self.n_varqd),
            # ("ig_cc_qd", "ig_qd", self.n_varqd),
            # c <-> vbrf, ibrf
            ("Vg_cc_qd", "V_qd_dc", self.n_varqd),
            ("ig_cc_qd", "I_qd_dc", self.n_varqd),
            # svpwm <-> cc
            ("v_qd_ref", "e_qd_fin", self.n_varqd),
            # inverter <-> svpwm
            ("din", "m_abc", self.n_ph),
            # Vbrf <-> mnt
            ("V_qd", "Vg_qd", self.n_varqd),
            # ibrf <-> mnt
            ("I_qd", "ig_qd", self.n_varqd),
        ]

        for fr, to, count in interface_list:
            interface_vars(fr, to, count)

        # self.M_dynamic_interface = M_dynamic_interface
        self.M_dynamic_iface_nrows = (
            M_dynamic.shape[0] - M_dynamic_shape_before_iface[0]
        )
        self.M_dynamic_iface_ncols = M_dynamic_shape_before_iface[1]

        # finally set the values in the object
        self.var_offset_dynamic = var_offset_dynamic
        self.M_dynamic = M_dynamic
        self.K_dynamic = K_dynamic
        self.num_vars_dynamic = M_dynamic.shape[1]
        self.num_eqns_dynamic = M_dynamic.shape[0]

        print(f"self.M_dynamic.shape: {self.M_dynamic.shape}")

    def _lcl_filter(self):
        # fmt:off
        b1_var_offset = {
            "V": 0,  # [Vgf_a, Vgf_b, Vgf_c] grid side node voltage
            "Vcf": self.n_ph,  # [Vcf_a, Vcf_b, Vcf_c] capacitor node voltage
            "Vinf": 2 * self.n_ph,  # [Vinf_a, Vinf_b, Vin_c] inverter side node voltage
            "Vab" : 3 * self.n_ph,
            "Vbc" : 1 + 3 * self.n_ph,
            "I": 2 + 3 * self.n_ph,  # [Igf_a, Igf_b, Igf_c] grid side node injection
            "Iinf": 2 + 4 * self.n_ph,  # [Vinf_a, Vinf_b, Vinf_c] inverter side node injection
            "vgf": 2 + 5 * self.n_ph,  # [vgf_a, vgf_b, vgf_c] grid side branch voltage
            "vcf": 2 + 6 * self.n_ph,  # [vcf_a, vcf_b, vcf_c] capacitor branch voltage
            "vinf": 2 + 7 * self.n_ph,  # [vinf_a, vinf_b, vinf_c] inverter side branch voltage
            "vccf": 2 + 8 * self.n_ph,  # [vccf_a, vccf_b, vccf_c] voltage across capcitor
            "igf": 2 + 9 * self.n_ph,  # [igf_a, igf_b, igf_c] current in grid side inductor
            "icf": 2 + 10 * self.n_ph,  # [icf_a, icf_b, icf_c] current into the capacitor branch
            "iinf": 2 + 11 * self.n_ph,  # [iinf_a, iinf_b, iinf_c] current into inverter side inductor
            "qf": 2 + 12 * self.n_ph,  # [qf_a, qf_b, qf_c] capacitor charge
            "lamda_g": 2 + 13 * self.n_ph,  # [lamda_ga, lamda_gb, lamda_gc] grid side flux linkage
            "lamda_in": 2 + 14 * self.n_ph,  # [lamda_ina, lamda_inb, lamda_inc] inverter side flux linakge
        }
        # fmt:on
        b1_num_vars = b1_var_offset["lamda_in"] + 3

        # coefficient of eqns for capacitor
        C = self.C_mat
        rg = self.r1_mat
        rc = self.r2_mat
        r_in = self.r3_mat
        Lg = self.L1_mat
        L_in = self.L2_mat
        eq14 = np.array([[1, 1, 1]])
        Vab = np.array([[1, -1, 0]])
        Vbc = np.array([[0, 1, -1]])

        # V1 - Vgf = 0
        # V2 - Vcf = 0
        # V3 - Vinf = 0
        # I3 - Iinf = 0
        # fmt: off
        b1_Mmat = sps.bmat(
            [
                # Vgf,    Vcf,    Vinf,  Vab,   Vbc,  Igf,    Iinf,   vgf,    vcf,    vinf,   vccf,   igf,    icf,    iinf,   qf,     lamda_g, lamda_in
                [-Id_ph,  Id_ph,  None,  None,  None,  None,   None,  -Id_ph,  None,   None,   None,   None,   None,   None,   None,   None,    None,],  # 1)Vcf-Vgf-vgf=0 (3)
                [ None,   None,   None,  None,  None, -Id_ph,  None,   None,   None,   None,   None,   -Id_ph,  None,   None,   None,   None,    None,],  # 2)-Igf - igf=0 (3)
                [ None,   None,   None,  None,  None,  None,   None,  -Id_ph,  None,   None,   None,   rg,     None,   None,   None,   None,    None,],  # 3)-vgf + rg*igf + d(lamda_g)/dt=0 (K) (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   None,   None,   None,  -Lg,     None,   None,   None,   Id_ph,  None,],  # 4)lamda_g - Lg*igf=0 (3)
                [ None,   Id_ph,  None,  None,  None,  None,   None,   None,   -Id_ph, None,   None,   None,   None,   None,   None,   None,    None,],  # 5)Vcf-vcf=0 (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   None,   None,   None,   Id_ph,  Id_ph,  -Id_ph, None,   None,    None,],  # 6)-iinf+icf+igf=0 (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   -Id_ph, None,   Id_ph,  None,   rc,     None,   None,   None,    None,],  # 7)-vcf+rc*icf+vccf=0 (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   None,   None,   C,      None,   None,   None,   -Id_ph, None,    None,],  # 8)C*vcc-q=0 (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   None,   None,   None,   None,   -Id_ph, None,   None,   None,    None,],  # 9)-icf + d(qf)/dt=0 (3)
                [ None,  -Id_ph,  Id_ph, None,  None,  None,   None,   None,   None,  -Id_ph,  None,   None,   None,   None,   None,   None,    None,],  # 10)Vinf-Vcf-vinf=0 (3)
                [ None,   None,   None,  None,  None,  None,  -Id_ph,  None,   None,   None,   None,   None,   None,   Id_ph,  None,   None,    None,],  # 11)-Inf+iinf=0 (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   None,  -Id_ph,  None,   None,   None,   r_in,   None,   None,    None,],  # 12)-vinf+r_in*iinf + d(lamda_in)/dt=0 (K) (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   None,   None,   None,   None,   None,  -L_in,   None,   None,    Id_ph,], # 13)lamda_inf -L_in*iinf=0 (3)
                [ None,   None,   None,  None,  None,  None,   None,   None,   None,   None,   None,   None,   eq14,   None,   None,   None,    None, ], # 14) icf_a+icf_b+icf_c=0 (1)
                [ None,  None,    -Vab,   Id_one,None,  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,    None, ],  # 15) -Vab + Va - Vb = 0 (1)
                [ None,  None,    -Vbc,   None,  Id_one,None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,    None, ],  # 16) -Vbc + Vb - Vc = 0 (1)
            ]
        )
        # fmt: on

        # fmt: off
        b1_Kmat = sps.bmat([
            #Vgf,    Vcf,    Vinf, Vab,     Vbc,   Igf,    Iinf,   vgf,    vcf,    vinf,   vccf,   igf,    icf,    iinf,   qf,     lamda_g, lamda_in
            [Z_ph,   Z_ph,  None,  None,    None,   None,   None,   Z_ph,   None,   None,   None,   None,   None,   None,   None,   None,    None,  ], #1)Vcf-Vgf-vgf=0
            [None,   None,  None,  None,    None,   Z_ph,   None,   None,   None,   None,   None,   Z_ph,   None,   None,   None,   None,    None,  ], #2)-Igf-igf=0
            [None,   None,  None,  None,    None,   None,   None,   Z_ph,   None,   None,   None,   Z_ph,   None,   None,   None,   Id_ph,   None,  ], #3)-vgf + rg*igf + d(lamda_g)/dt=0 (K)
            [None,   None,  None,  None,    None,   None,   None,   None,   None,   None,   None,   Z_ph,   None,   None,   None,   Z_ph,    None,  ], #4)lamda_g - Lg*igf=0      
            [None,   Z_ph,  None,  None,    None,   None,   None,   None,   Z_ph,   None,   None,   None,   None,   None,   None,   None,    None,  ], #5)Vcf-vcf=0
            [None,   None,  None,  None,    None,   None,   None,   None,   None,   None,   None,   Z_ph,   Z_ph,   Z_ph,   None,   None,    None,  ], #6)-Icf-iinf+icf+igf=0
            [None,   None,  None,  None,    None,   None,   None,   None,   Z_ph,   None,   Z_ph,   None,   Z_ph,   None,   None,   None,    None,  ], #7)-vcf+rc*icf+vcc=0
            [None,   None,  None,  None,    None,   None,   None,   None,   None,   None,   Z_ph,   None,   None,   None,   Z_ph,   None,    None,  ], #8)C*vcc-q=0
            [None,   None,  None,  None,    None,   None,   None,   None,   None,   None,   None,   None,   Z_ph,   None,   Id_ph,  None,    None,  ], #9)-icf + d(qf)/dt=0
            [None,   Z_ph,  Z_ph,  None,    None,   None,   None,   None,   None,   Z_ph,   None,   None,   None,   None,   None,   None,    None,  ], #10)Vinf-Vcf-vinf=0
            [None,   None,  None,  None,    None,   None,   Z_ph,   None,   None,   None,   None,   None,   None,   Z_ph,   None,   None,    None,  ], #11)-Inf-iinf=0
            [None,   None,  None,  None,    None,   None,   None,   None,   None,   Z_ph,   None,   None,   None,   Z_ph,   None,   None,    Id_ph, ], #12)-vinf+r_in*iinf + d(lamda_in)/dt=0 (K)
            [None,   None,  None,  None,    None,   None,   None,   None,   None,   None,   None,   None,   None,   Z_ph,   None,   None,    Z_ph,  ], #13)lamda_inf -L_in*iinf=0
            [None,   None,  None,  None,    None,   None,   None,   None,   None,   None,   None,   None,   Zr_ph,  None,   None,   None,    None,  ], #14) icf_a+icf_b+icf_c=0
            [ None,  None,  Zr_ph, Zero,    None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,    None, ],  # 15) -Vab + Va - Vb = 0 (1)
            [ None,  None,  Zr_ph, None,    Zero,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,    None, ],  # 16) -Vbc + Vb - Vc = 0 (1)
        ]) 

        # fmt:on
        assert b1_Kmat.shape == b1_Mmat.shape

        assert b1_Mmat.shape[0] == 42
        assert b1_Mmat.shape[1] == b1_num_vars, (
            f"b1_Mmat.shape[1]={b1_Mmat.shape[1]}, b1_num_vars={b1_num_vars}"
        )

        def _get_fy_dynamic_b1(t, y) -> np.ndarray:
            fy1 = np.zeros((b1_Mmat.shape[0], 1), dtype=float)
            return fy1

        def _get_u_dynamic_b1(t: float, y) -> np.ndarray:
            # block1: measurement and tranformation block
            # no entry in u1 for b1 block
            u1 = np.zeros((b1_Mmat.shape[0], 1), dtype=float)

            return u1

        assert not hasattr(self, "lcl_Mmat") or self.lcl_mat is None, (
            f"self.lcl_Mmat should not be at this point [Please Investigate]"
        )

        self.lcl_Mmat = b1_Mmat
        self.lcl_Kmat = b1_Kmat
        self.lcl_var_offset = b1_var_offset
        self.lcl_num_vars = b1_num_vars
        self.get_lcl_fy_dynamic = _get_fy_dynamic_b1
        self.get_lcl_u_dynamic = _get_u_dynamic_b1

    # Measurement and transformation block funtion
    def _mnt(self):
        # b2. Measurement and Transformation block
        b2_var_offset = {
            # Vg, Vg_ll, Vg_qd, ig, ig_qd
            "Vg": 0,  # [Vg_a, Vg_b, Vg_c]
            "Vg_ll": self.n_ph,  # [Vg_ab, Vg_bc]
            "Vg_qd": self.n_ph + self.n_varll,  # [Vg_d, Vg_q]
            "ig": self.n_ph + self.n_varll + self.n_varqd,  # [ig_a, ig_b, ig_c]
            "ig_qd": self.n_ph + self.n_varll + self.n_varqd + self.n_ph,
        }
        b2_num_vars = b2_var_offset["ig_qd"] + 2

        # coefficients for eqns

        eq1_Vg = np.array([[1, -1, 0]])
        eq2_Vg = np.array([[0, 1, -1]])

        # Measurement and Transformation Block
        # fmt:off
        b2_Mmat = sps.bmat([
            #[Vg,     Vg_ll,     Vg_qd,  ig,     ig_qd]
            [eq1_Vg, -ab,        None,   Zr_ph,  None], # 1)Vgll_ab-(Vga-Vgb)=0 (1)
            [eq2_Vg, -bc,        None,   None,   None], # 2)Vgll_bc-(Vgb-Vgc)=0 (1)
            [None,    None,      q,      None,   None], # 3)Vgq-K*Vg_ll=0 (fy) (1)
            [None,    None,      d,      None,   None], # 4)Vgd-K*Vg_ll=0 (fy) (1)
            [None,    None,      None,   None,      q], # 5)ig_q-K*ig=0 (fy) (1)
            [None,    None,      None,   None,      d], # 6)ig_d-K*ig=0 (fy) (1) 

        ])

        assert b2_Mmat.shape[0] == 6
        assert b2_Mmat.shape[1] == b2_num_vars, f"b2_Mmat.shape[1]={b2_Mmat.shape[1]}, b2_num_vars={b2_num_vars}"

        # no differential equation in this block
        b2_Kmat = sps.lil_matrix((b2_Mmat.shape), dtype = float)

        def _get_fy_dynamic_b2(t, y):

            #### fy2: Measurement and transformation block
        # non-linear expressions in measurement and transformation block b1
        # eqn 3, 4, 5, 6

            fy2 = np.zeros((b2_Mmat.shape[0], 1), dtype=float)

            # eq3: (-2/3)*(V_ab*cos(delta)-V_bc*cos(delta-2*pi/3))
            idx_eq3_start = self.n_varll

            idx_delta = self.var_offset_dynamic["delta"]
            idx_V_ab = self.var_offset_dynamic["Vg_ll"]
            idx_V_bc = idx_V_ab + 1
            idx_ig_a = self.var_offset_dynamic["ig"]  # a phase
            idx_ig_b = self.var_offset_dynamic["ig"] + 1  # bphase
            idx_ig_c = self.var_offset_dynamic["ig"] + 1  # c phase

            delta = y[idx_delta]
            V_ab = y[idx_V_ab]
            V_bc = y[idx_V_bc]
            ig_a = y[idx_ig_a]
            ig_b = y[idx_ig_b]
            ig_c = y[idx_ig_c]

            fy2[idx_eq3_start] = (-2 / 3) * (
                V_ab * np.cos(delta) - V_bc * np.cos(delta + 2 * np.pi / 3)
            )/self.Vqd_b

            # eq4:(-2/3)*(V_ab*sin(delta)-V_bc*sin(delta-2*pi/3))
            idx_eq4_start = idx_eq3_start + 1
            fy2[idx_eq4_start] = (-2 / 3) * (
                V_ab * np.sin(delta) - V_bc * np.sin(delta + 2 * np.pi / 3)
            )/self.Vqd_b

            # eq5:ig_q - K*ig  = 0
            idx_eq5_start = self.n_varll + self.n_varqd
            fy2[idx_eq5_start] = (-2 / 3) * (
                ig_a * np.cos(delta)
                + ig_b * np.cos(delta - 2 * np.pi / 3)
                + ig_c * np.cos(delta + 2 * np.pi / 3)
            )/self.Iqd_b

            # eq6:ig_d - K*ig = 0
            idx_eq6_start = idx_eq5_start + 1
            fy2[idx_eq6_start] = (-2 / 3) * (
                ig_a * np.sin(delta)
                + ig_b * np.sin(delta - 2 * np.pi / 3)
                + ig_c * np.sin(delta + 2 * np.pi / 3)
            )/self.Iqd_b

            return fy2

        def _get_u_dynamic_b2(t: float, y):
            # block1: measurement and tranformation block
            # no entry in u1 for b1 block
            u2 = np.zeros((b2_Mmat.shape[0], 1), dtype=float)

            return u2

        self.mnt_Mmat = b2_Mmat
        self.mnt_Kmat = b2_Kmat
        self.mnt_var_offset = b2_var_offset
        self.mnt_num_vars = b2_num_vars
        self.get_mnt_fy_dynamic = _get_fy_dynamic_b2
        self.get_mnt_u_dynamic = _get_u_dynamic_b2

    # Phase locked loop
    def _pll(self):
        # b3. Phase Locked loop block
        b3_var_offset = {
            # w, Vg_pll_qd, Vg_d_p, Vg_d_int, delta
            "w": 0,  # angular frequency of the inverter
            "Vg_pll_qd": 1,  # Vg_d obtained from block 1 as an input to PLL
            "Vg_d_p": 1 + self.n_varqd,  # d-axis Vg after proportional gain
            "Vg_d_int": 2 + self.n_varqd,  # accumulated error
            "delta": 3 + self.n_varqd,  # PLL voltage phase angle
        }
        b3_num_vars = b3_var_offset["delta"] + 1
        kp = self.obj.Kppll
        ki = self.obj.Kipll
        wb = self.w_b * Id_one

        # coefficients for eqns

        # fmt: off
        b3_Mmat = sps.bmat(
            [
                # [w,       Vg_pll_qd,  Vg_d_p, Vg_d_int,   delta]
                [  Zero,    kp * d,     Id_one, Zero,       Zero ],  # 1) Vg_d_p + kp*Vg_pll_d = 0 (1)
                [  None,    ki * d,     None,   None,       None ],  # 2) d(Vg_d_int)/dt + ki*Vg_pll_d = 0 (K mat) (1)
                [ -Id_one,  None,       Id_one, Id_one,     None ],  # 3) Vg_d_p + Vg_d_int - w = 0 (1)
                [ -wb,      None,       None,   None,       None ],  # 4) d(delta)/dt - w * wb = 0 (K mat) (1)
            ]
        )
        # fmt: on

        assert b3_Mmat.shape[0] == 4
        assert b3_Mmat.shape[1] == b3_num_vars

        # fmt: off
        b3_Kmat = sps.bmat(
            [
                # [w,   Vg_pll_qd,  Vg_d_p, Vg_d_int, delta]
                [ None, Zr_qd,      Zero,   None,     None  ],  # 1) Vg_d_p + kp*Vg_pll_d = 0
                [ None, Zr_qd,      None,   Id_one,   None  ],  # 2) d(Vg_d_int)/dt + ki*Vg_pll_d = 0 (K mat)
                [ Zero, None,       Zero,   Zero,     None  ],  # 3) Vg_d_p + Vg_d_int - w = 0
                [ Zero, None,       None,   None,     Id_one],  # 4) d(delta)/dt - w*wb = 0 (K mat)
            ]
        )
        # fmt: on

        assert b3_Kmat.shape == b3_Mmat.shape

        def _get_fy_dynamic_b3(t, y) -> np.ndarray:
            fy3 = np.zeros((b3_Mmat.shape[0], 1), dtype=float)
            return fy3

        def _get_u_dynamic_b3(t, y) -> np.ndarray:
            # no entry in u3 vector for b3 block
            u3 = np.zeros((b3_Mmat.shape[0], 1), dtype=float)
            return u3

        self.pll_Mmat = b3_Mmat
        self.pll_Kmat = b3_Kmat
        self.pll_var_offset = b3_var_offset
        self.pll_num_vars = b3_num_vars
        self.get_pll_fy_dynamic = _get_fy_dynamic_b3
        self.get_pll_u_dynamic = _get_u_dynamic_b3

    # b4. Current control block
    def _current_control(self):
        b4_var_offset = {
            # [Vg_cc_qd, iqd_ref, ig_cc_qd, iqd_diff, prop_cs_qd, int_cs_qd, pi_co_qd, e_qd, e_qd_fin]
            "Vg_cc_qd": 0,  # line to line grid voltage transformed to qd obtained from M&T block b1
            "iqd_ref": self.n_varqd,  # reference inverter output calculated from ref P and Q
            "ig_cc_qd": 2 * self.n_varqd,  # actual inverter output current
            "iqd_diff": 3 * self.n_varqd,  # difference between actuial and reference
            "prop_cs_qd": 4 * self.n_varqd,  # proportional controller signal
            "int_cs_qd": 5 * self.n_varqd,  # integral control signal
            "pi_co_qd": 6 * self.n_varqd,  # PI controller signal
            "e_qd": 7 * self.n_varqd,  # inverter terrminal voltage
            "e_qd_fin": 8
            * self.n_varqd,  # final inverter terminal voltage considering saturation
        }
        b4_num_vars = b4_var_offset["e_qd_fin"] + 2

        # coeffiencients for eqns
        iq_ref = np.array([[1, 0]])
        id_ref = np.array([[0, 1]])
        Kpc = self.obj.Kpc
        Kic = self.obj.Kic
        Vdc = self.obj.Vdc

        # fmt:offf
        b4_Mmat = sps.bmat(
            [
                # [Vg_cc_qd,  iqd_ref,    ig_cc_qd, iqd_diff,  prop_cs_qd, int_cs_qd, pi_co_qd, e_qd,  e_qd_fin,]
                [
                    Zr_qd,
                    iq_ref,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],  # 1)iq_ref - Pref/Vg_cc_q = 0 (fy) (1)
                [
                    None,
                    id_ref,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],  # 2)id_ref - Qref/Vg_cc_d = 0 (fy) (1)
                [
                    None,
                    -Id_qd,
                    Id_qd,
                    Id_qd,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],  # 3)iqd_diff-(iqd_ref - ig_cc_qd) = 0 (2)
                [
                    None,
                    None,
                    None,
                    -Kpc * Id_qd,
                    Id_qd,
                    None,
                    None,
                    None,
                    None,
                ],  # 4)prop_cs_qd - Kpc*iqd_diff = 0 (2)
                [
                    None,
                    None,
                    None,
                    -Kic * Id_qd,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],  # 5)d(in_cs_qd)/dt - Kic*iqd_diff = 0 (K) (2)
                [
                    None,
                    None,
                    None,
                    None,
                    -Id_qd,
                    -Id_qd,
                    Id_qd,
                    None,
                    None,
                ],  # 6)pi_co_qd - prop_cs_qd - int_cs_qd = 0 (2)
                [
                    -Id_qd,
                    None,
                    None,
                    None,
                    None,
                    None,
                    -Id_qd,
                    Id_qd,
                    None,
                ],  # 7)e_qd - vg_cc_qd - pi_co_qd + idq*XL = 0 (fy) (2)
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Id_qd,
                ],  # 8)e_qd_fin - C(e_qd, Vmax) = 0 (fy) (2)
            ]
        )

        assert b4_Mmat.shape[0] == 14
        assert b4_Mmat.shape[1] == b4_num_vars

        #### K matrix for block 4: Current Controller
        # fmt:off
        b4_Kmat = sps.bmat([
            #[Vg_cc_qd,  iqd_ref,    ig_cc_qd, iqd_diff, prop_cs_qd, int_cs_qd, pi_co_qd, e_qd,     e_qd_fin, ]
              [Zr_qd,    Zr_qd,      Zr_qd,    None,     None,       None,      None,     None,     None,     ], #1)iq_ref - Pref/Vg_cc_q = 0 (fy) (1)
              [None,     Zr_qd,      None,     None,     None,       None,      None,     None,     None,     ], #2)id_ref - Qref/Vg_cc_d = 0 (fy) (1)
              [None,     Z_qd,       Z_qd,     Z_qd,     None,       None,      None,     None,     None,     ], #3)iqd_diff-(iqd_ref - ig_cc_qd) = 0 (2)
              [None,     None,       None,     Z_qd,     Z_qd,       None,      None,     None,     None,     ], #4)prop_cs_qd - Kpc*iqd_diff = 0 (2)
              [None,     None,       None,     Z_qd,     None,       Id_qd,     None,     None,     None,     ], #5)d(int_cs_qd)/dt - Kic*iqd_diff = 0 (K) (2)
              [None,     None,       None,     None,     Z_qd,       Z_qd,      Z_qd,     None,     None,     ], #6)pi_co_qd - prop_cs_qd - int_cs_qd = 0 (2)
              [Z_qd,     None,       None,     None,     None,       None,      Z_qd,     Z_qd,     None,     ], #7)e_qd - vg_cc_qd - pi_co_qd + idq*XL = 0 (fy) (2)
              [None,     None,       None,     None,     None,       None,      None,     None,     Z_qd,     ], #8)e_qd_fin - C(e_qd, Vmax) = 0 (fy) (2)             
              
        ])
        # fmt:on

        assert b4_Kmat.shape == b4_Mmat.shape, (
            f"b4_Kmat shape = {b4_Kmat.shape}, b4_Mmat shape = {self.b4_Mmat.shape}"
        )

        def _get_fy_dynamic_b4(t, y) -> np.ndarray:
            #### fy4: Current control blcok
            # non-linear expressions in current control block eqns
            # eqn 1, 2, 7, 8, 9
            fy4 = np.zeros((b4_Mmat.shape[0], 1), dtype=float)

            idx_Vg_cc_q_start = self.var_offset_dynamic["Vg_cc_qd"]
            idx_Vg_cc_d_start = self.var_offset_dynamic["Vg_cc_qd"] + 1
            idx_w = self.var_offset_dynamic[
                "w"
            ]  # this needs to be corrected top be grid frequency wg
            idx_ig_cc_q = self.var_offset_dynamic["ig_cc_qd"]
            idx_ig_cc_d = self.var_offset_dynamic["ig_cc_qd"] + 1
            idx_e_q = self.var_offset_dynamic["e_qd"]
            idx_e_d = self.var_offset_dynamic["e_qd"] + 1
            idx_e_q_fin = self.var_offset_dynamic["e_qd_fin"]
            idx_e_d_fin = self.var_offset_dynamic["e_qd_fin"] + 1

            Vg_cc_q = y[idx_Vg_cc_q_start]
            Vg_cc_d = y[idx_Vg_cc_d_start]
            ig_cc_q = y[idx_ig_cc_q]
            ig_cc_d = y[idx_ig_cc_d]
            w = y[idx_w]
            e_q = y[idx_e_q]
            e_d = y[idx_e_d]
            e_mag = np.sqrt(e_q**2 + e_d**2)
            Vmax = self.obj.Vdc / np.sqrt(3) / self.Vqd_b

            # eq1:iq_ref - Pref/Vg_cc_q = 0
            idx_eq1_start = 0
            fy4[idx_eq1_start] = -(2 / 3) * ((self.Pref_total) / self.Pb) / Vg_cc_q
            # fy4[idx_eq1_start] = ((-5e5) / self.Pb) / Vg_cc_q

            # eq2:id_ref - Qref/Vg_cc_d = 0
            idx_eq2_start = idx_eq1_start + 1
            print(f"Vg_cc_q: {Vg_cc_q}")
            # fy4[idx_eq2_start] = 0  # in balanced this will cause division by 0 so set to 0
            # if abs(Vg_cc_d) < 1e-5:
            #     fy4[idx_eq2_start] = 0
            # else:
            #     fy4[idx_eq2_start] = ((-self.Qref_total)/self.Pb )/ Vg_cc_d
            fy4[idx_eq2_start] = (2 / 3) * ((self.Qref_total) / self.Pb) / Vg_cc_q

            # eq7:e_qd - vg_cc_qd - pi_co_qd + idq*XL = 0
            # q_axis:
            idx_eq7q_start = 5 * self.n_varqd
            idx_eq7d_start = 5 * self.n_varqd + 1

            fy4[idx_eq7q_start] = (
                -ig_cc_d
                * w
                * self.w_b
                * (self.L1_mat[0][0] + self.L2_mat[0][0])
                / self.Zb
            )
            fy4[idx_eq7d_start] = (
                ig_cc_q
                * w
                * self.w_b
                * (self.L1_mat[0][0] + self.L2_mat[0][0])
                / self.Zb
            )

            # eq8:e_qd_fin - C(e_qd, Vmax) = 0
            idx_eq8_q_start = 6 * self.n_varqd
            idx_eq8_d_start = 6 * self.n_varqd + 1
            if e_mag > Vmax:
                fy4[idx_eq8_q_start] = -(Vmax * e_q) / e_mag
                fy4[idx_eq8_d_start] = -(Vmax * e_d) / e_mag
            else:
                fy4[idx_eq8_q_start] = -e_q
                fy4[idx_eq8_d_start] = -e_d

            return fy4

        def _get_u_dynamic_b4(t, y) -> np.ndarray:
            u4 = np.zeros((b4_Mmat.shape[0], 1), dtype=float)
            return u4

        self.cc_Mmat = b4_Mmat
        self.cc_Kmat = b4_Kmat
        self.cc_var_offset = b4_var_offset
        self.cc_num_vars = b4_num_vars
        self.get_cc_fy_dynamic = _get_fy_dynamic_b4
        self.get_cc_u_dynamic = _get_u_dynamic_b4

    # svpwm block
    def _svpwm(self):
        # b6. SVPWM
        b5_var_offset = {
            "m_qd": 0,
            "m_qd_f": self.n_varqd,
            "v_qd_ref": 2 * self.n_varqd,
            "m_abc": 3 * self.n_varqd,
        }
        b5_num_vars = b5_var_offset["m_abc"] + self.n_ph

        # coefficients for eqns of SVPWM
        inv_Vdc_q = (1 / self.obj.Vdc) * self.Vqd_b * q
        inv_Vdc_d = (1 / self.obj.Vdc) * self.Vqd_b * d

        # fmt: off
        b5_Mmat = sps.bmat([
             #m_qd, m_qd_f, v_qd_ref,   m_abc
            [q,     None,   -inv_Vdc_q, None], #1) mq - vq/Vdc = 0 (1)
            [d,     None,   -inv_Vdc_d, None], #2) md - Vd/Vdc = 0 (1)
            [None,  q,      None,       None], #3) mq_f - C(mq, mx) = 0 (fy) (1)
            [None,  d,      None,       None], #4) md_f - C(md, mx) = 0 (fy) (1)
            [None,  None,   None,       a_ph], #5) m_a - Kinv(m_qd_f) (1)
            [None,  None,   None,       b_ph], #6) m_b - Kinv(m_qd_f) (1)
            [None,  None,   None,       c_ph], #7) m_c - Kinv(m_qd_f) (1)          

        ])
        # fmt: on

        assert b5_Mmat.shape[0] == 7
        assert b5_Mmat.shape[1] == 9
        assert b5_Mmat.shape[1] == b5_num_vars

        # fmt:off
        b5_Kmat = sps.bmat([
            # m_qd, m_qd_f, v_qd_ref, m_abc
            [Zr_qd,  None,   None,     None], #1) mq - vq/Vdc = 0 (1)
            [None,  Zr_qd,   None,     None], #2) md - Vd/Vdc = 0 (1)
            [None,  None,   Zr_qd,     None], #3) mq_f - C(mq, mx) = 0 (fy) (1)
            [None,  None,   None,     Zr_ph],#4) md_f - C(md, mx) = 0 (fy) (1)
            [Zr_qd,  None,   None,     None], #5) m_a - Kinv(m_qd_f) (fy) (1)
            [None,  Zr_qd,   None,     None], #6) m_b - Kinv(m_qd_f) (fy) (1)
            [None,  None,   Zr_qd,     None], #7) m_c - Kinv(m_qd_f) (fy) (1) 

        ])
        # fmt:on

        assert b5_Kmat.shape == b5_Mmat.shape, f"{b5_Kmat.shape} == {b5_Mmat.shape}"

        def _get_fy_dynamic_b5(t, y) -> np.ndarray:
            # fy6: SVPWM block
            # non-linearities in eqn 1, 2, 3, 4 and 5
            fy6 = np.zeros((b5_Mmat.shape[0], 1), dtype=float)
            idx_mq_f = b5_var_offset["m_qd_f"]
            idx_md_f = idx_mq_f + 1

            mq_f = y[idx_mq_f]
            md_f = y[idx_md_f]

            theta = y[self.var_offset_dynamic["delta"]]

            mq = y[self.var_offset_dynamic["m_qd"]]
            md = y[self.var_offset_dynamic["m_qd"] + 1]

            m_mag = np.sqrt(mq**2 + md**2)

            m_max = 1 / np.sqrt(3)

            if m_mag > m_max:  # compute Vmax in adapter and parse in the model
                print(f"m_mag = {m_mag}")
                # input("continue")

                fy6[2] = -(m_max * mq) / m_mag
                fy6[3] = -(m_max * md) / m_mag
                # input("continue?")
            else:
                print(f"mq = {mq} md = {md}")
                fy6[2] = -(mq)
                fy6[3] = -(md)

            # eqn 5, 6, 7: m_abc - (Kinv*m_qd_f) = 0
            idx_eqn5 = 4
            fy6[idx_eqn5] = -(mq_f * np.cos(theta) + md_f * np.sin(theta))

            idx_eqn6 = 5
            fy6[idx_eqn6] = -(
                mq_f * np.cos(theta - 2 * np.pi / 3)
                + md_f * np.sin(theta - 2 * np.pi / 3)
            )

            idx_eqn7 = 6
            fy6[idx_eqn7] = -(
                mq_f * np.cos(theta + 2 * np.pi / 3)
                + md_f * np.sin(theta + 2 * np.pi / 3)
            )

            return fy6

        def _get_u_dynamic_b5(t, y) -> np.ndarray:
            u6 = np.zeros((b5_Mmat.shape[0], 1), dtype=float)
            # u6[9] =  -0.5
            # u6[10] = -0.5
            # u6[11] = -0.5
            return u6

        assert not hasattr(self, "b6_Mmat") or self.b6_Mmat is None, (
            f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.svpwm_Mmat = b5_Mmat
        self.svpwm_Kmat = b5_Kmat
        self.svpwm_var_offset = b5_var_offset
        self.svpwm_num_vars = b5_num_vars
        self.get_svpwm_fy_dynamic = _get_fy_dynamic_b5
        self.get_svpwm_u_dynamic = _get_u_dynamic_b5

    def _inverter_bridge(self):
        # b6. 3 phase, 2-level inverter bridge
        b6_var_offset = {
            "E_abc": 0,  # [E_a, E_b, E_c] voltage at terminal of the inverter
            "v_abc": self.n_ph,  # [v_a, v_b, v_c] voltage across each phase
            "Eab": 2 * self.n_ph,
            "Ebc": 1 + 2 * self.n_ph,
            "i_abc": 2
            + 2 * self.n_ph,  # [E_a, E_b, E_c] current in each leg of inverter
            "din": 2
            + 3 * self.n_ph,  # [din_a, din_b, din_b] duty cycle obtained from SVPWM
        }
        b6_num_vars = b6_var_offset["din"] + 3

        # coefficients for eqns of inverter bridge

        ll_ab = np.array([[1, -1, 0]])  # line to line ab
        ll_bc = np.array([[0, 1, -1]])  # line to line bc

        # fx = (Vdc-vsw-ix*rsw)*din_x - (vd + ix*rd)*(1-din_x) for ix>=0
        # fx = (Vdc + vd-ix*rd)*din_x + (vsw-rsw*ix)*(1-din_x) for ix<0

        # fmt:off
        b6_Mmat = sps.bmat(
            [
                # E_abc, v_abc,   Eab,     Ebc,    i_abc,  din,                
                [ None,   a_ph,   None,    None,   Zr_ph,  Zr_ph],  # 1)va-fa=0 (fy) (1)
                [ None,   b_ph,   None,    None,   None,   None],  # 2)vb-fb=0 (fy) (1)
                [ None,   c_ph,   None,    None,   None,   None],  # 3)vc-fc=0 (fy) (1)
                [ a_ph,  -a_ph,   None,    None,   None,   None],  # 4)Ea-va=0 (1)
                [ b_ph,  -b_ph,   None,    None,   None,   None],  # 5)Eb-vb=0 (1)
                [ c_ph,  -c_ph,   None,    None,   None,   None],  # 6)Ec-vc=0 (1)
                [ ll_ab,  None,  -Id_one,  None,   None,   None],  # 7)-Eab + Ea - Eb =0 (1)
                [ ll_bc,  None,   None,   -Id_one, None,   None],  # 8)-Ebc + Eb - Ec=0 (1)
            ]
        )
        # fmt: on

        assert b6_Mmat.shape[0] == 8
        assert b6_Mmat.shape[1] == 14
        assert b6_Mmat.shape[1] == b6_num_vars

        b6_Kmat = sps.lil_matrix(b6_Mmat.shape, dtype=float)

        def _get_fy_dynamic_b6(t, y) -> np.ndarray:
            # fy6: inverter bridge block
            # non-linearity in eqn 1, 2 and 3
            # vx-fx = 0 : x ={a, b, c}
            # fx = (Vdc-vsw-ix*rsw)*din_x - (vd + ix*rd)*(1-din_x) for ix>=0
            # fx = (Vdc + vd+ix*rd)*din_x + (vsw-rsw*ix)*(1-din_x) for ix<0
            fy6 = np.zeros((b6_Mmat.shape[0], 1), dtype=float)
            idx_ia = self.var_offset_dynamic["i_abc"]  # a phase
            idx_ib = self.var_offset_dynamic["i_abc"] + 1  # b phase
            idx_ic = self.var_offset_dynamic["i_abc"] + 2  # c phase
            idx_din_a = self.var_offset_dynamic["din"]  # a phase
            idx_din_b = self.var_offset_dynamic["din"] + 1  # b phase
            idx_din_c = self.var_offset_dynamic["din"] + 2  # c phase
            ia = y[idx_ia]
            ib = y[idx_ib]
            ic = y[idx_ic]
            da = y[idx_din_a]
            db = y[idx_din_b]
            dc = y[idx_din_c]

            # inverter constants:TODO check if all these are paresed
            Vdc = self.obj.Vdc  # DC voltage
            vsw = self.obj.Vsw  # voltage drop across switch
            rsw = self.obj.rsw  # resistance across switch
            vd = self.obj.Vd  # voltage drop across diode
            rd = self.obj.rd  # resistance across diode

            # eq1: fa
            idx_eq1_start = 0
            if ia >= 0:
                fy6[idx_eq1_start] = -(
                    (Vdc - vsw - rsw * ia) * da - (vd + rd * ia) * (1 - da)
                )
            else:
                fy6[idx_eq1_start] = -(
                    (Vdc + vd + rd * ia) * da + (vsw - rsw * ia) * (1 - da)
                )

            # fy7[idx_eq1_start] = -(Vdc * da) / 2

            # eq2 : fb
            idx_eq2_start = 1
            if ib >= 0:
                fy6[idx_eq2_start] = -(
                    (Vdc - vsw - rsw * ib) * db - (vd + rd * ib) * (1 - db)
                )
            else:
                fy6[idx_eq2_start] = -(
                    (Vdc + vd + rd * ib) * db + (vsw - rsw * ib) * (1 - db)
                )

            # fy7[idx_eq2_start] = -(Vdc * db) / 2

            # eq3: fc
            idx_eq3_start = 2
            if ic >= 0:
                fy6[idx_eq3_start] = -(
                    (Vdc - vsw - rsw * ic) * dc - (vd + rd * ic) * (1 - dc)
                )
            else:
                fy6[idx_eq3_start] = -(
                    (Vdc + vd + rd * ic) * dc + (vsw - rsw * ic) * (1 - dc)
                )

            # fy7[idx_eq3_start] = -(Vdc * dc) / 2

            return fy6

        def get_u_dynamic_b6(t, y) -> np.ndarray:
            u6 = np.zeros((b6_Mmat.shape[0], 1), dtype=float)
            return u6

        assert not hasattr(self, "b7_Mmat") or self.b7_Mmat is None, (
            f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.inv_br_Mmat = b6_Mmat
        self.inv_br_Kmat = b6_Kmat
        self.inv_br_var_offset = b6_var_offset
        self.inv_br_num_vars = b6_num_vars
        self.get_inv_br_dynamic = _get_fy_dynamic_b6
        self.get_inv_br_dynamic = get_u_dynamic_b6

    def _vbrf(self):
        ################## Band reject filter for voltage
        # fmt: off
        Vbrf_var_offset = {
            "V_qd": 0, # [Vin_q, Vin_d] input with both dc and oscillating component with frequency 2w
            "Vbrf1_qd": self.n_varqd, # [Vbrf1_q, Vbrf1_d] intermediate output of filter
            "Vbrf2_qd": 2 * self.n_varqd, # [Vbrf2_q, Vbrf2_d] intermediate output of filter
            "Vbrf3_qd": 3 * self.n_varqd, # [Vbrf3_q, Vbrf3_d] intermdediate output of filter
            "V_qd_dc": 4 * self.n_varqd, # [X_q_dc, X_d_dc] output of Band reject filter- dc component of pos/neg sequence input
        }
        # fmt: on
        Vbrf_num_vars = Vbrf_var_offset["V_qd_dc"] + 2

        Id_qd = sps.identity(self.n_varqd, format="coo")
        w_band = 2 * np.pi * 20 * Id_qd  # rad/s
        Zqd_mat = sps.lil_matrix((self.n_varqd, self.n_varqd))

        # fmt:off
        Vbrf_Mmat = sps.bmat([
            # V_qd,   Vbrf1_qd, Vbrf2_qd, Vbrf3_qd, V_qd_dc
            [None,    Zqd_mat,  Id_qd,    None,     None   ], #1) Vbrf2_qd - d(Vbrf1_qd)/dt = 0 (K) (2)
            [-Id_qd,  None,     w_band,   None,     None   ], #2) d(Vbrf1_qd)/dt  + w_band * Vbrf2_qd + w^2*Vbrf1_qd - V_qd = 0 (K) (fy) (2)
            [None,    Zqd_mat,  None,     Id_qd,    None   ], #3) Vbref3_qd - w_band*d(Vbrf1_qd)/dt = 0 (K)  (2)
            [-Id_qd,  None,     None,     Id_qd,    Id_qd  ], #4) V_qd_dc - V_qd + Vbrf3_qd = 0 (2)           
         
        ])
        # fmt:on

        assert Vbrf_Mmat.shape[0] == 8
        assert Vbrf_Mmat.shape[1] == 10
        assert Vbrf_Mmat.shape[1] == Vbrf_num_vars

        # fmt:off
        Vbrf_Kmat = sps.bmat([
            # V_qd,     Vbrf1_qd,   Vbrf2_qd,   Vbrf3_qd,   X_qd_dc
            [Zqd_mat,   -Id_qd,     None,       None,       None   ], #1) Q - d(brf1)/dt = 0 (K) (2)
            [None,      None,       Id_qd,      None,       None   ], #2) d(brf2)/dt  + w_band*Q + w^2*P - Xqd = 0 (K) (fy) (2)
            [None,      -w_band,    None,       Zqd_mat,    None   ], #3) R - w_band*d(brf1)/dt = 0 (K)  (2)
            [None,      None,       None,       None,       Zqd_mat], #4) X_qd_dc - X_qd + R = 0 (2)          
        ])
        # fmt: on

        assert Vbrf_Mmat.shape == Vbrf_Kmat.shape

        def _get_fy_dynamic_Vbrf(t, y) -> np.ndarray:
            fy_vbrf = np.zeros((self.Ibrf_Mmat.shape[0], 1), dtype=float)

            # 2) d(Ibrf2_qd)/dt  + w_band*Ibrf2_qd + w^2*Ibrf1_qd - I_qd = 0 (K) (fy) (2)
            w_reject = 2 * const.w_nominal
            Vbrf1_q = y[self.var_offset_dynamic["Vbrf1_qd"]]
            Vbrf1_d = y[self.var_offset_dynamic["Vbrf1_qd"] + 1]

            fy_vbrf[2] = (w_reject**2) * Vbrf1_q
            fy_vbrf[3] = (w_reject**2) * Vbrf1_d

            return fy_vbrf

        def get_u_Vbrf(t, y) -> np.ndarray:
            u_vbrf = np.zeros((self.Ibrf_Mmat.shape[0], 1), dtype="float")
            return u_vbrf

        assert not hasattr(self, "Vbrf_Mmat") or self.Vbrf_Mmat is None

        self.Vbrf_Mmat = Vbrf_Mmat
        self.Vbrf_Kmat = Vbrf_Kmat
        self.var_offset_dynamic_Vbrf = Vbrf_var_offset
        self.Vbrf_num_vars = Vbrf_num_vars
        self.get_fy_dynamic_Vbrf = _get_fy_dynamic_Vbrf
        self.get_u_Vbrf_dynamic = get_u_Vbrf

    def _ibrf(self):
        ########## Band reject filter for current
        Ibrf_var_offset = {
            "I_qd": 0,  # [I_q, I_d] input with both dc and oscillating component with frequency 2w
            "Ibrf1_qd": self.n_varqd,  # [Ibrf1_q, Ibrf1_d] intermediate output of filter
            "Ibrf2_qd": 2
            * self.n_varqd,  # [Ibrf2_q, Ibrf2_d] intermediate output of filter
            "Ibrf3_qd": 3
            * self.n_varqd,  # [Ibrf3_q, Ibrf3_d] intermdediate output of filter
            "I_qd_dc": 4
            * self.n_varqd,  # [I_q_dc, I_d_dc] output of Band reject filter- dc component of pos/neg sequence input
        }
        Ibrf_num_vars = Ibrf_var_offset["I_qd_dc"] + 2

        Zqd_mat = sps.lil_matrix((self.n_varqd, self.n_varqd))
        Id_qd = sps.identity(self.n_varqd, format="coo")
        w_band = 2 * np.pi * 20 * Id_qd  # rad/s

        # fmt:off
        Ibrf_Mmat = sps.bmat([
            # I_qd,   Ibrf1_qd, Ibrf2_qd, Ibrf3_qd, I_qd_dc
            [None,    Zqd_mat,  Id_qd,    None,     None   ], #1) Ibrf2_qd - d(Ibrf1)/dt = 0 (K) (2)
            [-Id_qd,  None,     w_band,   None,     None   ], #2) d(Ibrf2_qd)/dt  + w_band*Q + w^2*Ibrf1_qd - I_qd = 0 (K) (fy) (2)
            [None,    Zqd_mat,  None,     Id_qd,    None   ], #3) Ibrf3_qd - w_band*d(Ibrf1_qd)/dt = 0 (K)  (2)
            [-Id_qd,  None,     None,     Id_qd,    Id_qd  ], #4) I_qd_dc - I_qd + Ibrf3_qd = 0 (2) 
        ])
        # fmt:on

        assert Ibrf_Mmat.shape[0] == 8
        assert Ibrf_Mmat.shape[1] == 10
        assert Ibrf_Mmat.shape[1] == Ibrf_num_vars

        # fmt:off
        Ibrf_Kmat = sps.bmat([
            # I_qd,     Ibrf1_qd,   Ibrf2_qd,   Ibrf3_qd,   X_qd_dc
            [Zqd_mat,  -Id_qd,      None,       None,       None      ], #1) Q - d(brf1)/dt = 0 (K) (2)
            [None,      None,       Id_qd,      None,       None      ], #2) d(brf2)/dt  + w_band*Q + w^2*P - Xqd = 0 (K) (fy) (2)
            [None,     -w_band,    None,       Zqd_mat,    None      ], #3) R - w_band*d(brf1)/dt = 0 (K)  (2)
            [None,      None,      None,       None,       Zqd_mat   ], #4) X_qd_dc - X_qd + R = 0 (2)          
        ])
        # fmt:on

        assert Ibrf_Mmat.shape == Ibrf_Kmat.shape

        def _get_fy_dynamic_Ibrf(t, y) -> np.ndarray:
            fy_Ibrf = np.zeros((self.Ibrf_Mmat.shape[0], 1), dtype=float)

            # 2) d(Ibrf2_qd)/dt  + w_band*Ibrf2_qd + w^2*Ibrf1_qd - I_qd = 0 (K) (fy) (2)
            w_reject = 2 * const.w_nominal
            Ibrf1_q = y[self.var_offset_dynamic["Ibrf1_qd"]]
            Ibrf1_d = y[self.var_offset_dynamic["Ibrf1_qd"] + 1]

            fy_Ibrf[2] = (w_reject**2) * Ibrf1_q
            fy_Ibrf[3] = (w_reject**2) * Ibrf1_d

            return fy_Ibrf

        def get_u_ibrf(t, y) -> np.ndarray:
            u_ibrf = np.zeros((self.Ibrf_Mmat.shape[0], 1), dtype="float")
            return u_ibrf

        assert not hasattr(self, "Ibrf_Mmat") or self.Ibrf_Mmat is None

        self.Ibrf_Mmat = Ibrf_Mmat
        self.Ibrf_Kmat = Ibrf_Kmat
        self.var_offset_dynamic_Ibrf = Ibrf_var_offset
        self.Ibrf_num_vars = Ibrf_num_vars
        self.get_fy_dynamic_Ibrf = _get_fy_dynamic_Ibrf
        self.get_u_ibrf_dynamic = get_u_ibrf

    ################# end of block funtions #########################
    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        return self.M_dynamic

    def get_K_dynamic(self) -> sps.coo_array:
        return self.K_dynamic

    def get_fy_dynamic(self, t, y) -> sps.coo_array:
        fy_lcl = self.get_lcl_fy_dynamic(t, y)
        fy_mnt = self.get_mnt_fy_dynamic(t, y)
        fy_pll = self.get_pll_fy_dynamic(t, y)
        fy_cc = self.get_cc_fy_dynamic(t, y)
        fy_svpwm = self.get_svpwm_fy_dynamic(t, y)
        fy_inv_br = self.get_inv_br_dynamic(t, y)
        fy_vbrf = self.get_fy_dynamic_Vbrf(t, y)
        fy_ibrf = self.get_fy_dynamic_Ibrf(t, y)
        fy_interface = np.zeros((self.M_dynamic_iface_nrows, 1), dtype="float")

        # fmt: off
        fy = sps.bmat(
            [[fy_lcl],
             [fy_mnt],
             [fy_pll],
             [fy_cc],
             [fy_svpwm],
             [fy_inv_br],
             [fy_vbrf],
             [fy_ibrf],
             [fy_interface]]
        )
        # fmt: on

        assert fy_lcl.shape[0] == self.lcl_Mmat.shape[0]
        assert fy_mnt.shape[0] == self.mnt_Mmat.shape[0]
        assert fy_pll.shape[0] == self.pll_Mmat.shape[0]
        assert fy_cc.shape[0] == self.cc_Mmat.shape[0]
        assert fy_svpwm.shape[0] == self.svpwm_Mmat.shape[0]
        assert fy_inv_br.shape[0] == self.inv_br_Mmat.shape[0]

        assert fy.shape[0] == self.num_eqns_dynamic

        return fy.toarray().flatten()

    def get_u_dynamic(self, t: float, y) -> np.ndarray:
        u_lcl = self.get_lcl_u_dynamic(t, y)
        u_mnt = self.get_mnt_u_dynamic(t, y)
        u_pll = self.get_pll_u_dynamic(t, y)
        u_cc = self.get_cc_u_dynamic(t, y)
        u_svpwm = self.get_svpwm_u_dynamic(t, y)
        u_inv_br = self.get_inv_br_dynamic(t, y)
        u_vbrf = self.get_u_Vbrf_dynamic(t, y)
        u_ibrf = self.get_u_ibrf_dynamic(t, y)

        u_interface = np.zeros((self.M_dynamic_iface_nrows, 1), dtype="float")

        # fmt: off
        u_dynamic = sps.bmat(
            [[u_lcl],
             [u_mnt],
             [u_pll],
             [u_cc],
             [u_svpwm],
             [u_inv_br],
             [u_vbrf],
             [u_ibrf],
             [u_interface]]
        )
        # fmt: on

        return u_dynamic.toarray().flatten()

    def initial_guess_dynamic_zero(self, y_comp: list, wnom) -> np.ndarray:
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        # initialization for start from zero:
        Vnom = list(self.nominal_voltage.values())[0]
        V = Vnom
        delta = 0

        # dyn_idx_w = self.var_offset_dynamic["w"]
        # y0_dyn[dyn_idx_w] = wnom

        # 1. init "constants"
        # - Pref
        # - Qref

        V_start_idx = self.var_offset["V"] + self.n_ph  # voltage at Vc
        V_end_idx = V_start_idx + self.n_ph
        Vc = y_comp[V_start_idx:V_end_idx]

        I_start_idx = self.var_offset["I"]
        I_end_idx = I_start_idx + self.n_ph
        I = y_comp[I_start_idx:I_end_idx]
        I = -I  # since we want to use this I to calculate S for the node

        S = Vc * I.conjugate()

        self.Pref = S.real
        self.Qref = S.imag

        # self.Pref_total = 0
        # self.Qref_total = 0
        self.Pref_total = sum(self.Pref[i] for i in range(len(self.Pref)))
        self.Qref_total = sum(self.Qref[i] for i in range(len(self.Qref)))

        # self.Pref_total = 5e5  # from Oindrilla's code

        Iref = (2 / 3) * self.Pref_total / (V * np.sqrt(2) / self.Iqd_b)

        # multiply by a factor of 0.8 to keep the grid voltage lower than nominal
        # multiply by sqrt(2) to get the peak value
        Va = 0.8 * np.sqrt(2) * V * np.cos(delta)
        Vb = 0.8 * np.sqrt(2) * V * np.cos(delta - 2 * np.pi / 3)
        Vc = 0.8 * np.sqrt(2) * V * np.cos(delta + 2 * np.pi / 3)

        # LCL filter:
        # Vgf,    Vcf,    Vinf, Vab,     Vbc,   Igf,    Iinf,   vgf,    vcf,    vinf,   vccf,   igf,    icf,    iinf,   qf,     lamda_g, lamda_in
        # 1 Vgf
        idx_V_start = self.var_offset_dynamic["V"]
        idx_V_end = idx_V_start + self.n_ph
        y0_dyn[idx_V_start:idx_V_end] = [Va, Vb, Vc]

        Vg_q = (
            (2 / 3)
            * (
                Va * np.cos(delta)
                + Vb * np.cos(delta - 2 * np.pi / 3)
                + Vc * np.cos(delta + 2 * np.pi / 3)
            )
            / self.Vqd_b
        )
        Vg_d = (
            (2 / 3)
            * (
                Va * np.sin(delta)
                + Vb * np.sin(delta - 2 * np.pi / 3)
                + Vc * np.sin(delta + 2 * np.pi / 3)
            )
            / self.Vqd_b
        )

        Ig_q = self.Pref_total / self.Pb / Vg_q
        # Ig_q = 5e5 / self.Pb / Vg_q
        Ig_d = 0

        Ia = (Ig_q * np.cos(delta) + Ig_d * np.sin(delta)) * self.I_base
        Ib = (
            Ig_q * np.cos(delta - 2 * np.pi / 3) + Ig_d * np.sin(delta - 2 * np.pi / 3)
        ) * self.I_base
        Ic = (
            Ig_q * np.cos(delta + 2 * np.pi / 3) + Ig_d * np.sin(delta + 2 * np.pi / 3)
        ) * self.I_base

        # 2 Igf
        idx_Igf_start = self.var_offset_dynamic["I"]
        idx_Igf_end = self.var_offset_dynamic["I"] + self.n_ph

        idx_igf_start = self.var_offset_dynamic["igf"]
        idx_igf_end = idx_igf_start + self.n_ph

        print(f"Ia.shape:{Ia.shape}")
        print(
            f"y0_dyn[idx_Igf_start:idx_Igf_end].shape: {y0_dyn[idx_Igf_start:idx_Igf_end].shape}"
        )
        print(f"Ia:{Ia}")
        print(f"Ib:{Ib}")
        print(f"Ic:{Ic}")
        input("continue?")

        y0_dyn[idx_Igf_start:idx_Igf_end] = [Ia, Ib, Ic]

        # 3 igf
        y0_dyn[idx_igf_start:idx_igf_end] = [Ia, Ib, Ic]

        # 4 vgf
        vgf_a = self.r1_mat[0][0] * Ia
        vgf_b = self.r1_mat[1][1] * Ib
        vgf_c = self.r1_mat[2][2] * Ic

        idx_vgf_start = self.var_offset_dynamic["vgf"]
        idx_vgf_end = idx_vgf_start + self.n_ph

        y0_dyn[idx_vgf_start:idx_vgf_end] = [vgf_a, vgf_b, vgf_c]

        # 5 Vcf
        # 6 vcf
        # 7 vccf

        Vcf_a = Va + vgf_a
        Vcf_b = Vb + vgf_b
        Vcf_c = Vc + vgf_c

        idx_Vcf_start = self.var_offset_dynamic["Vcf"]
        idx_Vcf_end = idx_Vcf_start + self.n_ph
        y0_dyn[idx_Vcf_start:idx_Vcf_end] = [Vcf_a, Vcf_b, Vcf_c]

        idx_vcf_start = self.var_offset_dynamic["vcf"]
        idx_vcf_end = idx_vcf_start + self.n_ph
        y0_dyn[idx_vcf_start:idx_vcf_end] = [Vcf_a, Vcf_b, Vcf_c]

        idx_vccf_start = self.var_offset_dynamic["vccf"]
        idx_vccf_end = idx_vccf_start + self.n_ph
        y0_dyn[idx_vccf_start:idx_vccf_end] = [Vcf_a, Vcf_b, Vcf_c]

        # 8 Iinf = Igf
        # 9 iinf = igf
        idx_Iinf_start = self.var_offset_dynamic["Iinf"]
        idx_Iinf_end = idx_Iinf_start + self.n_ph
        y0_dyn[idx_Iinf_start:idx_Iinf_end] = [Ia, Ib, Ic]

        idx_igf_start = self.var_offset_dynamic["igf"]
        idx_igf_end = idx_igf_start + self.n_ph
        y0_dyn[idx_igf_start:idx_igf_end] = [Ia, Ib, Ic]

        # 10 icf = 0
        # 11 qf = 0
        # 12 lamda_in = 0
        # 13 lamda_g = 0

        # 14 vinf
        vinf_a = self.r2_mat[0][0] * Ia
        vinf_b = self.r2_mat[1][1] * Ib
        vinf_c = self.r2_mat[2][2] * Ia

        idx_vinf_start = self.var_offset_dynamic["vinf"]
        idx_vinf_end = idx_vinf_start + self.n_ph
        y0_dyn[idx_vinf_start:idx_vinf_end] = [vinf_a, vinf_b, vinf_c]

        # 15 Vinf
        Vinf_a = Vcf_a + vinf_a
        Vinf_b = Vcf_b + vinf_b
        Vinf_c = Vcf_c + vinf_c

        idx_Vinf_start = self.var_offset_dynamic["Vinf"]
        idx_Vinf_end = idx_Vinf_start + self.n_ph
        y0_dyn[idx_Vinf_start:idx_Vinf_end] = [Vinf_a, Vinf_b, Vinf_c]

        # 16 Vab
        # 17 Vbc
        Vab = Va - Vb
        Vbc = Vb - Vc
        y0_dyn[self.var_offset_dynamic["Vab"]] = Vab
        y0_dyn[self.var_offset_dynamic["Vbc"]] = Vbc

        ###### mnt block
        # [Vg,     Vg_ll,     Vg_qd,  ig,     ig_qd]

        # 1 Vg = Vgf = V
        idx_Vg_start = self.var_offset_dynamic["Vg"]
        idx_Vg_end = idx_Vg_start + self.n_ph
        y0_dyn[idx_Vg_start:idx_Vg_end] = [Va, Vb, Vc]

        # 2 Vgll = [Vab, Vbc]
        idx_Vgll_start = self.var_offset_dynamic["Vg_ll"]
        idx_Vgll_end = idx_Vgll_start + 2
        y0_dyn[idx_Vgll_start:idx_Vgll_end] = [Vab, Vbc]

        # 3 Vg_qd
        idx_Vg_qd_start = self.var_offset_dynamic["Vg_qd"]
        idx_Vg_qd_end = idx_Vg_qd_start + 2
        y0_dyn[idx_Vg_qd_start:idx_Vg_qd_end] = [Vg_q, Vg_d]

        # 4 ig = igf
        idx_ig_start = self.var_offset_dynamic["ig"]
        idx_ig_end = idx_ig_start + self.n_ph
        y0_dyn[idx_ig_start:idx_ig_end] = [Ia, Ib, Ic]

        # 5 ig_qd = [Ig_q, Ig_d]
        idx_ig_qd_start = self.var_offset_dynamic["ig_qd"]
        idx_ig_qd_end = idx_ig_qd_start + 2
        y0_dyn[idx_ig_qd_start:idx_ig_qd_end] = [Ig_q, Ig_d]

        ##### pll block
        # [w, Vg_pll_qd, Vg_d_p, Vg_d_int, delta]
        # w
        y0_dyn[self.var_offset_dynamic["w"]] = 1  # we/wb and we = wb

        # Vg_pll_qd
        idx_vgpll_qd_start = self.var_offset_dynamic["Vg_pll_qd"]
        idx_vgpll_qd_end = idx_vgpll_qd_start + 2
        y0_dyn[idx_vgpll_qd_start:idx_vgpll_qd_end] = [Vg_q, Vg_d]

        # Vg_d_p
        y0_dyn[self.var_offset_dynamic["Vg_d_p"]] = -self.obj.Kppll * Vg_d

        # Vg_d_int
        y0_dyn[self.var_offset_dynamic["Vg_d_int"]] = 1  # oindrilla's code

        # delta = 0

        ##### current controller
        # Vg_cc_qd,  iqd_ref,    ig_cc_qd, iqd_diff,  prop_cs_qd, int_cs_qd, pi_co_qd, e_qd,  e_qd_fin,
        # 1 Vg_cc_qd
        idx_Vg_cc_qd_start = self.var_offset_dynamic["Vg_cc_qd"]
        idx_Vg_cc_qd_end = idx_Vg_cc_qd_start + 2
        y0_dyn[idx_Vg_cc_qd_start:idx_Vg_cc_qd_end] = [Vg_q, Vg_d]

        # 2 iqd_ref
        # 3 ig_cc_qd
        idx_iqd_ref_start = self.var_offset_dynamic["iqd_ref"]
        idx_iqd_ref_end = idx_iqd_ref_start + 2
        y0_dyn[idx_iqd_ref_start:idx_iqd_ref_end] = [Ig_q, Ig_d]

        idx_ig_cc_qd_start = self.var_offset_dynamic["ig_cc_qd"]
        idx_ig_cc_qd_end = idx_ig_cc_qd_start + 2
        y0_dyn[idx_ig_cc_qd_start:idx_ig_cc_qd_end] = [Ig_q, Ig_d]

        # 4) iqd_diff = 0
        # 5) prop_cs_qd = 0

        Ea = Vinf_a - self.obj.Vsw - self.obj.Vd + Ia * (self.obj.rsw + self.obj.rd)
        Eb = Vinf_b - self.obj.Vsw - self.obj.Vd + Ib * (self.obj.rsw + self.obj.rd)
        Ec = Vinf_c - self.obj.Vsw - self.obj.Vd + Ib * (self.obj.rsw + self.obj.rd)

        e_q = (
            (2 / 3)
            * (
                Ea * np.cos(delta)
                + Eb * np.cos(delta - 2 * np.pi / 3)
                + Ec * np.cos(delta + 2 * np.pi / 3)
            )
            / self.Vqd_b
        )
        e_d = (
            (2 / 3)
            * (
                Ea * np.sin(delta)
                + Eb * np.sin(delta - 2 * np.pi / 3)
                + Ec * np.sin(delta + 2 * np.pi / 3)
            )
            / self.Vqd_b
        )

        # 6 e_qd
        # 7 e_qd_f
        idx_e_qd_start = self.var_offset_dynamic["e_qd"]
        idx_e_qd_end = idx_e_qd_start + 2
        y0_dyn[idx_e_qd_start:idx_e_qd_end] = [e_q, e_d]

        idx_e_qdf_start = self.var_offset_dynamic["e_qd_fin"]
        idx_e_qdf_end = idx_e_qdf_start + 2
        y0_dyn[idx_e_qdf_start:idx_e_qdf_end] = [e_q, e_d]

        int_cs_q = e_q - Vg_q - Ig_d * (self.r1_mat[0][0] + self.r2_mat[0][0]) / self.Zb
        pi_co_q = e_q - Vg_q - Ig_d * (self.r1_mat[0][0] + self.r2_mat[0][0]) / self.Zb
        y0_dyn[self.var_offset_dynamic["int_cs_qd"]] = int_cs_q
        y0_dyn[self.var_offset_dynamic["pi_co_qd"]] = pi_co_q

        ####svpwm
        # m_qd, m_qd_f, v_qd_ref,   m_abc

        # 1 v_qd_ref
        idx_v_qd_ref_start = self.var_offset_dynamic["v_qd_ref"]
        idx_v_qd_ref_end = idx_v_qd_ref_start + 2
        y0_dyn[idx_v_qd_ref_start:idx_v_qd_ref_end] = [e_q, e_d]

        # 2 m_qd,
        # 3 m_qd_f

        # m_q = self.obj.Vdc/self.V_base/e_q
        # m_d = self.obj.Vdc/self.V_base/e_d
        m_q = e_q * self.Vqd_b * 2 / self.obj.Vdc
        m_d = e_d * self.Vqd_b * 2 / self.obj.Vdc
        idx_m_qd_start = self.var_offset_dynamic["m_qd"]
        idx_m_qd_end = idx_m_qd_start + 2
        y0_dyn[idx_m_qd_start:idx_m_qd_end] = [m_q, m_d]

        idx_mqd_f_start = self.var_offset_dynamic["m_qd_f"]
        idx_mqd_f_end = idx_mqd_f_start + 2
        y0_dyn[idx_mqd_f_start:idx_mqd_f_end] = [m_q, m_d]

        # 4 m_abc
        ma = m_q * np.cos(delta) + m_d * np.sin(delta)
        mb = m_q * np.cos(delta - 2 * np.pi / 3) + m_d * np.sin(delta - 2 * np.pi / 3)
        mc = m_q * np.cos(delta + 2 * np.pi / 3) + m_d * np.sin(delta + 2 * np.pi / 3)

        idx_m_abc_start = self.var_offset_dynamic["m_abc"]
        idx_m_abc_end = idx_m_abc_start + self.n_ph
        y0_dyn[idx_m_abc_start:idx_m_abc_end] = [ma, mb, mc]

        ####inverter bridge
        # E_abc, v_abc,   Eab,     Ebc,    i_abc,  din,
        # 1  din
        idx_din_start = self.var_offset_dynamic["din"]
        idx_din_end = idx_din_start + self.n_ph
        y0_dyn[idx_din_start:idx_din_end] = [ma, mb, mc]

        # 2 i_abc
        idx_i_abc_start = self.var_offset_dynamic["i_abc"]
        idx_i_abc_end = idx_i_abc_start + self.n_ph
        y0_dyn[idx_i_abc_start:idx_i_abc_end] = [Ia, Ib, Ic]

        # 3 Eab
        # 4 Ebc
        y0_dyn[self.var_offset_dynamic["Eab"]] = Vinf_a - Vinf_b
        y0_dyn[self.var_offset_dynamic["Ebc"]] = Vinf_b - Vinf_c

        # 5 E_abc
        # 6 v_abc
        idx_Eabc_start = self.var_offset_dynamic["E_abc"]
        idx_Eabc_end = idx_Eabc_start + self.n_ph
        y0_dyn[idx_Eabc_start:idx_Eabc_end] = [Vinf_a, Vinf_b, Vinf_c]

        idx_vabc_start = self.var_offset_dynamic["v_abc"]
        idx_vabc_end = idx_vabc_start + self.n_ph
        y0_dyn[idx_vabc_start:idx_vabc_end] = [Vinf_a, Vinf_b, Vinf_c]

        # Vbrf
        # V_qd,     Vbrf1_qd,   Vbrf2_qd,   Vbrf3_qd,   X_qd_dc
        w_reject = 2 * const.w_nominal
        w2 = 1 / w_reject**2
        idx_V_qd_start = self.var_offset_dynamic["V_qd"]
        idx_V_qd_end = idx_V_qd_start + self.n_varqd
        y0_dyn[idx_V_qd_start:idx_V_qd_end] = [Vg_q, Vg_d]

        idx_V_qd_dc_start = self.var_offset_dynamic["V_qd_dc"]
        idx_V_qd_dc_end = idx_V_qd_dc_start + self.n_varqd
        y0_dyn[idx_V_qd_dc_start:idx_V_qd_dc_end] = [Vg_q, Vg_d]

        idx_Vbrf1_qd_start = self.var_offset_dynamic["Vbrf1_qd"]
        idx_Vbrf1_qd_end = idx_Vbrf1_qd_start + self.n_varqd
        y0_dyn[idx_Vbrf1_qd_start:idx_Vbrf1_qd_end] = [w2 * Vg_q, w2 * Vg_d]

        # Ibrf
        # I_qd,   Ibrf1_qd, Ibrf2_qd, Ibrf3_qd, I_qd_dc
        idx_I_qd_start = self.var_offset_dynamic["I_qd"]
        idx_I_qd_end = idx_I_qd_start + self.n_varqd
        y0_dyn[idx_I_qd_start:idx_I_qd_end] = [Ig_q, Ig_d]

        idx_I_qd_dc_start = self.var_offset_dynamic["I_qd_dc"]
        idx_I_qd_dc_end = idx_I_qd_dc_start + self.n_varqd
        y0_dyn[idx_I_qd_dc_start:idx_I_qd_dc_end] = [Ig_q, Ig_d]

        idx_Ibrf1_qd_start = self.var_offset_dynamic["Ibrf1_qd"]
        idx_Ibrf1_qd_end = idx_Ibrf1_qd_start + self.n_varqd
        y0_dyn[idx_Ibrf1_qd_start:idx_Ibrf1_qd_end] = [w2 * Ig_q, w2 * Ig_d]

        # V_start_idx = self.var_offset["V"] + self.n_ph  # voltage at Vc
        # V_end_idx = V_start_idx + self.n_ph
        # Vc = y_comp[V_start_idx:V_end_idx]

        # Vg_start_idx = self.var_offset["V"]  # voltage at Vg
        # Vg_end_idx = V_start_idx + self.n_ph
        # Vg = y_comp[V_start_idx:V_end_idx]

        # I_start_idx = self.var_offset["I"]
        # I_end_idx = I_start_idx + self.n_ph
        # I = y_comp[I_start_idx:I_end_idx]
        # I = -I  # since we want to use this I to calculate S for the node

        # S = Vc * I.conjugate()

        # self.Pref = S.real
        # self.Qref = S.imag

        # # self.Pref_total = 0
        # # self.Qref_total = 0
        # self.Pref_total = sum(self.Pref[i] for i in range(len(self.Pref)))
        # self.Qref_total = sum(self.Qref[i] for i in range(len(self.Qref)))

        # delta = 0

        # # Vg
        # Vg_start_idx, Vg_end_idx = utils.get_start_end_idx(self.var_offset_dynamic, "Vg", self.n_ph)
        # Vg = [
        #     np.sqrt(2) * phasor_to_timedomain(val) for val in Vg
        # ]
        # y0_dyn[Vg_start_idx: Vg_end_idx] = Vg

        # Vg_qd_start, Vg_qd_end = utils.get_start_end_idx(self.var_offset_dynamic, "Vg_cc_qd", self.n_varqd)
        # Vg_q = (-2 / 3) * (
        #         Vg[0] * np.cos(delta)
        #         + Vg[1] * np.cos(delta + 2 * np.pi / 3)
        #         + Vg[2] * np.cos(delta - 2 * np.pi / 3)
        #     )/self.Vqd_b
        # Vg_d = (-2 / 3) * (
        #         Vg[0] * np.sin(delta)
        #         + Vg[1] * np.sin(delta + 2 * np.pi / 3)
        #         + Vg[2] * np.sin(delta - 2 * np.pi / 3)
        #     )/self.Vqd_b
        # y0_dyn[Vg_qd_start: Vg_qd_end] = [Vg_q, Vg_d]

        # Vc_q = (-2 / 3) * (
        #         Vc[0] * np.cos(delta)
        #         + Vc[1] * np.cos(delta + 2 * np.pi / 3)
        #         + Vc[2] * np.cos(delta - 2 * np.pi / 3)
        #     )/self.Vqd_b
        # Vc_d = (-2 / 3) * (
        #         Vc[0] * np.sin(delta)
        #         + Vc[1] * np.sin(delta + 2 * np.pi / 3)
        #         + Vc[2] * np.sin(delta - 2 * np.pi / 3)
        #     )/self.Vqd_b

        # y0_dyn[Vg_qd_start: Vg_qd_end] = [Vg_q, Vg_d]

        return y0_dyn

    def initial_guess_dynamic(self, y_comp: list) -> np.ndarray:
        raise NotImplementedError
        assert len(y_comp) == self.num_vars

        # 1. init "constants"
        # - Pref
        # - Qref

        V = y_comp[0 : self.n_ph]

        I_start_idx = self.var_offset["I"]
        I_end_idx = I_start_idx + self.n_ph
        I = y_comp[I_start_idx:I_end_idx]

        S = V * I.conjugate()

        self.Pref = S.real
        self.Qref = S.imag

        # 2. init vars
        # resume-here
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        y0_dyn[0] = y_comp[0].real

        # rest:

        # powerflow variables:
        # [w     V     I      v      i     q     lamda]

        # initialize values from powerflow (pf) to dynamic (dyn):
        #
        # 1. terminal V and I (initialized Vgf and Igf as LCL vars):
        # [ ] anything else?
        #
        # 2. LCL filter
        # [x] Vgf <- pf V[0-3] (V1)
        # [x] Vcf <- pf V[3-6] (V2)
        # [x] Vinf <- pf V[6-9](V3)
        # [x] Igf <- pf V[0-3] (I1)
        # [x] Icf <- pf I[3-6] (I2)
        # [x] Iinf <- pf (I3)
        # [xj] vgf <- pf (v1)
        # [x] vcf <- pf (v2)
        # [x] vinf <- pf (v3)
        # [x ] vccf <- pf (vc)
        # [x] igf <- pf (i1)
        # [x] icf <- pf (i2)
        # [x] iinf <- pf (i3)
        # [x ] qf
        # [x] lamda_g
        # [x] lamda_in

        # Vgf, Vcf, Vinf:
        # NOTE: since these variables are consecutive both in PF and Dyn, initilizating
        # these together.
        pf_V_start_idx = self.var_offset["V"]
        pf_V_end_idx = pf_V_start_idx + 3 * self.n_ph
        dyn_V_start_idx = self.var_offset_dynamic["Vgf"]
        dyn_V_end_idx = dyn_V_start_idx + 3 * self.n_ph
        y0_dyn[dyn_V_start_idx:dyn_V_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_V_start_idx:pf_V_end_idx]
        ]

        # Igf, Icf, Iinf:
        # NOTE: since these variables are consecutive both in PF and Dyn, initilizating
        # these together.
        pf_I_start_idx = self.var_offset["I"]
        pf_I_end_idx = pf_I_start_idx + 3 * self.n_ph
        dyn_I_start_idx = self.var_offset["I"]
        dyn_I_end_idx = dyn_I_start_idx + 3 * self.n_ph
        y0_dyn[dyn_I_start_idx:dyn_I_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_I_start_idx:pf_I_end_idx]
        ]

        # vgf, vcf, vinf, vccf:
        # vgf = v1
        pf_v1_start_idx = self.var_offset["v"]
        pf_v1_end_idx = pf_v1_start_idx + self.n_ph
        dyn_vgf_start_idx = self.var_offset["v"]
        dyn_vgf_end_idx = dyn_vgf_start_idx + self.n_ph
        y0_dyn[dyn_vgf_start_idx:dyn_vgf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_v1_start_idx:pf_v1_end_idx]
        ]
        # vcf = v2
        pf_v2_start_idx = self.var_offset["v"] + self.n_ph
        pf_v2_end_idx = pf_v2_start_idx + self.n_ph
        dyn_vcf_start_idx = self.var_offset_dynamic["vcf"]
        dyn_vcf_end_idx = dyn_vcf_start_idx + self.n_ph
        y0_dyn[dyn_vcf_start_idx:dyn_vcf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_v2_start_idx:pf_v2_end_idx]
        ]
        # vccf = vcc
        pf_vcc_start_idx = self.var_offset["v"] + 2 * self.n_ph
        pf_vcc_end_idx = pf_vcc_start_idx + self.n_ph
        dyn_vccf_start_idx = self.var_offset_dynamic["vccf"]
        dyn_vccf_end_idx = dyn_vccf_start_idx + self.n_ph
        y0_dyn[dyn_vccf_start_idx:dyn_vccf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_vcc_start_idx:pf_vcc_end_idx]
        ]
        # vinf = v3
        pf_v3_start_idx = self.var_offset["v"] + 3 * self.n_ph
        pf_v3_end_idx = pf_v3_start_idx + self.n_ph
        dyn_vinf_start_idx = self.var_offset_dynamic["vinf"]
        dyn_vinf_end_idx = dyn_vinf_start_idx + self.n_ph
        y0_dyn[dyn_vinf_start_idx:dyn_vinf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_v3_start_idx:pf_v3_end_idx]
        ]

        # igf, icf, iinf:
        # NOTE: since these variables are consecutive both in PF and Dyn, initilizating
        # these together.
        pf_i_start_idx = self.var_offset["i"]
        pf_i_end_idx = pf_i_start_idx + 3 * self.n_ph
        dyn_i_start_idx = self.var_offset["i"]
        dyn_i_end_idx = dyn_i_start_idx + 3 * self.n_ph
        y0_dyn[dyn_i_start_idx:dyn_i_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_i_start_idx:pf_i_end_idx]
        ]

        # qf = q
        pf_q_start_idx = self.var_offset["q"]
        pf_q_end_idx = pf_q_start_idx + self.n_ph
        dyn_qf_start_idx = self.var_offset_dynamic["qf"]
        dyn_qf_end_idx = dyn_qf_start_idx + self.n_ph
        y0_dyn[dyn_qf_start_idx:dyn_qf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_q_start_idx:pf_q_end_idx]
        ]

        # lamda_g = lamda1
        # lamda_in = lamda2]
        pf_lamda_start_idx = self.var_offset["lamda"]
        pf_lamda_end_idx = pf_lamda_start_idx + 2 * self.n_ph
        dyn_lamda_start_idx = self.var_offset_dynamic["lamda_g"]
        dyn_lamda_end_idx = dyn_lamda_start_idx + 2 * self.n_ph
        y0_dyn[dyn_lamda_start_idx:dyn_lamda_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_lamda_start_idx:pf_lamda_end_idx]
        ]

        # inverter output voltage E_abc
        # E_abc = V3
        pf_V3_start_idx = self.var_offset["V"] + 2 * self.n_ph
        pf_V3_end_idx = pf_V3_start_idx + self.n_ph
        dyn_E_abc_start_idx = self.var_offset_dynamic["E_abc"]
        dyn_E_abc_end_idx = dyn_E_abc_start_idx + self.n_ph
        y0_dyn[dyn_E_abc_start_idx:dyn_E_abc_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_V3_start_idx:pf_V3_end_idx]
        ]

        # inverter output current
        # i_abc = i3
        pf_i3_start_idx = self.var_offset["i"] + 2 * self.n_ph
        pf_i3_end_idx = pf_i3_start_idx + self.n_ph
        dyn_i_abc_start_idx = self.var_offset_dynamic["i_abc"]
        dyn_i_abc_end_idx = dyn_i_abc_start_idx + self.n_ph
        y0_dyn[dyn_i_abc_start_idx:dyn_i_abc_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_i3_start_idx:pf_i3_end_idx]
        ]
        return y0_dyn
