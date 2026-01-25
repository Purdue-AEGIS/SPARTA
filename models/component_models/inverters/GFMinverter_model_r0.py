from oodesign import *
import utils
from models.model import Model, ValType
from oodesign import GFMInverter, GFMInverter3Ph, Source
import numpy as np
import scipy.sparse as sps
import const
from const import NodeSide, StudyType
from pprint import pformat
from utils import phasor_to_timedomain, get_start_end_idx, get_start_end_idx_with_offset, abc_to_qd, qd_to_abc
from models.component_models.source_model import SourceModel
import itertools
from models.component_models.inverters.inverter_model import *

# Revision Info:
# r0 : All 3 studies working


class GFMInverter3PhModel(GFMInverterModel):
    def __init__(self, inverter_obj: GFMInverter3Ph):
        super().__init__(inverter_obj)

        self.seq_mode = False
        self.Eabc_derived = None
        self.n_ph = 3
        self.n_varqd = 2
        self.n_varll = 2

        # Note: For powerflow the models remains same for both sequence and non-sequence control
        # y_pf = [w, V, I, v, i , q , lamda]
        # V = [V1, V2, V3]
        # I = [I1, I2, I3]
        # v = [ v1, v2, vc, v3]
        # i = [i1, i2, i3]
        # q = [q]
        # lamda = [lamda1, lamda2]
        self.Pref = None  # pu
        self.Qref = None  # pu

        # self.V_base = self.obj.Vdc / np.sqrt(2)  # V (based on SVPWM) rms value (l-l)
        # self.V_base = self.obj.Vdc / 2  # V peak phase voltage
        self.V_base = self.obj.V_base
        self.Pb = self.obj.Pb
        self.I_base = self.Pb / (np.sqrt(3) * self.V_base)
        self.Zb = self.V_base**2 / self.Pb

        self.Vqd_b = np.sqrt(2 / 3) * self.V_base
        self.Iqd_b = np.sqrt(2) * self.I_base
        self.w_b = const.w_nominal

        ##########Powerflow#####################
        self.num_vars_real = 1  # w

        self.num_vars_complex = (
            3 * self.n_ph  # V1, V2, V3
            + 3 * self.n_ph  # I1, I2, I3
            + 4 * self.n_ph  # v1, v2, vc, v3
            + 3 * self.n_ph  # i1, i2, i3
            + self.n_ph  # q
            + 2 * self.n_ph  # lamda1, lamda2
        )

        self.num_vars = self.num_vars_real + self.num_vars_complex

        self.var_offset_real = {
            "w": 0,
        }

        # fmt: off
        self.var_offset_complex = {
            "V": 0,
            "I": 3 * self.n_ph,
            "v": 3 * self.n_ph + 3 * self.n_ph,
            "i": 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph,
            "q": 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph + 3 * self.n_ph,
            "lamda": 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph + 3 * self.n_ph + self.n_ph,
        }
        # fmt: on

        # fmt: off
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + 3 * self.n_ph,
            "v": 1 + 3 * self.n_ph + 3 * self.n_ph,
            "i": 1 + 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph,
            "q": 1 + 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph + 3 * self.n_ph,
            "lamda": 1 + 3 * self.n_ph + 3 * self.n_ph + 4 * self.n_ph + 3 * self.n_ph + self.n_ph,
        }
        # fmt: on

        assert self.var_offset["q"] == self.var_offset_complex["q"] + 1
        assert self.var_offset["lamda"] == self.var_offset_complex["lamda"] + 1

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

        # dynamic variables and eqns book keeping
        # TODO: this should be done only when doing dynamic studies
        self.num_eqns_dynamic = None
        self.num_vars_dynamic = None
        self.M_dynamic = None

        self.b1_Mmat = None
        self.b2_Mmat = None
        self.b3_Mmat = None
        self.b4_Mmat = None
        self.b5_Mmat = None
        self.b6_Mmat = None
        self.b7_Mmat = None

        self.var_offset_dynamic_b1 = None
        self.var_offset_dynamic_b2 = None
        self.var_offset_dynamic_b3 = None
        self.var_offset_dynamic_b4 = None
        self.var_offset_dynamic_b5 = None
        self.var_offset_dynamic_b6 = None
        self.var_offset_dynamic_b7 = None

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

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # y_pf = [w, V, I, v, i , q , lamda]
        # V = [V1, V2, V3]
        # I = [I1, I2, I3]
        # v = [ v1, v2, vc, v3]
        # i = [i1, i2, i3]
        # q = [q]
        # lamda = [lamda1, lamda2]
        # vin = [vin_a, vin_b, vin_c]  inverter leg voltage
        # din = [din_a, din_b, din_c]  duty cycle

        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_V1_start = self.var_offset["V"]
        idx_V1_end = idx_V1_start + self.n_ph
        y_0[idx_V1_start:idx_V1_end, 0] = v_phasors

        idx_V3_start = self.var_offset["V"] + 2 * self.n_ph
        idx_V3_end = idx_V3_start + self.n_ph
        y_0[idx_V3_start:idx_V3_end, 0] = v_phasors

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
        # V = [V1, V2, V3]
        V1 = sps.bmat([[Id_ph, Z_ph, Z_ph]])
        V2 = sps.bmat([[Z_ph, Id_ph, Z_ph]])
        V3 = sps.bmat([[Z_ph, Z_ph, Id_ph]])

        # I = [I1, I2, I3]
        I1 = sps.bmat([[Id_ph, Z_ph, Z_ph]])
        I2 = sps.bmat([[Z_ph, Id_ph, Z_ph]])
        I3 = sps.bmat([[Z_ph, Z_ph, Id_ph]])

        # v = [ v1, v2, vc, v3]
        v1 = sps.bmat([[Id_ph, Z_ph, Z_ph, Z_ph]])
        v2 = sps.bmat([[Z_ph, Id_ph, Z_ph, Z_ph]])
        vc = sps.bmat([[Z_ph, Z_ph, Id_ph, Z_ph]])
        v3 = sps.bmat([[Z_ph, Z_ph, Z_ph, Id_ph]])

        # i = [i1, i2, i3]
        i1 = sps.bmat([[Id_ph, Z_ph, Z_ph]])
        i2 = sps.bmat([[Z_ph, Id_ph, Z_ph]])
        i3 = sps.bmat([[Z_ph, Z_ph, Id_ph]])
        i1_r1 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]])
        i2_r1 = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0]])
        i3_r1 = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1]])
        V3_r1 = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1]])
        V1_r1 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]])
        V2_r1 = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0]])

        # q = [q]
        q = Id_ph

        # lamda = [lamda1, lamda2]
        lamda1 = sps.bmat([[Id_ph, Z_ph]])
        lamda2 = sps.bmat([[Z_ph, Id_ph]])

        C = sps.bmat([[Z_ph, Z_ph, self.C_mat, Z_ph]])
        r1 = sps.bmat([[self.r1_mat, Z_ph, Z_ph]])
        r2 = sps.bmat([[Z_ph, self.r2_mat, Z_ph]])
        r3 = sps.bmat([[Z_ph, Z_ph, self.r3_mat]])
        L1 = sps.bmat([[self.L2_mat, Z_ph, Z_ph]])
        L2 = sps.bmat([[Z_ph, Z_ph, self.L1_mat]])
        # fmt:on

        Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)
        Z_wr = sps.lil_matrix((1, 1), dtype=float)

        # coefficient matrices for eqns
        eq1_V = V3 - V2
        eq6_i = i1 + i2 - i3
        eq7_v = -v2 + vc
        eq10_V = V2 - V1
        # eq15_i = i2 # i2a+i2b+i2c = 0
        # eq16_i = i1 # i1a+i1b+i1c = 0

        # create the M matrix
        # fmt:off
        # M = sps.bmat([
        #     #[w     V           I       v     vin,    i       din    q       lamda]
        #      [Z_w,  eq1_V,    None,   -v3,    None,  None,    Z_ph,    None,   None ],  #1)V3 - V2 - v3 = 0 
        #      [Z_w,  None,     -I3,    None,   None,  i3,      None,    None,   None ],  #2)-I3 + i3
        #      [Z_w,  None,     None,   -v3,    None,  r3,      None,    None,   None ],  #3)-v3 + r3*ir3 + jw*lamda2 (fy)
        #      [Z_w,  None,     None,   None,   None,  -L2,     None,    None,   lamda2], #4)lamda2 - L2*i3 =0
        #      [Z_w,  V2,       None,   -v2,    None,  None,    None,    None,   None],   #5)V2-v2=0
        #      [Z_w,  None,     -I2,    None,   None,  eq6_i,   None,    None,   None],   #6)-I2 -i3+i2+i1 =0
        #      [Z_w,  None,     None,   eq7_v,  None,  r2,      None,    None,   None],   #7)-v2 + r2*i2 + vc
        #      [Z_w,  None,     None,   C,      None,  None,    None,    -q,     None],   #8)Cvc - q = 0
        #      [Z_w,  None,     None,   None,   None,  -i2,     None,    None,   None],   #9)-i2 + jw*q =0 (fy)
        #      [Z_w,  eq10_V,   None,   -v1,    None,  None,    None,    None,   None],   #10)V2-V1-v1 =0
        #      [Z_w,  None,     -I1,    None,   None,  -i1,     None,    None,   None],   #11)-I1-i1 =0
        #      [Z_w,  None,     None,   -v1,    None,  r1,      None,    None,   None],   #12)-v1+r1*i1+jw*lamda1 =0 (fy)
        #      [Z_w,  None,     None,   None,   None,  -L1,     None,    None,   lamda1], #13)lamda1-L1*i1=0
        #      [Z_w,  -V1,      None,   None,   None,  None,    None,    None,   None],   #14)-V1 + [u] =0 (u)
        #      [Z_wr,  None,     None,   None,   None,  i3_r1,   None,    None,   None],  #15)i3a+i3b+i3c =0   
        #      [Z_wr,  None,     None,   None,   None,  i2_r1,  None,    None,   None],   #16)i2a+i2b+i2c=0 15
             
        #      #inverter eqns
        #      [Z_w,  V3,       None,   None,  -Id_ph, None,    None,    None,   None],   #17)V3 - vin = 0 16
        #      [Z_w,  None,     None,   None,  Id_ph,  None,    None,    None,   None],   #18)vin - fx = 0 (fy) 17
        #      [Z_wr, V3_r1,    None,   None,  None,   None,  None,    None,   None],     #16)i2a+i2b+i2c=0 15
        # ])    
        # fmt:on

        # fmt: off
        M = sps.bmat([
            #[w     V           I       v      i         q       lamda]
             [Z_w,  eq1_V,    None,   -v3,    None,      None,   None ],  #1)V3 - V2 - v3 = 0 
             [Z_w,  None,     -I3,    None,   i3,        None,   None ],  #2)-I3 + i3
             [Z_w,  None,     None,   v3,     r3,        None,   None ],  #3)v3 + r3*ir3 + jw*lamda2 (fy)
             [Z_w,  None,     None,   None,   -L2,       None,   lamda2], #4)lamda2 - L2*i3 =0
             [Z_w,  V2,       None,   -v2,    None,      None,   None],   #5)V2-v2=0
             [Z_w,  None,     -I2,    None,   eq6_i,     None,   None],   #6)-I2 -i3+i2+i1 =0
             [Z_w,  None,     None,   eq7_v,  r2,        None,   None],   #7)-v2 + r2*i2 + vc
             [Z_w,  None,     None,   C,      None,      -q,     None],   #8)Cvc - q = 0
             [Z_w,  None,     None,   None,   -i2,       None,   None],   #9)-i2 + jw*q =0 (fy)
             [Z_w,  eq10_V,   None,   -v1,    None,      None,   None],   #10)V2-V1-v1 =0
             [Z_w,  None,     -I1,    None,   -i1,       None,   None],   #11)-I1-i1 =0
             [Z_w,  None,     None,   -v1,    r1,        None,   None],   #12)-v1+r1*i1+jw*lamda1 =0 (fy)
             [Z_w,  None,     None,   None,   -L1,       None,   lamda1], #13)lamda1-L1*i1=0
             [Z_w,  -V2,      None,   None,   None,      None,   None],   #14)-V2 + [u] =0 (u)
            #  [Z_wr,  None,     None,   None,   i3_r1,     None,   None], #15)i3a+i3b+i3c =0   
            #  [Z_wr,  None,     None,   None,   i2_r1,    None,   None],    #16)i2a+i2b+i2c=0 15
            #  [Z_wr, V3_r1,    None,   None,   None,      None,    None, ],      #17)V3a+V3b+V3c=0 15
             [Z_w,  None,     I2,     None,   None,      None,   None],   #18) Ic = 0
        ])
        # fmt:on

        return M

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

        # fy update for eqn 3 : jw*lamda2
        idx_lamda2_start = self.var_offset["lamda"] + self.n_ph
        idx_lamda2_end = idx_lamda2_start + self.n_ph

        idx_eq3_start = self.n_ph + self.n_ph  # 1)V1-V2-v1 = 0  # 2)-I1 + i1 = 0
        idx_eq3_end = idx_eq3_start + self.n_ph

        fy[idx_eq3_start:idx_eq3_end] = 0 + 1j * (
            w * y[idx_lamda2_start:idx_lamda2_end]
        )

        # fy update for eqn 9 : jw*q
        idx_q_start = self.var_offset["q"]
        idx_q_end = idx_q_start + self.n_ph

        idx_eq9_start = 8 * self.n_ph
        idx_eq9_end = idx_eq9_start + self.n_ph

        fy[idx_eq9_start:idx_eq9_end] = 0 + 1j * (w * y[idx_q_start:idx_q_end])

        # fy update eqn 12 : jw*lamda1
        idx_lamda1_start = self.var_offset["lamda"]
        idx_lamda1_end = idx_lamda1_start + self.n_ph

        idx_eq12_start = 11 * self.n_ph
        idx_eq12_end = idx_eq12_start + self.n_ph

        fy[idx_eq12_start:idx_eq12_end] = 0 + 1j * (
            w * y[idx_lamda1_start:idx_lamda1_end]
        )

        # # fy update for eqn 17
        # Vdc = self.obj.Vdc
        # vsw = self.obj.Vsw
        # rsw = self.obj.rsw
        # vd = self.obj.Vd
        # rd = self.obj.rd

        # idx_i3_start = self.var_offset["i"] + 2*self.n_ph # i3 inverter side branch current
        # idx_din_start = self.var_offset["din"] # duty cycle

        # idx_eqn17_start = 15*self.n_ph + 2

        # for offset in range(self.n_ph):
        #     ix = y[idx_i3_start + offset, 0]
        #     dx = y[idx_din_start + offset, 0]
        #     print(f"type of ix:{type(ix)}")
        #     print(f"ix : {ix}")

        #     print(f"shape of ix:{ix.shape}")

        #     ix_mag = np.abs(ix)
        #     print(f"shape of ix_mag:{ix_mag.shape}")
        #     ix_angle = np.angle(ix, deg = False)
        #     print(f"shape of ix_angle:{ix_angle.shape}")

        #     if ix_mag*np.cos(ix_angle) >=0:
        #         fy[idx_eqn17_start + offset] = -((Vdc - vsw - rsw*ix)*dx - (vd + rd*ix)*(1 - dx))
        #     else:
        #         fy[idx_eqn17_start + offset] = -((Vdc + vd + rd*ix)*dx + (vsw - rsw*ix)*(1 - dx))

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
        lamda1_re_start_offset = self.var_offset["lamda"]
        lamda1_im_start_offset = self.var_offset_complex["lamda"]
        lamda2_re_start_offset = self.var_offset["lamda"] + self.n_ph
        lamda2_im_start_offset = self.var_offset_complex["lamda"] + self.n_ph

        # eq3 real
        eq3_re_start_row = 2 * self.n_ph  # after eqn set 1, 2

        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset

            lamda2_re_col_offset = lamda2_re_start_offset + offset
            lamda2_im_col_offset = lamda2_im_start_offset + offset
            lamda2_re = y_real[lamda2_re_col_offset, 0]
            lamda2_im = y_imag[lamda2_im_col_offset, 0]

            # derivatrive wrt w
            pd_fy_split[row, w_col_offset] = -lamda2_im

            # derivative wrt lamda2_re =0

            # derivative wrt lamda2_im
            pd_fy_split[row, self.num_vars + lamda2_im_col_offset] = -w

        # eq3 imaginary part
        eq3_im_start_row = self.num_eqns + eq3_re_start_row

        for offset in range(self.n_ph):
            row = eq3_im_start_row + offset

            lamda2_re_col_offset = lamda2_re_start_offset + offset
            lamda2_im_col_offset = lamda2_im_start_offset + offset
            lamda2_re = y_real[lamda2_re_col_offset, 0]
            lamda2_im = y_imag[lamda2_im_col_offset, 0]

            # derivative wrt w
            pd_fy_split[row, w_col_offset] = lamda2_re

            # derivative wrt lamda2_re
            pd_fy_split[row, lamda2_re_col_offset] = w

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

            lamda1_re_col_offset = lamda1_re_start_offset + offset
            lamda1_im_col_offset = lamda1_im_start_offset + offset
            lamda1_re = y_real[lamda1_re_col_offset, 0]
            lamda1_im = y_imag[lamda1_im_col_offset, 0]

            # derivatrive wrt w
            pd_fy_split[row, w_col_offset] = -lamda1_im

            # derivative wrt lamda1_re =0

            # derivative wrt lamda1_im
            pd_fy_split[row, self.num_vars + lamda1_im_col_offset] = -w

        # eq12 imaginary part
        eq12_im_start_row = self.num_eqns + eq12_re_start_row

        for offset in range(self.n_ph):
            row = eq12_im_start_row + offset

            lamda1_re_col_offset = lamda1_re_start_offset + offset
            lamda1_im_col_offset = lamda1_im_start_offset + offset
            lamda1_re = y_real[lamda1_re_col_offset, 0]
            lamda1_im = y_imag[lamda1_im_col_offset, 0]

            # derivative wrt w
            pd_fy_split[row, w_col_offset] = lamda1_re

            # derivative wrt lamda1_re
            pd_fy_split[row, lamda1_re_col_offset] = w

            # derivative wrt lamda1_im = 0

        # # eq17
        # eq17_re_start_row = 15*self.n_ph + 2
        # eq17_im_start_row = self.num_eqns + eq17_re_start_row
        # i3_real_start = self.var_offset["i"] + 2*self.n_ph # i3 inverter side branch current
        # i3_imag_start = self.var_offset_complex["i"]
        # din_real_start = self.var_offset["din"] # duty cycle
        # din_imag_start = self.var_offset_complex["din"]

        # Vdc = self.obj.Vdc
        # vsw = self.obj.Vsw
        # rsw = self.obj.rsw
        # vd = self.obj.Vd
        # rd = self.obj.rd

        # din_imag_start = self.var_offset_complex["din"]
        # for offset in range(self.n_ph):
        #     ix_real = y_real[i3_real_start + offset, 0]
        #     ix_imag = y_imag[i3_imag_start + offset, 0]
        #     ix = ix_real + 1j*ix_imag
        #     dx_real = y_real[din_real_start + offset, 0]
        #     dx_imag = y_real[din_imag_start + offset, 0]
        #     ix_mag = np.abs(ix)
        #     ix_angle = np.angle(ix, deg = False)

        #     if ix_mag*np.cos(ix_angle) >=0:
        #         #### derivative for real part of the expression:
        #         row  = eq17_re_start_row + offset
        #         # wrt ix_real
        #         ix_re_col_start = i3_real_start + offset
        #         pd_fy_split[row, ix_re_col_start] = (rsw*dx_real - rd + rd*dx_real)

        #         # wrt ix_imag
        #         ix_im_col_start = i3_imag_start + offset
        #         pd_fy_split[row, ix_im_col_start] = (-rsw*dx_imag - rd*dx_imag)

        #         # wrt dx_real
        #         dx_re_col_start = din_real_start + offset
        #         pd_fy_split[row, dx_re_col_start] = (-Vdc + vsw + rsw*ix_real + vd + rd*ix_real)

        #         # wrt dx_imag
        #         dx_im_col_start = din_imag_start + offset
        #         pd_fy_split[row, dx_im_col_start] = (-rsw*ix_imag - rd*ix_imag)

        #         print(f"shape of pd_fy : {pd_fy_split.shape}")
        #         ###### derivative for imaginary part of the expression
        #         row  = eq17_im_start_row + offset
        #         # wrt ix_real
        #         pd_fy_split[row, ix_re_col_start] = (rsw*dx_imag + rd*dx_imag)

        #         # wrt ix_imag
        #         pd_fy_split[row, ix_im_col_start] = (rsw*dx_real - rd + rd*dx_real)

        #         # wrt dx_real
        #         pd_fy_split[row, dx_re_col_start] = (rsw*ix_imag + rd*ix_imag)

        #         # wrt dx_imag
        #         pd_fy_split[row, dx_im_col_start] = (rsw*ix_real + vd + rd*ix_real)
        #     else:
        #         ##### derivative for real part of the expression
        #         row  = eq17_re_start_row + offset
        #         # wrt ix_real
        #         ix_re_col_start = i3_real_start + offset
        #         pd_fy_split[row, ix_re_col_start] = (rd*dx_real - rsw + rsw*dx_real)

        #         # wrt ix_imag
        #         ix_im_col_start = i3_imag_start + offset
        #         pd_fy_split[row, ix_im_col_start] = (-rd*dx_imag - rsw*dx_imag)

        #         # wrt dx_real
        #         dx_re_col_start = din_real_start + offset
        #         pd_fy_split[row, dx_re_col_start] = (-Vdc - vd + rd*ix_real - vsw + rsw*ix_real)

        #         # wrt dx_imag
        #         dx_im_col_start = din_imag_start + offset
        #         pd_fy_split[row, dx_im_col_start] = (rd*ix_imag - rsw*ix_imag)

        #         ##### derivative for imaginary part of the expression
        #         row = eq17_im_start_row + offset
        #         #wrt ix_real
        #         pd_fy_split[row, ix_re_col_start] =(rd*dx_imag + rsw*dx_imag)

        #         #wrt ix_imag
        #         pd_fy_split[row, ix_im_col_start] = (rd*dx_real - rsw + rsw*dx_real)

        #         # wrt dx_real
        #         pd_fy_split[row, dx_re_col_start] = (rd*ix_imag + rsw*ix_imag)
        #         # wrt dx_imag
        #         pd_fy_split[row, dx_im_col_start] = (-Vdc - vd + rd*ix_real - vsw + rsw*ix_real)

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

    def _lcl(self):
        # fmt: off
        # b1. LCL filter
        b1_var_offset = {
            "V": 0,  # [V_a, V_b, V_c] grid side node voltage
            "Vcf": self.n_ph,  # [Vcf_a, Vcf_b, Vcf_c] capacitor node voltage
            "Vinf": 2 * self.n_ph,  # [Vinf_a, Vinf_b, Vin_c] inverter side node voltage
            "Vab" : 3 * self.n_ph,
            "Vbc" : 1 + 3 * self.n_ph,
            "I":  2 + 3 * self.n_ph,  # [I_a, I_b, I_c] grid side node injection
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
        b1_num_vars = b1_var_offset["lamda_in"] + self.n_ph
        # fmt: on

        # coefficient of eqns for capacitor
        # C = sps.bmat([[Z_ph, Z_ph, self.C_mat, Z_ph]])
        C = self.C_mat
        # rg = sps.bmat([[self.r1_mat, Z_ph, Z_ph]])
        rg = self.r1_mat
        # rc = sps.bmat([[Z_ph, self.r2_mat, Z_ph]])
        rc = self.r2_mat
        # r_in = sps.bmat([[Z_ph, Z_ph, self.r3_mat]])
        r_in = self.r3_mat
        # Lg = sps.bmat([[self.L1_mat, Z_ph, Z_ph]])
        Lg = self.L2_mat
        # L_in = sps.bmat([[Z_ph, Z_ph, self.L2_mat]])
        L_in = self.L1_mat
        Id_row = np.array([[1, 1, 1]])
        Id_ph = sps.identity(self.n_ph, format="coo")
        Id_one = sps.identity((1), dtype=float)
        ab = np.array([[1, -1, 0]])
        bc = np.array([[0, 1, -1]])

        # V1 - Vgf = 0
        # V2 - Vcf = 0
        # V3 - Vinf = 0
        # I3 - Iinf = 0
        # fmt: off
        # block for LCL filter
        b1_Mmat = sps.bmat(
            [
                # V,     Vcf,    Vinf,   Vab,     Vbc,   I,      Iinf,   vgf,   vcf,  vinf, vccf, igf,  icf,   iinf,  qf,    lamda_g, lamda_in
                [-Id_ph, Id_ph,  None,   None,    None,  None,   None,  -Id_ph, None, None, None, None, None,  None,  None,  None,    None, ],  # 1) Vcf - V - vgf=0 (3)
                [ None,  None,   None,   None,    None,  -Id_ph, None,   None,  None, None, None,-Id_ph,None,  None,  None,  None,    None, ],  # 2)-I-igf=0 (3)
                [ None,  None,   None,   None,    None,  None,   None,  -Id_ph, None, None, None, rg,   None,  None,  None,  None,    None, ],  # 3) -vgf + rg*igf + d(lamda_g)/dt=0 (K) (3)
                [ None,  None,   None,   None,    None,  None,   None,   None,  None, None, None,-Lg,   None,  None,  None,  Id_ph,   None, ],  # 4)lamda_g - Lg*igf=0 (3)
                [ None,  Id_ph,  None,   None,    None,  None,   None,   None, -Id_ph,None, None, None, None,  None,  None,  None,    None, ],  # 5)Vcf-vcf=0 (3)
                [ None,  None,   None,   None,    None,  None,   None,   None,  None, None, None, Id_ph,Id_ph,-Id_ph, None,  None,    None, ],  # 6) -iinf+icf+igf=0 (3)
                [ None,  None,   None,   None,    None,  None,   None,   None, -Id_ph,None, Id_ph,None, rc,    None,  None,  None,    None, ],  # 7) -vcf + rc*icf + vccf=0 (3)
                [ None,  None,   None,   None,    None,  None,   None,   None,  None, None, C,    None, None,  None, -Id_ph, None,    None, ],  # 8) C * vccf - qf = 0 (3)
                [ None,  None,   None,   None,    None,  None,   None,   None,  None, None, None, None,-Id_ph, None,  None,  None,    None, ],  # 9)-icf + d(qf)/dt=0 (K) (3)
                [ None, -Id_ph,  Id_ph,  None,    None,  None,   None,   None,  None,-Id_ph,None, None, None,  None,  None,  None,    None, ],  # 10)Vinf-Vcf-vinf=0 (3)
                [ None,  None,   None,   None,    None,  None,  -Id_ph,  None,  None, None, None, None, None,  Id_ph, None,  None,    None, ],  # 11)-Inf+iinf=0 (3)
                [ None,  None,   None,   None,    None,  None,   None,   None,  None,-Id_ph,None, None, None,  r_in,  None,  None,    None, ],  # 12)-vinf+r_in*iinf + d(lamda_in)/dt=0 (K) (3)
                [ None,  None,   None,   None,    None,  None,   None,   None,  None, None, None, None, None, -L_in,  None,  None,    Id_ph,],  # 13)lamda_inf -L_in*iinf=0 (3)
                [ None,  None,   ab,    -Id_one,  None,  None,   None,   None,  None, None, None, None, None,  None,  None,  None,    None, ],  # 14) -Vab + Va - Vb = 0 (1)
                [ None,  None,   bc,     None,  -Id_one,  None,  None,   None,  None, None, None, None, None,  None,  None,  None,    None, ],  # 15) -Vbc + Vb - Vc = 0 (1)
 
                [ None,  None,   None,    None,   None,  None,   None,   None,  None, None, None, None, Id_row,None,  None,  None,    None, ],  # 16) icf_a+icf_b+icf_c=0 (1)
              # [ None,  None,  None,   None,  None,  None, None, None,  None, None, None, None, None,  iinf, None, None, None, ],  # 15) iinf_a + iinf_b + iinf_c = 0
            ]
        )
        # fmt: on
        assert b1_Mmat.shape[0] == 42
        assert b1_Mmat.shape[1] == 47
        assert b1_Mmat.shape[1] == b1_num_vars

        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        Zero = sps.lil_matrix((1, Z_ph.shape[0]), dtype=float)
        Z_one = np.zeros((1, 1))

        # fmt: off
        b1_Kmat = sps.bmat(
            [
                # V,    Vcf,  Vinf, Vab ,  Vbc,   I,    Icf,  vgf,  vcf,  vinf, vccf, igf,  icf,  iinf, qf,   lamda_g,lamda_in
                [ Z_ph, Z_ph, None, None,  None,  None, None, Z_ph, None, None, None, None, None, None, None, None, None, ],  # 1)Vcf-V-vgf=0
                [ None, None, None, None,  None,  Z_ph, None, None, None, None, None, Z_ph, None, None, None, None, None, ],  # 2)-I-igf=0
                [ None, None, None, None,  None,  None, None, Z_ph, None, None, None, Z_ph, None, None, None, Id_ph,None, ],  # 3)-vgf + rg*igf + d(lamda_g)/dt=0 (K)
                [ None, None, None, None,  None,  None, None, None, None, None, None, Z_ph, None, None, None, None, None, ],  # 4)lamda_g - Lg*igf=0
                [ None, None, None, None,  None,  None, None, None, Z_ph, None, None, None, None, None, None, None, None, ],  # 5)Vcf-vcf=0
                [ None, None, None, None,  None,  None, Z_ph, None, None, None, None, Z_ph, Z_ph, Z_ph, None, None, None, ],  # 6)-Icf-iinf+icf+igf=0
                [ None, None, None, None,  None,  None, None, None, Z_ph, None, Z_ph, None, Z_ph, None, None, None, None, ],  # 7)-vcf+rc*icf+vcc=0
                [ None, None, None, None,  None,  None, None, None, None, None, Z_ph, None, None, None, Z_ph, None, None, ],  # 8)C*vcc-q=0
                [ None, None, None, None,  None,  None, None, None, None, None, None, None, Z_ph, None, Id_ph,None, None, ],  # 9)-icf + d(qf)/dt=0
                [ None, Z_ph, Z_ph, None,  None,  None, None, None, None, Z_ph, None, None, None, None, None, None, None, ],  # 10)Vinf-Vcf-vinf=0
                [ None, None, None, None,  None,  None, None, None, None, None, None, None, None, Z_ph, None, None, None, ],  # 11)-Inf-iinf=0
                [ None, None, None, None,  None,  None, None, None, None, Z_ph, None, None, None, Z_ph, None, None, Id_ph, ],  # 12)-vinf+r_in*iinf + d(lamda_in)/dt=0 (K)
                [ None, None, None, None,  None,  None, None, None, None, None, None, None, None, Z_ph, None, None, Z_ph, ],  # 13)lamda_in - L_in*iinf=0
                [ Zero, None, None, Z_one,  None,  None, None, None, None, None, None, None, None, None, None,  None, None, ],  # 14) Vab - Va + Vb = 0 (1)
                [ Zero, None, None, None,  Z_one,  None, None, None, None, None, None, None, None, None, None,  None, None, ],  # 15) Vbc - Vb + Vc = 0 (1)

               [ None,  None, None, None,  None,  None, None, None,  None, None, None, None,Zero, None, None,  None, None],  # 16) icf_a+icf_b+icf_c=0 (1)
              # [ None, None, None,   None,  None,  None, None, None,  None, None, None, None, None,  iinf, None, None, None, ],  # 15) iinf_a + iinf_b + iinf_c = 0
            ]
        )
        # fmt: on

        assert b1_Mmat.shape == b1_Kmat.shape

        nrow, ncol = b1_Mmat.shape

        def _get_fy_dynamic_b1(t, y) -> np.ndarray:
            fy1 = np.zeros((self.b1_Mmat.shape[0], 1), dtype=float)
            return fy1

        def get_u1(t: float, y) -> np.ndarray:
            u1 = np.zeros((nrow, 1), dtype=float)
            return u1

        assert not hasattr(self, "b1_Mmat") or self.b1_Mmat is None, (
            f"self.b1_Mmat should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.b1_Mmat = b1_Mmat
        self.b1_Kmat = b1_Kmat
        self.var_offset_dynamic_b1 = b1_var_offset
        self.b1_num_vars = b1_num_vars
        self._get_fy_dynamic_b1 = _get_fy_dynamic_b1
        self._get_u1_dynamic = get_u1

    def _mnt(self):
        # b2. Measurement and Transformation Block
        b2_var_offset = {
            "Vc": 0,  # [Vc_a, Vc_b, Vc_c]
            "vcc": self.n_ph,  # [vcc_a, vcc_b, vcc_c]
            "Vc_qd": 2 * self.n_ph,  # [Vc_q, Vc_d]
            "vcc_qd": 2 * self.n_ph + 2,  # [vcc_q, vcc_d]
            "ig": 2 * self.n_ph + 4,  # [ig_a, ig_b, ig_c]
            "iin": 3 * self.n_ph + 4,  # [iin_a, iin_b, iin_c]
            "ig_qd": 4 * self.n_ph + 4,  # [ig_q, ig_d]
            "iin_qd": 4 * self.n_ph + 6,  # [iin_q, iin_d]
        }
        b2_num_vars = b2_var_offset["iin_qd"] + 2

        # coefficients for eqns
        q = np.array([[1, 0]])
        d = np.array([[0, 1]])
        Zrow = sps.lil_matrix((1, self.n_ph), dtype=float)

        # Vg - V1 = 0
        # Vc - V2 = 0
        # fmt:off
        # Measurement and Transformation Block
        b2_Mmat = sps.bmat([
            #[Vc,  vcc,  Vc_qd,  vcc_qd,  ig,    iin,    ig_qd,  iin_qd ]            
            [Zrow, None,  q,     None,   None,  None,   None,   None,  ], # 1) Vc_q-K*Vc=0 (fy)   (1)
            [None, None,  d,     None,   None,  None,   None,   None,  ], # 2) Vc_d-K*Vc=0 (fy)   (1)
            [None, None,  None,  None,   Zrow,  None,   q,      None,  ], # 3) ig_q-K*ig=0 (fy)   (1) #c
            [None, None,  None,  None,   None,  None,   d,      None,  ], # 4) ig_d-K*ig=0 (fy)   (1) #c
            [None, None,  None,  None,   None,  Zrow,   None,   q,    ],# 5) iin_q-K*iin=0 (fy) (1)
            [None, None,  None,  None,   None,  None,   None,   d,    ],# 6) iin_d-K*iin=0 (fy)(1)
            [None, Zrow,  None,  q,      None,  None,   None,   None, ], # 7) vcc_q-K*vcc=0 (fy)(1)
            [None, Zrow,  None,  d,      None,  None,   None,   None, ], # 8) vcc_d-K*vcc=0 (fy)(1)
        ])
        # fmt: on
        print(f"b2_Mmat.shape: {b2_Mmat.shape}")

        assert b2_Mmat.shape[0] == 8
        assert b2_Mmat.shape[1] == 20
        assert b2_Mmat.shape[1] == b2_num_vars

        b2_Kmat = sps.lil_matrix(b2_Mmat.shape, dtype=float)

        def _get_fy_dynamic_b2(t, y) -> np.ndarray:
            fy2 = np.zeros((self.b2_Mmat.shape[0], 1), dtype=float)

            idx_theta = self.var_offset_dynamic["theta"]
            idx_V_ca = self.var_offset_dynamic["Vc"]  # a phase
            idx_V_cb = self.var_offset_dynamic["Vc"] + 1  # b phase
            idx_V_cc = self.var_offset_dynamic["Vc"] + 2  # c phase
            idx_ig_a = self.var_offset_dynamic["ig"]  # a phase
            idx_ig_b = self.var_offset_dynamic["ig"] + 1  # bphase
            idx_ig_c = self.var_offset_dynamic["ig"] + 2  # c phase
            idx_iin_a = self.var_offset_dynamic["iin"]  # a phase
            idx_iin_b = self.var_offset_dynamic["iin"] + 1  # bphase
            idx_iin_c = self.var_offset_dynamic["iin"] + 2  # c phase
            idx_vcc_a = self.var_offset_dynamic["vcc"]  # a phase
            idx_vcc_b = self.var_offset_dynamic["vcc"] + 1  # b phase
            idx_vcc_c = self.var_offset_dynamic["vcc"] + 2  # c phase

            theta = y[idx_theta]
            V_ca = y[idx_V_ca]
            V_cb = y[idx_V_cb]
            V_cc = y[idx_V_cc]
            ig_a = y[idx_ig_a]
            ig_b = y[idx_ig_b]
            ig_c = y[idx_ig_c]
            iin_a = y[idx_iin_a]
            iin_b = y[idx_iin_b]
            iin_c = y[idx_iin_c]
            vcc_a = y[idx_vcc_a]
            vcc_b = y[idx_vcc_b]
            vcc_c = y[idx_vcc_c]

            # fmt: off

            # eq1:# Vcq - K*Vc
            idx_eq1 = 0
            fy2[idx_eq1] = (
                (-2 / 3)
                * (
                    V_ca * np.cos(theta)
                    + V_cb * np.cos(theta - 2 * np.pi / 3)
                    + V_cc * np.cos(theta + 2 * np.pi / 3)
                )
                / self.Vqd_b
            )

            # eq2 : Vcd-K*Vc
            idx_eq2 = 1
            fy2[idx_eq2] = (
                (-2 / 3)
                * (
                    V_ca * np.sin(theta)
                    + V_cb * np.sin(theta - 2 * np.pi / 3)
                    + V_cc * np.sin(theta + 2 * np.pi / 3)
                )
                / self.Vqd_b
            )

            # eq3 : ig_q-K*ig = 0
            idx_eq3 = 2
            fy2[idx_eq3] = (
                (-2 / 3)
                * (
                    ig_a * np.cos(theta)
                    + ig_b * np.cos(theta - 2 * np.pi / 3)
                    + ig_c * np.cos(theta + 2 * np.pi / 3)
                )
                / self.Iqd_b
            )

            # eq4 : ig_d-K*ig
            idx_eq4 = 3
            fy2[idx_eq4] = (
                (-2 / 3)
                * (
                    ig_a * np.sin(theta)
                    + ig_b * np.sin(theta - 2 * np.pi / 3)
                    + ig_c * np.sin(theta + 2 * np.pi / 3)
                )
                / self.Iqd_b
            )

            # eq5 : iin_q-K*iin
            idx_eq5 = 4
            fy2[idx_eq5] = (-2 / 3) * (
                iin_a * np.cos(theta)
                + iin_b * np.cos(theta - 2 * np.pi / 3)
                + iin_c * np.cos(theta + 2 * np.pi / 3)
            ) / self.Iqd_b

            # eq6 : iin_d-K*iin
            idx_eq6 = 5
            fy2[idx_eq6] = (-2 / 3) * (
                iin_a * np.sin(theta)
                + iin_b * np.sin(theta - 2 * np.pi / 3)
                + iin_c * np.sin(theta + 2 * np.pi / 3)
            ) / self.Iqd_b

            # eqn7 : vcc_q - K*vcc
            idx_eq7 = 6
            fy2[idx_eq7] = (-2 / 3) * (
                vcc_a * np.cos(theta)
                + vcc_b * np.cos(theta - 2 * np.pi / 3)
                + vcc_c * np.cos(theta + 2 * np.pi / 3)
            ) / self. Vqd_b

            # eqn8 : vcc_d - K*vcc
            idx_eq8 = 7
            fy2[idx_eq8] = (-2 / 3) * (
                vcc_a * np.sin(theta)
                + vcc_b * np.sin(theta - 2 * np.pi / 3)
                + vcc_c * np.sin(theta + 2 * np.pi / 3)
            ) / self.Vqd_b

            # fmt: on

            return fy2

        def get_u2(t, y) -> np.ndarray:
            u2 = np.zeros((self.b2_Mmat.shape[0], 1), dtype=float)
            return u2

        assert not hasattr(self, "b2_Mmat") or self.b2_Mmat is None

        self.b2_Mmat = b2_Mmat
        self.b2_Kmat = b2_Kmat
        self.var_offset_dynamic_b2 = b2_var_offset
        self.b2_num_vars = b2_num_vars
        self._get_fy_dynamic_b2 = _get_fy_dynamic_b2
        self._get_u2_dynamic = get_u2

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

        return Vbrf_Mmat, Vbrf_Kmat, Vbrf_var_offset, Vbrf_num_vars, None, None

    def _ibrf(self):
        ########## Band reject filter for current
        # fmt: off
        Ibrf_var_offset = {
            "I_qd": 0,  # [I_q, I_d] input with both dc and oscillating component with frequency 2w
            "Ibrf1_qd": self.n_varqd,  # [Ibrf1_q, Ibrf1_d] intermediate output of filter
            "Ibrf2_qd": 2 * self.n_varqd,  # [Ibrf2_q, Ibrf2_d] intermediate output of filter
            "Ibrf3_qd": 3 * self.n_varqd,  # [Ibrf3_q, Ibrf3_d] intermdediate output of filter
            "I_qd_dc": 4 * self.n_varqd,  # [I_q_dc, I_d_dc] output of Band reject filter- dc component of pos/neg sequence input
        }
        # fmt: on
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
        self._get_fy_dynamic_Ibrf = _get_fy_dynamic_Ibrf
        self._get_u_ibrf_dynamic = get_u_ibrf

    def _pw_droop(self):
        # fmt:off
        b3_var_offset = {
            "w": 0,
            "w_pu": 1,
            "Vdr_qd": 2,  # [Vdr_q,Vdr_d] # measured at point of measurement
            "idr_qd": 2 + self.n_varqd,  # [idr_q, idr_d] # measured at point of measurement
            "Pe": 2 + 2 * self.n_varqd,  # Active power
            "Pe_f": 3 + 2 * self.n_varqd,  # Active power filtered
            "theta": 4 + 2 * self.n_varqd,  # theta
        }
        # fmt:on
        b3_num_vars = b3_var_offset["theta"] + 1

        # coefficients for primary control block
        # to be written
        kw = self.obj.kw
        kq = self.obj.kq
        tfp = self.obj.taus

        Id_one = sps.identity((1), dtype=float)
        Kw_one = kw * Id_one * (self.Vqd_b * self.Iqd_b / self.Pb)

        Tfp_one = 1 / tfp * Id_one
        Z1 = sps.lil_matrix((1, 1), dtype=float)
        wb = self.w_b * Id_one
        Z_qd = np.array([[0, 0]])

        # P-w droop block
        # fmt:off
        b3_Mmat = sps.bmat([
            #w,     w_pu,      Vdr_qd, idr_qd, Pe,        Pe_f,    theta           
            [None,  None,      Z_qd,   Z_qd,   Id_one,    None,    None ], #1) Pe-(Vdr_q * idr_q + Vdr_d * idr_d)=0 (fy)   (1) #c
            [None,  Id_one,    None,   None,   None,      Kw_one,  None ], #2) w_pu - wnom - kw * (Pref - Pef)=0 (u) (1) #c
            [None,  None,      None,   None,  -Tfp_one,   Tfp_one, None ], #3) d(Pe_f)/dt + Pe_f/tauP - Pe/tauP=0 (K)(1) #c
            [None,  -wb,       None,   None,   None,      None,    Z1   ], #4) d(theta)/dt - w=0 (K) (1) #c
            [Id_one,-wb,       None,   None,   None,      None,    None ]  #5) w - w_pu * w_b = 0
        ])

        # # Q-V droop
        # [None,      Z_qd,    None,    None,    Z_qd,   None,      None,    Id_one,  None,    None ], #5) Qe-(Vdr_q * idr_d - Vdr_d * idr_q)=0 (fy)   (1) #c
        # [None,      None,    Id_one,  None,    None,   None,      None,    None,    Kq_one,  None ], #6) Vref_amp - Vnom - kq * (Qref - Qef)=0 (u) (1) #c
        # [None,      None,    None,    None,    None,   None,      None,   -Tfq_one, Tfq_one, None ], #7) d(Qe_f)/dt + Qe_f/tauq - Qe/tauq=0 (K)(1) #c
        # [None,      None,    None,    Id_ph,   None,   None,      None,    None,    None,    Z_col], #8) Vrefc - Vref_amp * cos(theta*) = 0 (fy) (3)

        print(f"b3_Mmat.shape: {b3_Mmat.shape}")

        assert b3_Mmat.shape[0] == 5
        assert b3_Mmat.shape[1] == 9, f"expected: 8, got: {b3_Mmat.shape[1]}"
        assert b3_Mmat.shape[1] == b3_num_vars

        # fmt: off
        b3_Kmat = sps.bmat([
            #w,     w_pu,  Vdr_qd, idr_qd, Pe,    Pe_f,  theta           
            [None,  None,  Z_qd,   Z_qd,   Z1,    None,  None ], #1) Pe-(Vdr_q * idr_q + Vdr_d * idr_d)=0 (fy)   (1) 
            [Z1,    None,  None,   None,   None,  Z1,    None ], #2) w_pu - wnom - kw * (Pref-Pef)=0 (u) (1) 
            [None,  None,  None,   None,   None,  Id_one,None ], #3) d(Pe_f)/dt + Pe_f/tauP - Pe/tauP=0 (K)(1) 
            [None,  None,  None,   None,   None,  None,  Id_one], #4) d(theta)/dt - w=0 (K) (1) 
            [None,  Z1,    None,   None,   None,  None,  None ]  #5) w - w_pu * w_b = 0
        ])
        # fmt:on

        assert b3_Mmat.shape == b3_Kmat.shape

        def _get_fy_dynamic_b3(t, y) -> np.ndarray:
            # fy3 :Droop control block b3
            # non-linearity in eqns 1, 5 and 10
            fy3 = np.zeros((self.b3_Mmat.shape[0], 1), dtype=float)
            idx_vq = self.var_offset_dynamic["Vdr_qd"]
            idx_vd = self.var_offset_dynamic["Vdr_qd"] + 1
            idx_iq = self.var_offset_dynamic["idr_qd"]
            idx_id = self.var_offset_dynamic["idr_qd"] + 1
            vq = y[idx_vq]
            vd = y[idx_vd]
            iq = y[idx_iq]
            id = y[idx_id]

            # eq1 : Pe - (vq*iq + vd*id) = 0
            idx_eq1_start = 0
            fy3[idx_eq1_start] = -(3 / 2) * (vq * iq + vd * id)
            # print(f"Pe calculate in fy:{-(3 / 2) * (vq * iq + vd * id)}")
            return fy3

        def get_u3(t, y) -> np.ndarray:
            u3 = np.zeros((self.b3_Mmat.shape[0], 1), dtype=float)

            # Pref, Pe
            Pref_pu = self.Pref_total / self.Pb
            Pef_pu = y[self.var_offset_dynamic["Pe_f"]]
            print(f"Pref_pu:{Pref_pu}, Pef_pu:{Pef_pu}")

            # eqn2: p-w droop:w-wnom-Kw(Pref - Pe_f)
            # wnom is the one given in input and so should Vnom for a network
            kw = self.obj.kw
            idx_eqn2 = 1
            u3[idx_eqn2] = -1 - kw * self.Pref_total / self.Pb
            return u3

        assert not hasattr(self, "b3_Mmat") or self.b3_Mmat is None, (
            f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.b3_Mmat = b3_Mmat
        self.b3_Kmat = b3_Kmat
        self.var_offset_dynamic_b3 = b3_var_offset
        self.b3_num_vars = b3_num_vars
        self._get_fy_dynamic_b3 = _get_fy_dynamic_b3
        self._get_u3_dynamic = get_u3

    def _qv_droop(self):
        # fmt:off
        qv_var_offset = {
            "V_qvdroop_qd": 0,  # [V_qvdroop_q,V_qvdroop_d] # measured at point of measurement
            "i_qvdroop_qd": self.n_varqd,  # [i_qvdroop_q, i_qvdroop_d] # measured at point of measurement
            "Qe": 2 * self.n_varqd,  # Reactive power computed
            "Qe_f": 1 + 2 * self.n_varqd,  #  Computed reactive power filtered
            "Vref_qd": 2 + 2 * self.n_varqd,  # ref amplitude computed Vref_q = amp. Vref_d =0
        }
        # fmt:on
        qv_num_vars = qv_var_offset["Vref_qd"] + self.n_varqd

        # coefficients for the matrix
        Id_one = sps.identity((1), dtype=float)
        tfp = (
            self.obj.taus
        )  # to be checked for values in input or make another for QV droop
        Tfq_one = 1 / tfp * Id_one
        q = np.array([[1, 0]])
        d = np.array([[0, 1]])
        Z_qd = np.array([[0, 0]])
        Z1 = sps.lil_matrix((1, 1), dtype=float)
        eq3_kq = self.obj.kq * Id_one

        # fmt:off
        qv_Mmat = sps.bmat([
            # V_qvdroop_qd, i_qvdroop_qd,  Qe,       Qe_f,    Vref_qd
            [Z_qd,          Z_qd,          Id_one,   None,    None   ], #1) Qe-(Vdr_q * idr_d - Vdr_d * idr_q)=0 (fy)   (1)
            [None,          None,         -Tfq_one,  Tfq_one, None   ], #2) d(Qe_f)/dt + Qe_f/tauq - Qe/tauq=0 (K)   (1) 
            [None,          None,          None,     eq3_kq,  q      ], #3) Vref_q - Vnom - kq * ( Qref - Qe_f) = 0 (fy) (u) (1)
            [None,          None,          None,     None,    d      ], #4) Vref_d = 0 (1)
        ])

        qv_Kmat = sps.bmat([
            # V_qvdroop_qd, i_qvdroop_qd,  Qe,       Qe_f,    Vref_qd
            [Z_qd,          Z_qd,          Z1,       None,    None   ], #1) Qe-(Vdr_q * idr_d - Vdr_d * idr_q)=0 (fy)   (1)
            [None,          None,          None,     Id_one,  None   ], #2) d(Qe_f)/dt + Qe_f/tauq - Qe/tauq=0 (K)   (1) 
            [None,          None,          None,     None,    Z_qd   ], #3) Vref_q - Vnom - kq * ( Qref - Qe_f) = 0 (fy) (u) (1)
            [None,          None,          None,     None,    Z_qd   ], #4) Vref_d = 0 (1)
        ])
        # fmt: on

        assert qv_Mmat.shape == qv_Kmat.shape, f"{qv_Mmat.shape} == {qv_Kmat.shape}"
        assert qv_Mmat.shape[1] == qv_num_vars

        def _get_fy_qv(t, y) -> np.ndarray:
            fy_qv_dr = np.zeros((self.qv_Mmat.shape[0], 1), dtype=float)

            # 1) Qe-(Vdr_q * idr_d - Vdr_d * idr_q)=0 (fy)   (1)
            idx_eqn1 = 0
            idx_Vdr_qd = self.var_offset_dynamic_qv["V_qvdroop_qd"]
            idx_idr_qd = self.var_offset_dynamic_qv["i_qvdroop_qd"]
            Vq = y[idx_Vdr_qd]
            Vd = y[idx_Vdr_qd + 1]
            iq = y[idx_idr_qd]
            id = y[idx_idr_qd + 1]
            Qe = Vq * id - Vd * iq
            fy_qv_dr[idx_eqn1] = -Qe

            return fy_qv_dr

        def get_u_qv(t, y) -> np.ndarray:
            u_qv_dr = np.zeros((self.qv_Mmat.shape[0], 1), dtype=float)
            kq = self.obj.kq

            # Qref, Qe
            Qref_pu = self.Qref_total / self.Pb
            Qef_pu = y[self.var_offset_dynamic["Qe_f"]]
            print(f"Qref_pu:{Qref_pu}, Qef_pu:{Qef_pu}")

            # Vnom_peak = self.nominal_voltage_rms * np.sqrt(2) # Vnom peak
            Vnom_peak_pu = self.nominal_voltage_rms * np.sqrt(2) / self.Vqd_b  # Vnom peak
            # u_qv_dr[2] = (-Vnom_peak - kq * self.Qref_total) / self.Vqd_b
            u_qv_dr[2] = -Vnom_peak_pu - kq * self.Qref_total / (self.Vqd_b * self.Iqd_b)

            return u_qv_dr

        assert not hasattr(self, "qv_Mmat") or self.qv_Mmat is None, (
            f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.qv_Kmat = qv_Kmat
        self.qv_Mmat = qv_Mmat
        self.var_offset_dynamic_qv = qv_var_offset
        self.qv_num_vars = qv_num_vars
        self._get_fy_dynamic_qv = _get_fy_qv
        self._get_u_qv_dynamic = get_u_qv

    def _voltage_controller(self):
        # b4. Voltage Controller
        # fmt:off
        b4_var_offset = {
            "Vvr_cqd": 0,  # [Vvr_cq, Vvr_cd] measure capacitor M&T block
            "Vcref_qd": self.n_varqd,  # [Vcref_q, Vcref_d] obtained from Q-V droop or system constant
            "vvr_ccqd": 2 * self.n_varqd, # [vvr_ccq, vvr_ccd]
            "iref_qd": 3 * self.n_varqd,  # [iref_q, iref_d] reference calculated to maintain capacitor voltage
            "iref_qd_f": 4 * self.n_varqd,  # [iref_q_f, iref_d_f] ref current after antiwindup (final output of VC)
            "iff_qd": 5 * self.n_varqd,  # [iff_q, iff_d] feedforward
            "ivr_cqd": 6 * self.n_varqd,  # [ivr_cq, ivr_cd] speed current across capacitor
            "ivr_gqd": 7 * self.n_varqd,  # [ivr_gq, ivr_gd] grid current from M&T block
            "z_qd": 8 * self.n_varqd,  # [z_q, z_d]  intermediate(iref_qd_f - iff_qd)
            "z_qdf": 9 * self.n_varqd,  # [z_qf, z_df] after antiwindup
            "zvr_qdo": 10 * self.n_varqd,  # [zvr_qo, zvr_do] output of integrator of voltage regulator
        }
        b4_num_vars = b4_var_offset["zvr_qdo"] + 2
        # fmt:on

        # coefficients for the eqns
        q = np.array([[1, 0]])
        d = np.array([[0, 1]])
        Kpv_q = self.obj.Kpv * q  # coefficient for vcref_q
        Kpv_d = self.obj.Kpv * d  # coefficient for vcref_q
        inv_tauiv_q = (1 / self.obj.tauiv) * q
        inv_tauiv_d = (1 / self.obj.tauiv) * d
        Z_qd = np.array([[0, 0]])

        # fmt: off
        b4_Mmat = sps.bmat([
            #Vvr_cqd,   Vcref_qd,  vvr_ccqd   iref_qd, iref_qd_f,iff_qd, ivr_cqd, ivr_gqd, z_qd, z_qdf        zvr_qdo           
            [Kpv_q,    -Kpv_q,     None,      q,       None,    -q,      None,    None,    None, None,       -q        ], #1)iref_q - kpv(Vcref_q-Vvrc_q)-iffq-zvr_qo=0 (1) #c
            [Kpv_d,    -Kpv_d,     None,      d,       None,    -d,      None,    None,    None, None,       -d        ], #2)iref_d - kpv(Vcref_d-Vvrc_d)-iffd-zvr_do=0 (1) #c
            [None,      None,      None,      None,    q,        None,   None,    None,    None, None,        None,    ], #3)iref_qf - C(., Imx)=0 (fy) (1) #c
            [None,      None,      None,      None,    d,        None,   None,    None,    None, None,        None,    ], #4)iref_df - C(., Imx)=0 (fy) (1) #c
            [None,      None,      None,      None,    None,     q,     -q,      -q,       None, None,        None,    ], #5)iff_q - ivr_cq - ivr_gq=0 (1)
            [None,      None,      None,      None,    None,     d,     -d,      -d,       None, None,        None,    ], #6)iff_d - ivr_cd - ivr_gd=0 (1)
            [None,      None,      None,      None,    -q,       q,      None,    None,    q,    None,        None,    ], #7)z_q - iref_qf + iff_q=0 (1)
            [None,      None,      None,      None,    -d,       d,      None,    None,    d,    None,        None,    ], #8)z_d - iref_df + iff_d=0 (1)
            [None,      None,      None,      None,    None,     None,   None,    None,    None, q,           None,    ], #9)z_qf - C(.Imx)=0 (fy) (1)
            [None,      None,      None,      None,    None,     None,   None,    None,    None, d,           None,    ], #10)z_df - C(.Imx)=0 (fy) (1)
            [None,      None,      None,      None,    None,     None,   None,    None,    None,-inv_tauiv_q, inv_tauiv_q,], #11)d(zvr_qo)/dt - 1/tau*zq_f + 1/tau*zvr_qo (K) (1)
            [None,      None,      None,      None,    None,     None,   None,    None,    None,-inv_tauiv_d, inv_tauiv_d,], #12)d(zvr_do)/dt - 1/tau*zd_f + 1/tau*zvr_do (K) (1)
            [None,      None,      Z_qd,      None,    None,     None,   q,       None,    None, None,        None,     ], #13)ivr_cq - w*C*vvr_ccd = 0 (fy) (1)
            [None,      None,      Z_qd,      None,    None,     None,   d,       None,    None, None,        None,     ], #14)ivr_cd + w*C*vvr_ccq = 0 (fy) (1)
            # [None,      -q,        None,      None,    None,     None,   None,    None,    None, None,        None,     ], #15) -Vcref_q + u = 0 (u) (1)
            # [None,      -d,        None,      None,    None,     None,   None,    None,    None, None,        None,     ], #16) -Vcref_d + u = 0 (u) (1)
        ])
        print(f"b4_Mmat.shape: {b4_Mmat.shape}")

        assert b4_Mmat.shape[0] == 14, f"expected: 14, got: {b4_Mmat.shape[0]}"     
        assert b4_Mmat.shape[1] == 22
        assert b4_Mmat.shape[1] == b4_num_vars

        # fmt: off
        b4_Kmat = sps.bmat(
            [
                # Vvr_qd, Vcref_qd, vvr_ccqd   iref_qd, iref_qd_f,iff_qd, ivr_cqd, ivr_gqd, z_qd, z_qdf   zvr_qdo               
                [ Z_qd,   None,     None,      None,    None,     None,   None,    None,    None,  None,  None, ],  # 1)iref_q - kp(Vref_q-Vc_q)-iffq-zvr_qo=0
                [ None,   None,     None,      Z_qd,    None,     None,   None,    None,    None,  None,  None, ],  # 2)iref_d - kp(Vref_d-Vc_d)-iffd-zvr_do=0
                [ None,   None,     None,      None,    Z_qd,     None,   None,    None,    None,  None,  None, ],  # 3)iref_qf - C(., Imx)=0 (fy)
                [ None,   None,     None,      None,    None,     Z_qd,   None,    None,    None,  None,  None, ],  # 4)iref_df - C(., Imx)=0 (fy)
                [ None,   None,     None,      None,    None,     None,   Z_qd,    None,    None,  None,  None, ],  # 5)iff_q - ivr_cq-ivr_gq=0
                [ None,   None,     None,      None,    None,     None,   None,    Z_qd,    None,  None,  None, ],  # 6)iff_d - ivr_cd-ivr_gd=0
                [ None,   None,     None,      None,    None,     None,   None,    None,    Z_qd,  None,  None, ],  # 7) z_q-iref_qf+iff_q=0
                [ None,   None,     None,      None,    None,     None,   None,    None,    None,  Z_qd,  None, ],  # 8)z_d-iref_df + iff_d=0
                [ None,   None,     None,      None,    None,     None,   None,    None,    None,  None,  Z_qd, ],  # 9)z_qf-C(.Imx)=0 (fy)
                [ None,   None,     None,      None,    None,     None,   None,    None,    None,  None,  Z_qd, ],  # 10)z_qf-C(.Imx)=0 (fy)
                [ None,   None,     None,      None,    None,     None,   None,    None,    None,  Z_qd,  q,    ],  # 11)d(zvr_qo)/dt-(1/tau*zq_f-1/tau*zvr_qo) (K)
                [ None,   None,     None,      None,    None,     None,   None,    None,    None,  Z_qd,  d,    ],  # 12)d(zvr_do)/dt-(1/tau*zd_f-1/tau*zvr_do) (K)
                [ None,   Z_qd,     Z_qd,      None,    None,     None,   Z_qd,    None,    None,  None,  None, ],  # 13)ivr_cq - w*C*vvr_ccqd = 0 (fy)
                [ None,   None,     Z_qd,      None,    None,     None,   Z_qd,    None,    None,  None,  None, ],  # 14)ivr_cd - w*C*vvr_ccqd = 0 (fy)
                # [None,    Z_qd,     None,      None,    None,     None,   None,    None,    None, None,   None, ], #15) -Vcref_q + u = 0 (u) (1)
                # [None,    Z_qd,     None,      None,    None,     None,   None,    None,    None, None,   None, ], #16) -Vcref_d + u = 0 (u) (1)
            ]
        )
        # fmt: on

        assert b4_Mmat.shape == b4_Kmat.shape, f"{b4_Mmat.shape} == {b4_Kmat.shape}"

        def _get_fy_dynamic_b4(t, y) -> np.ndarray:
            # fy4: Voltage controller block
            # non-linearity in eqns 1,2, 5, 6, 11, 12, 13, 14
            fy4 = np.zeros((self.b4_Mmat.shape[0], 1), dtype=float)

            idx_iref_q = self.var_offset_dynamic["iref_qd"]
            idx_iref_d = self.var_offset_dynamic["iref_qd"] + 1
            idx_z_q = self.var_offset_dynamic["z_qd"]
            idx_z_d = self.var_offset_dynamic["z_qd"] + 1
            idx_w = self.var_offset_dynamic["w"]
            idx_w_pu = self.var_offset_dynamic["w_pu"]
            idx_vvr_ccq = self.var_offset_dynamic["vvr_ccqd"]
            idx_vvr_ccd = self.var_offset_dynamic["vvr_ccqd"] + 1

            iref_q = y[idx_iref_q]
            iref_d = y[idx_iref_d]
            iref_mag = np.sqrt(iref_q**2 + iref_d**2)
            z_q = y[idx_z_q]
            z_d = y[idx_z_d]
            z_mag = np.sqrt(z_q**2 + z_d**2)
            vvr_ccq = y[idx_vvr_ccq]
            vvr_ccd = y[idx_vvr_ccd]
            w = y[idx_w]
            w_pu = y[idx_w_pu]
            C = self.obj.Ca

            # eq3: iref_qf - C(iref_q, Imx) = 0
            # eq4: iref_qf - C(iref_d, Imx) = 0
            idx_eq3 = 2
            idx_eq4 = 3
            if (
                iref_mag > self.obj.Imx
            ):  # compute Imax in adapter and parse it in the model
                print(f"tmp:: iref_mag > Imx...")
                fy4[idx_eq3] = -(self.obj.Imx * iref_q) / iref_mag
                fy4[idx_eq4] = -(self.obj.Imx * iref_d) / iref_mag
                print(f"fy4[idx_eq3]={fy4[idx_eq3]}, fy4[idx_eq4]={fy4[idx_eq4]}")
                # input("continue?")
            else:
                print(f"tmp:: iref_mag <= Imx...")
                fy4[idx_eq3] = -iref_q
                fy4[idx_eq4] = -iref_d

            # eq11: z_qf - C(z_q., Imx) = 0
            # eq12: z_df - C(z_d., Imx) = 0
            idx_eq9_start = 8
            idx_eq10_start = 9
            if (
                z_mag > self.obj.Imx
            ):  # compute Imax in adapter and parse it in the model
                print(f"tmp:: z_mag > Imx...")
                print(f"tmp:: z_mag={z_mag}")
                fy4[idx_eq9_start] = -(self.obj.Imx * z_q) / z_mag
                fy4[idx_eq10_start] = -(self.obj.Imx * z_d) / z_mag
                print(
                    f"fy4[idx_eq9_start]={fy4[idx_eq9_start]}, fy4[idx_eq10_start]={fy4[idx_eq10_start]}"
                )
                # input("continue?")
            else:
                print(f"tmp:: z_mag <= Imx...")
                fy4[idx_eq9_start] = -z_q
                fy4[idx_eq10_start] = -z_d

            # eq13: ivr_cq - w*C*vvr_ccd = 0
            idx_eq13_start = 12
            fy4[idx_eq13_start] = -(w_pu * self.w_b * C * vvr_ccd) / self.Zb
            # print(f"ivr_cq (fy): {fy4[idx_eq13_start]}")

            # eq14: ivr_cd +  w*C*vvr_ccq = 0
            idx_eq14_start = 13
            fy4[idx_eq14_start] = (w_pu * self.w_b * C * vvr_ccq) / self.Zb
            # print(f"ivr_cd (fy): {fy4[idx_eq14_start]}")

            return fy4

        def get_u4(t, y) -> np.ndarray:
            u4 = np.zeros((self.b4_Mmat.shape[0], 1), dtype=float)

            # 15,16) -Vcref_qd + u = 0
            # idx_eqn15 = 14
            # u4[idx_eqn15] = 1
            # # u4[idx_eqn15] = np.sqrt(2) * self.nominal_voltage_rms/self.V_base
            # idx_eqn16 = 15
            # u4[idx_eqn16] = 0
            return u4

        assert not hasattr(self, "b4_Mmat") or self.b4_Mmat is None, (
            f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.b4_Mmat = b4_Mmat
        self.b4_Kmat = b4_Kmat
        self.var_offset_dynamic_b4 = b4_var_offset
        self.b4_num_vars = b4_num_vars
        self._get_fy_dynamic_b4 = _get_fy_dynamic_b4
        self._get_u4_dynamic = get_u4

    def _current_controller(self):
        # b5.Current controller
        # fmt:off
        b5_var_offset = {
            "Vin_qd": 0,  # [Vin_q, Vin_d] reference for inverter terminal voltage
            "Vin_qdo": self.n_varqd,  # [Vin_qo, Vin_do] final out of current controller after antiwindup
            "iin_cc_qd": 2 * self.n_varqd,  # [iin_cc_q, iin_cc_d] current inverter output current from M&T block
            "icc_qdref": 3 * self.n_varqd,  # [icc_qref, icc_dref] current reference obtained from Voltage controller
            "zi_qdo": 4 * self.n_varqd,  # [zi_q, zi_d] output of integrator of current controller
        }
        # fmt:on
        b5_num_vars = b5_var_offset["zi_qdo"] + 2

        # coefficients for eqns of current controller
        q = np.array([[1, 0]])
        d = np.array([[0, 1]])
        Kpc_q = self.obj.Kpc * q
        Kpc_d = self.obj.Kpc * d
        inv_tauic_q = (1 / self.obj.tauic) * q
        inv_tauic_d = (1 / self.obj.tauic) * d

        # fmt:off
        b5_Mmat = sps.bmat([
            #Vin_qd,   Vin_qdo,    iin_cc_qd,  icc_qdref,  zi_qdo
            [q,        None,       Kpc_q,     -Kpc_q,     -q          ], # 1)Vin_q - Kp*(icc_qref - iincc_q) - zi_qo=0 (1)
            [d,        None,       Kpc_d,     -Kpc_d,     -d          ], # 2)Vin_d - Kp*(icc_dref - iincc_d) - zi_do=0 (1)
            [None,     q,          None,       None,       None       ], # 3)Vin_qo-C(.,Vmx)=0 (fy) (1)
            [None,     d,          None,       None,       None       ], # 4)Vin_do-C(.,Vmx)=0 (fy) (1)
            [None,    -inv_tauic_q,None,       None,       inv_tauic_q], # 5) d(zi_qo)/dt - (1/taui)*(Vin_qo - zi_qo)=0 (K) (1)
            [None,    -inv_tauic_d,None,       None,       inv_tauic_d], # 6) d(zi_do)/dt - (1/taui)*(Vin_do - zi_do)=0 (K) (1)
        ])
        # fmt:on

        assert b5_Mmat.shape[0] == 6
        assert b5_Mmat.shape[1] == 10
        assert b5_Mmat.shape[1] == b5_num_vars

        Z_qd = np.array([[0, 0]])

        # fmt:off
        b5_Kmat = sps.bmat([
            #Vin_qd,   Vin_qdo,  iin_cc_qd,  icc_qdref,  zi_qdo
            [Z_qd,     None,     None,       None,       None    ], # 1)Vin_q - Kp*(icc_qref - iincc_q) - zi_qo=0 (1)
            [None,     Z_qd,     None,       None,       None    ], # 2)Vin_d - Kp*(icc_dref - iincc_d) - zi_do=0 (1)
            [None,     None,     Z_qd,       None,       None    ], # 3)Vin_qo-C(.,Vmx)=0 (fy) (1)
            [None,     None,     None,       Z_qd,       None    ], # 4)Vin_do-C(.,Vmx)=0 (fy) (1)
            [None,     None,     None,       None,       q       ], # 5) d(zi_qo)/dt - (1/taui)*(Vin_qo - zi_qo)=0 (K) (1)
            [None,     None,     None,       None,       d       ], # 6) d(zi_do)/dt - (1/taui)*(Vin_do - zi_do)=0 (K) (1)
        ])
        # fmt : on

        assert b5_Mmat.shape == b5_Kmat.shape

        def _get_fy_dynamic_b5(t, y) -> np.ndarray:
            # fy5: Current controller block
            # non-linearities in eqn 3 and 4
            fy5 = np.zeros((self.b5_Mmat.shape[0], 1), dtype=float)
            idx_Vin_q = self.var_offset_dynamic["Vin_qd"]
            idx_Vin_d = idx_Vin_q + 1
            Vin_q = y[idx_Vin_q]
            Vin_d = y[idx_Vin_d]
            Vin_mag = np.sqrt(Vin_q**2 + Vin_d**2)

            print(f">> Vmx: {self.obj.Vmx}")
            print(f">> Vin_mag: {Vin_mag}")
            # assert Vin_mag <= 3

            idx_eqn3 = 2
            idx_eqn4 = 3
            if Vin_mag > self.obj.Vmx:  # compute Vmax in adapter and parse in the model
                fy5[idx_eqn3] = -(self.obj.Vmx * Vin_q) / Vin_mag
                fy5[idx_eqn4] = -(self.obj.Vmx * Vin_d) / Vin_mag
                # input("continue?")
            else:
                fy5[idx_eqn3] = -(Vin_q)
                fy5[idx_eqn4] = -(Vin_d)

            return fy5

        def get_u5(t, y) -> np.ndarray:
            u5 = np.zeros((self.b5_Mmat.shape[0], 1), dtype=float)
            return u5

        assert not hasattr(self, "b5_Mmat") or self.b5_Mmat is None, f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"

        self.b5_Mmat = b5_Mmat
        self.b5_Kmat = b5_Kmat
        self.var_offset_dynamic_b5 = b5_var_offset
        self.b5_num_vars = b5_num_vars
        self._get_fy_dynamic_b5 = _get_fy_dynamic_b5
        self._get_u5_dynamic = get_u5

    def _svpwm(self):
        # b6. SVPWM
        # fmt:off
        b6_var_offset = {
            "m_qd": 0,
            "m_qd_f": self.n_varqd,
            "v_qd_ref": 2 * self.n_varqd,
            "m_abc": 3 * self.n_varqd,
        }
        # fmt:on
        b6_num_vars = b6_var_offset["m_abc"] + self.n_ph

        # coefficients for eqns of SVPWM
        a = np.array([[1, 0, 0]])
        b = np.array([[0, 1, 0]])
        c = np.array([[0, 0, 1]])
        q = np.array([[1, 0]])
        d = np.array([[0, 1]])
        inv_Vdc_q = (1 / self.obj.Vdc) * self.Vqd_b * q
        inv_Vdc_d = (1 / self.obj.Vdc) * self.Vqd_b * d
        Z_qd = np.array([[0, 0]])
        Z_abc = np.array([[0, 0, 0]])
        Id_one = sps.identity((1), dtype=float)

        # fmt: off
        b6_Mmat = sps.bmat([
            # m_qd, m_qd_f, v_qd_ref,   m_abc
            [q,     None,   -inv_Vdc_q, None], #1) mq - vq/Vdc = 0 (1)
            [d,     None,   -inv_Vdc_d, None], #2) md - Vd/Vdc = 0 (1)
            [None,  q,      None,       None], #3) mq_f - C(mq, mx) = 0 (fy) (1)
            [None,  d,      None,       None], #4) md_f - C(md, mx) = 0 (fy) (1)
            [None,  None,   None,       a   ], #5) m_a - Kinv(m_qd_f) (1)
            [None,  None,   None,       b   ], #6) m_b - Kinv(m_qd_f) (1)
            [None,  None,   None,       c   ], #7) m_c - Kinv(m_qd_f) (1)          

        ])
        # fmt: on

        assert b6_Mmat.shape[0] == 7
        assert b6_Mmat.shape[1] == 9
        assert b6_Mmat.shape[1] == b6_num_vars

        # fmt:off
        b6_Kmat = sps.bmat([
            # m_qd, m_qd_f, v_qd_ref,   m_abc
            [Z_qd,  None,   None,      None], #1) mq - vq/Vdc = 0 (1)
            [None,  Z_qd,   None,      None], #2) md - Vd/Vdc = 0 (1)
            [None,  None,   Z_qd,      None], #3) mq_f - C(mq, mx) = 0 (fy) (1)
            [None,  None,   None,      Z_abc],#4) md_f - C(md, mx) = 0 (fy) (1)
            [Z_qd,  None,   None,      None], #5) m_a - Kinv(m_qd_f) (fy) (1)
            [None,  Z_qd,   None,      None], #6) m_b - Kinv(m_qd_f) (fy) (1)
            [None,  None,   Z_qd,      None], #7) m_c - Kinv(m_qd_f) (fy) (1) 

        ])
        # fmt:on

        def _get_fy_dynamic_b6(t, y) -> np.ndarray:
            # fy6: SVPWM block
            # non-linearities in eqn 1, 2, 3, 4 and 5
            fy6 = np.zeros((self.b6_Mmat.shape[0], 1), dtype=float)
            idx_mq_f = self.var_offset_dynamic_b6["m_qd_f"]
            idx_md_f = idx_mq_f + 1

            mq_f = y[idx_mq_f]
            md_f = y[idx_md_f]

            theta = y[self.var_offset_dynamic["theta"]]

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

        def get_u6(t, y) -> np.ndarray:
            u6 = np.zeros((self.b6_Mmat.shape[0], 1), dtype=float)
            # u6[9] =  -0.5
            # u6[10] = -0.5
            # u6[11] = -0.5
            return u6

        assert not hasattr(self, "b6_Mmat") or self.b6_Mmat is None, (
            f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.b6_Mmat = b6_Mmat
        self.b6_Kmat = b6_Kmat
        self.var_offset_dynamic_b6 = b6_var_offset
        self.b6_num_vars = b6_num_vars
        self._get_fy_dynamic_b6 = _get_fy_dynamic_b6
        self._get_u6_dynamic = get_u6

    def _inverter_bridge(self):
        # b7. 3 phase, 2-level inverter bridge
        # fmt:off
        b7_var_offset = {
            "E_abc": 0,  # [E_a, E_b, E_c] voltage at terminal of the inverter
            "v_abc": self.n_ph,  # [v_a, v_b, v_c] voltage across each phase
            "Eab": 2 * self.n_ph,
            "Ebc": 1 + 2 * self.n_ph,
            "i_abc": 2 + 2 * self.n_ph,  # [E_a, E_b, E_c] current in each leg of inverter
            "din": 2 + 3 * self.n_ph,  # [din_a, din_b, din_b] duty cycle obtained from SVPWM
        }
        # fmt:on
        b7_num_vars = b7_var_offset["din"] + 3

        # coefficients for eqns of inverter bridge
        a = np.array([[1, 0, 0]])
        b = np.array([[0, 1, 0]])
        c = np.array([[0, 0, 1]])
        ab = np.array([[1, -1, 0]])
        bc = np.array([[0, 1, -1]])
        Zrow = sps.lil_matrix((1, self.n_ph), dtype=float)
        Id_one = sps.identity((1), dtype=float)

        # fx = (Vdc-vsw-ix*rsw)*din_x - (vd + ix*rd)*(1-din_x) for ix>=0
        # fx = (Vdc + vd-ix*rd)*din_x + (vsw-rsw*ix)*(1-din_x) for ix<0

        # fmt:off
        b7_Mmat = sps.bmat(
            [
                # E_abc, v_abc,   Eab,     Ebc,    i_abc,  din,                
                [ None,   a,      None,    None,   Zrow,   Zrow],  # 1)va-fa=0 (fy) (1)
                [ None,   b,      None,    None,   None,   None],  # 2)vb-fb=0 (fy) (1)
                [ None,   c,      None,    None,   None,   None],  # 3)vc-fc=0 (fy) (1)
                [ a,     -a,      None,    None,   None,   None],  # 4)Ea-va=0 (1)
                [ b,     -b,      None,    None,   None,   None],  # 5)Eb-vb=0 (1)
                [ c,     -c,      None,    None,   None,   None],  # 6)Ec-vc=0 (1)
                [ ab,  None,     -Id_one,  None,   None,   None],  # 7)-Eab + Ea - Eb =0 (1)
                [ bc,  None,      None,   -Id_one, None,   None],  # 8)-Ebc + Eb - Ec=0 (1)
            ]
        )
        # fmt: on

        assert b7_Mmat.shape[0] == 8
        assert b7_Mmat.shape[1] == 14
        assert b7_Mmat.shape[1] == b7_num_vars

        b7_Kmat = sps.lil_matrix(b7_Mmat.shape, dtype=float)

        def _get_fy_dynamic_b7(t, y) -> np.ndarray:
            # fy7: inverter bridge block
            # non-linearity in eqn 1, 2 and 3
            # vx-fx = 0 : x ={a, b, c}
            # fx = (Vdc-vsw-ix*rsw)*din_x - (vd + ix*rd)*(1-din_x) for ix>=0
            # fx = (Vdc + vd+ix*rd)*din_x + (vsw-rsw*ix)*(1-din_x) for ix<0
            fy7 = np.zeros((self.b7_Mmat.shape[0], 1), dtype=float)
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
                fy7[idx_eq1_start] = -(
                    (Vdc - vsw - rsw * ia) * da - (vd + rd * ia) * (1 - da)
                )
            else:
                fy7[idx_eq1_start] = -(
                    (Vdc + vd + rd * ia) * da + (vsw - rsw * ia) * (1 - da)
                )

            # fy7[idx_eq1_start] = -(Vdc * da) / 2

            # eq2 : fb
            idx_eq2_start = 1
            if ib >= 0:
                fy7[idx_eq2_start] = -(
                    (Vdc - vsw - rsw * ib) * db - (vd + rd * ib) * (1 - db)
                )
            else:
                fy7[idx_eq2_start] = -(
                    (Vdc + vd + rd * ib) * db + (vsw - rsw * ib) * (1 - db)
                )

            # fy7[idx_eq2_start] = -(Vdc * db) / 2

            # eq3: fc
            idx_eq3_start = 2
            if ic >= 0:
                fy7[idx_eq3_start] = -(
                    (Vdc - vsw - rsw * ic) * dc - (vd + rd * ic) * (1 - dc)
                )
            else:
                fy7[idx_eq3_start] = -(
                    (Vdc + vd + rd * ic) * dc + (vsw - rsw * ic) * (1 - dc)
                )

            # fy7[idx_eq3_start] = -(Vdc * dc) / 2

            return fy7

        def get_u7(t, y) -> np.ndarray:
            u7 = np.zeros((self.b7_Mmat.shape[0], 1), dtype=float)
            return u7

        assert not hasattr(self, "b7_Mmat") or self.b7_Mmat is None, (
            f"this M matrix should not be present at this point [PLEASE INVESTIGATE]"
        )

        self.b7_Mmat = b7_Mmat
        self.b7_Kmat = b7_Kmat
        self.var_offset_dynamic_b7 = b7_var_offset
        self.b7_num_vars = b7_num_vars
        self._get_fy_dynamic_b7 = _get_fy_dynamic_b7
        self._get_u7_dynamic = get_u7

    ###########################dynamic simulation functions#########################################
    def init_dynamic_simulation(self):
        # block by block
        # then interface them
        # and remember to update the offsets while interfacing
        # also create an overall vars list from individual "blocks"

        var_offset_dynamic = {}
        eqn_offsets = {}

        # ---------------------------------------

        self._lcl()
        self._mnt()
        # Vbrf_Mmat, Vbrf_Kmat, Vbrf_var_offset, Vbrf_num_vars = self._vbrf()
        self._ibrf()
        self._pw_droop()
        self._qv_droop()
        self._voltage_controller()
        self._current_controller()
        self._svpwm()
        self._inverter_bridge()

        # create the combined var_offset dict
        var_offset_dynamic = {}
        shift = 0
        for offsets, num_vars in [
            (self.var_offset_dynamic_b1, self.b1_num_vars),
            (self.var_offset_dynamic_b2, self.b2_num_vars),
            (self.var_offset_dynamic_b3, self.b3_num_vars),
            (self.var_offset_dynamic_b4, self.b4_num_vars),
            (self.var_offset_dynamic_b5, self.b5_num_vars),
            (self.var_offset_dynamic_b6, self.b6_num_vars),
            (self.var_offset_dynamic_b7, self.b7_num_vars),
            # (var_offset_dynamic_Vbrf, Vbrf_num_vars),
            (self.var_offset_dynamic_Ibrf, self.Ibrf_num_vars),
            (self.var_offset_dynamic_qv, self.qv_num_vars),
        ]:
            # print(f"tmp:: offsets:{pformat(offsets)}")
            # print(f"tmp:: shift:{shift}")
            for var in offsets:
                # sanity check for variable names
                assert var not in var_offset_dynamic
                # shift offset
                var_offset_dynamic[var] = offsets[var] + shift
            shift += num_vars

        assert "ig_qd" in var_offset_dynamic

        # should not have any duplicate indices
        assert len(var_offset_dynamic.keys()) == len(set(var_offset_dynamic.values()))

        # also store the final var offsets by block for better checking
        for offsets in [
            self.var_offset_dynamic_b1,
            self.var_offset_dynamic_b2,
            self.var_offset_dynamic_b3,
            self.var_offset_dynamic_b4,
            self.var_offset_dynamic_b5,
            self.var_offset_dynamic_b6,
            self.var_offset_dynamic_b7,
            # var_offset_dynamic_Vbrf,
            self.var_offset_dynamic_Ibrf,
            self.var_offset_dynamic_qv,
        ]:
            for key in offsets:
                final_offset = var_offset_dynamic[key]
                offsets[key] = final_offset

        # interfacing:
        # - stack block matrices diagonally
        # - add interfacing rows to connect the variables between different blocks

        # fmt: off
        # M_dynamic = sps.bmat(
        #     [
        #         [b1_Mmat, None,    None,    None,    None,    None,    None,    None,     None   ],
        #         [None,    b2_Mmat, None,    None,    None,    None,    None,    None,     None   ],
        #         [None,    None,    b3_Mmat, None,    None,    None,    None,    None,     None   ],
        #         [None,    None,    None,    b4_Mmat, None,    None,    None,    None,     None   ],
        #         [None,    None,    None,    None,    b5_Mmat, None,    None,    None,     None   ],
        #         [None,    None,    None,    None,    None,    b6_Mmat, None,    None,     None   ],
        #         [None,    None,    None,    None,    None,    None,    b7_Mmat, None,     None   ],
        #         [None,    None,    None,    None,    None,    None,    None,    Vbrf_Mmat,None   ],
        #         [None,    None,    None,    None,    None,    None,    None,    None,     Ibrf_Mmat],
        #     ]
        # )
        # # fmt: on

        M_dynamic = sps.bmat(
            [
                [self.b1_Mmat, None,         None,         None,         None,         None,         None,         None,            None],
                [None,         self.b2_Mmat, None,         None,         None,         None,         None,         None,            None],
                [None,         None,         self.b3_Mmat, None,         None,         None,         None,         None,            None],
                [None,         None,         None,         self.b4_Mmat, None,         None,         None,         None,            None],
                [None,         None,         None,         None,         self.b5_Mmat, None,         None,         None,            None],
                [None,         None,         None,         None,         None,         self.b6_Mmat, None,         None,            None],
                [None,         None,         None,         None,         None,         None,         self.b7_Mmat, None,            None],            
                [None,         None,         None,         None,         None,         None,         None,         self.Ibrf_Mmat,  None],
                [None,         None,         None,         None,         None,         None,         None,         None,            self.qv_Mmat],
            ]
        )
        # fmt: on
        assert M_dynamic.shape[1] == (
            self.b1_num_vars
            + self.b2_num_vars
            + self.b3_num_vars
            + self.b4_num_vars
            + self.b5_num_vars
            + self.b6_num_vars
            + self.b7_num_vars
            # + Vbrf_num_vars
            + self.Ibrf_num_vars
            + self.qv_num_vars
        )

        M_dynamic_shape_before_iface = M_dynamic.shape

        # fmt: off
        K_dynamic = sps.bmat(
            [
                [self.b1_Kmat, None,         None, None, None, None, None,       None, None],
                [None,         self.b2_Kmat, None, None, None, None, None,       None, None],
                [None,         None,         self.b3_Kmat, None, None, None, None,       None, None],
                [None,         None,         None, self.b4_Kmat, None, None, None,       None, None],
                [None,         None,         None, None, self.b5_Kmat, None, None,       None, None],
                [None,         None,         None, None, None, self.b6_Kmat, None,       None, None],
                [None,         None,         None, None, None, None, self.b7_Kmat,       None, None],                
                [None,         None,         None, None, None, None,    None,  self.Ibrf_Kmat, None],
                [None,         None,         None, None, None, None,    None,       None, self.qv_Kmat],
            ]
        )
        # fmt: on

        assert M_dynamic.shape == K_dynamic.shape

        ###################Interface the blocks########################################
        # Note: Create and empty block of this size to stack below K matrix and also vectors of size=rows added to stack below fy and u
        # Interface equations are created with respect to the input one block requires and so are the rows named

        # tmp start: -----------

        # TODO: add args: fr_block, to_block -- to check that fr and to vars actually exist in fr_block and to_block
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

        # fmt:off
        interface_list = [
            # b2 <-> b1
            ("Vc", "Vcf", self.n_ph),
            ("ig", "igf", self.n_ph),
            ("iin", "iinf", self.n_ph),
            ("vcc", "vccf", self.n_ph),

            # # b3 <-> b2
            # ("Vdr_qd", "Vc_qd", self.n_varqd),
            # ("idr_qd", "ig_qd", self.n_varqd),
            
            # b3 <-> (b2, brf)
            ("Vdr_qd", "Vc_qd", self.n_varqd),
            ("idr_qd", "I_qd_dc", self.n_varqd),

            # qv_droop <-> (b2, brf)
            ("V_qvdroop_qd", "Vc_qd", self.n_varqd),
            ("i_qvdroop_qd", "I_qd_dc", self.n_varqd),

            # b4 <-> (b2, b3), b4 <-> (qv_droop))
            ("Vvr_cqd", "Vc_qd", self.n_varqd),
            ("ivr_gqd", "ig_qd", self.n_varqd),
            ("vvr_ccqd", "vcc_qd", self.n_varqd),
            ("Vcref_qd", "Vref_qd", self.n_varqd),

            # b5 <-> (b2, b4)
            ("iin_cc_qd", "iin_qd", self.n_varqd),
            ("icc_qdref", "iref_qd_f", self.n_varqd),

            # b6 <-> b5
            ("v_qd_ref", "Vin_qdo", self.n_varqd),

            # b7 <-> b6
            ("din", "m_abc", self.n_ph),

            # b1 <-> b7
            ("Vab", "Eab", 1),
            ("Vbc", "Ebc", 1),
            ("i_abc", "Iinf", self.n_ph),

            # brf stuff
            ("I_qd", "ig_qd", self.n_varqd),

            # TODO: interface qv and update the list above accordingly
        ]
        # fmt:on

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

    def get_K_dynamic(self, stage=None) -> sps.coo_array:
        return self.K_dynamic

    # def _get_fy_dynamic_Vbrf(self, t, y) -> np.ndarray:
    #     fy_Vbrf = np.zeros((self.Vbrf_Mmat.shape[0], 1), dtype=float)

    #    #2) d(Vbrf2_qd)/dt  + w_band*Vbrf2_qd + w^2*Vbrf1_qd - V_qd = 0 (K) (fy) (2)
    #     w_reject = 2 * 2*np.pi*60
    #     Vbrf1_q = y[self.var_offset_dynamic["Vbrf1_qd"]]
    #     Vbrf1_d = y[self.var_offset_dynamic["Vbrf1_qd"] + 1]

    #     fy_Vbrf[2] = (w_reject**2) * Vbrf1_q
    #     fy_Vbrf[3] = (w_reject**2) * Vbrf1_d

    #     return fy_Vbrf

    def _get_fy_dynamic_interface(self, t, y) -> np.ndarray:
        fy_interface = np.zeros((self.M_dynamic_iface_nrows, 1), dtype="float")
        return fy_interface

    def get_fy_dynamic(self, t, y, yp, stage=None) -> np.ndarray:
        fy1 = self._get_fy_dynamic_b1(t, y)
        fy2 = self._get_fy_dynamic_b2(t, y)
        fy3 = self._get_fy_dynamic_b3(t, y)
        fy4 = self._get_fy_dynamic_b4(t, y)
        fy5 = self._get_fy_dynamic_b5(t, y)
        fy6 = self._get_fy_dynamic_b6(t, y)
        fy7 = self._get_fy_dynamic_b7(t, y)

        # fy1 = self._get_fy1_dynamic(t, y)
        # fy2 = self._get_fy2_dynamic(t, y)
        # fy3 = self._get_fy3_dynamic(t, y)
        # fy4 = self._get_fy4_dynamic(t, y)
        # fy5 = self._get_fy5_dynamic(t, y)
        # fy6 = self._get_fy6_dynamic(t, y)
        # fy7 = self._get_fy7_dynamic(t, y)

        # fy_Vbrf = self._get_fy_dynamic_Vbrf(t, y)
        # fy_Ibrf = self._get_fy_dynamic_Ibrf(t, y)

        fy_Ibrf = self._get_fy_dynamic_Ibrf(t, y)
        fy_qv = self._get_fy_dynamic_qv(t, y)

        fy_interface = self._get_fy_dynamic_interface(t, y)

        # fmt: off
        fy = sps.bmat(
            [[fy1],
             [fy2],
             [fy3],
             [fy4],
             [fy5],
             [fy6],
             [fy7],
            #  [fy_Vbrf],
             [fy_Ibrf],
             [fy_qv],
             [fy_interface]]
        )
        # fmt: on

        assert self.b1_Mmat.shape[0] == fy1.shape[0]
        assert self.b2_Mmat.shape[0] == fy2.shape[0]
        assert self.b3_Mmat.shape[0] == fy3.shape[0]
        assert self.b4_Mmat.shape[0] == fy4.shape[0]
        assert self.b5_Mmat.shape[0] == fy5.shape[0]
        assert self.b6_Mmat.shape[0] == fy6.shape[0]
        assert self.b7_Mmat.shape[0] == fy7.shape[0]
        assert self.Ibrf_Mmat.shape[0] == fy_Ibrf.shape[0]
        assert self.qv_Mmat.shape[0] == fy_qv.shape[0]
        assert self.M_dynamic_iface_nrows == fy_interface.shape[0]

        assert fy.shape[0] == self.num_eqns_dynamic, (
            f"{fy.shape[0]} == {self.num_eqns_dynamic}"
        )

        return fy.toarray().flatten()

    # def get_u_dynamic
    def get_u_dynamic(self, t: float, y) -> np.ndarray:
        u1 = self._get_u1_dynamic(t, y)
        u2 = self._get_u2_dynamic(t, y)
        u3 = self._get_u3_dynamic(t, y)
        u4 = self._get_u4_dynamic(t, y)
        u5 = self._get_u5_dynamic(t, y)
        u6 = self._get_u6_dynamic(t, y)
        u7 = self._get_u7_dynamic(t, y)

        u_interface = np.zeros((self.M_dynamic_iface_nrows, 1), dtype="float")

        # u_vbrf = np.zeros((self.Vbrf_Mmat.shape[0], 1), dtype="float")
        # u_ibrf = np.zeros((self.Ibrf_Mmat.shape[0], 1), dtype="float")
        u_ibrf = self._get_u_ibrf_dynamic(t, y)
        u_qv = self._get_u_qv_dynamic(t, y)

        u_dynamic = sps.bmat(
            [
                [u1],
                [u2],
                [u3],
                [u4],
                [u5],
                [u6],
                [u7],
                # [u_vbrf],
                [u_ibrf],
                [u_qv],
                [u_interface],
            ]
        )

        assert u_dynamic.shape[0] == self.num_eqns_dynamic

        return u_dynamic.toarray().flatten()

    def _initial_guess_dynamic_b1(self, y_comp: list, wnom, y0_dyn: np.ndarray):
        # block wise initialization of b1 block
        # 1. LCL filter
        # [x] Vgf <- pf V[0-3] (V1)
        # [x] Vcf <- pf V[3-6] (V2)
        # [x] Vinf <- pf V[6-9](V3)
        # [x] Igf <- pf V[0-3] (I1)
        # [x] Icf <- pf I[3-6] (I2)
        # [x] Iinf <- pf (I3)
        # [xj] vgf <- pf (v1)
        # [x] vcf <- pf (v2)
        # [x] vinf <- pf (v3)
        # [ ] vccf <- pf (vc)
        # [x] igf <- pf (i1)
        # [x] icf <- pf (i2)
        # [x] iinf <- pf (i3)
        # [ ] qf
        # [ ] lamda_g
        # [ ] lamda_in

        # from powerflow
        # Vgf, Vcf, Vinf:
        pf_V_start_idx = self.var_offset["V"]
        pf_V_end_idx = pf_V_start_idx + 3 * self.n_ph
        dyn_V_start_idx = self.var_offset_dynamic_b1["V"]
        dyn_V_end_idx = dyn_V_start_idx + 3 * self.n_ph
        y0_dyn[dyn_V_start_idx:dyn_V_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_V_start_idx:pf_V_end_idx]
        ]

        print(
            f"tmp:: Vinf (init): {y0_dyn[dyn_V_start_idx + 2 * self.n_ph : dyn_V_end_idx]}"
        )
        print(
            f"tmp:: Vinf (init): {1 / 2 * y0_dyn[dyn_V_start_idx + 2 * self.n_ph : dyn_V_end_idx]}"
        )

        # Igf, Icf, Iinf:
        pf_I_start_idx = self.var_offset["I"]
        pf_I_end_idx = pf_I_start_idx + 3 * self.n_ph
        dyn_I_start_idx = self.var_offset_dynamic_b1["I"]
        dyn_I_end_idx = dyn_I_start_idx + 3 * self.n_ph
        y0_dyn[dyn_I_start_idx:dyn_I_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_I_start_idx:pf_I_end_idx]
        ]

        # vgf, vcf, vinf, vccf:
        # vgf = v1
        pf_v1_start_idx = self.var_offset["v"]
        pf_v1_end_idx = pf_v1_start_idx + self.n_ph
        dyn_vgf_start_idx = self.var_offset_dynamic_b1["vgf"]
        dyn_vgf_end_idx = dyn_vgf_start_idx + self.n_ph
        y0_dyn[dyn_vgf_start_idx:dyn_vgf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_v1_start_idx:pf_v1_end_idx]
        ]
        # vcf = v2
        pf_v2_start_idx = self.var_offset["v"] + self.n_ph
        pf_v2_end_idx = pf_v2_start_idx + self.n_ph
        dyn_vcf_start_idx = self.var_offset_dynamic_b1["vcf"]
        dyn_vcf_end_idx = dyn_vcf_start_idx + self.n_ph
        y0_dyn[dyn_vcf_start_idx:dyn_vcf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_v2_start_idx:pf_v2_end_idx]
        ]
        # vccf = vcc
        pf_vcc_start_idx = self.var_offset["v"] + 2 * self.n_ph
        pf_vcc_end_idx = pf_vcc_start_idx + self.n_ph
        dyn_vccf_start_idx = self.var_offset_dynamic_b1["vccf"]
        dyn_vccf_end_idx = dyn_vccf_start_idx + self.n_ph
        y0_dyn[dyn_vccf_start_idx:dyn_vccf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_vcc_start_idx:pf_vcc_end_idx]
        ]
        # vinf = v3
        pf_v3_start_idx = self.var_offset["v"] + 3 * self.n_ph
        pf_v3_end_idx = pf_v3_start_idx + self.n_ph
        dyn_vinf_start_idx = self.var_offset_dynamic_b1["vinf"]
        dyn_vinf_end_idx = dyn_vinf_start_idx + self.n_ph
        y0_dyn[dyn_vinf_start_idx:dyn_vinf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_v3_start_idx:pf_v3_end_idx]
        ]
        # igf, icf, iinf:
        # pf:i1 -> dyn:igf
        pf_i_start_idx = self.var_offset["i"]
        pf_i_end_idx = pf_i_start_idx + self.n_ph
        dyn_i_start_idx = self.var_offset_dynamic_b1["igf"]
        dyn_i_end_idx = dyn_i_start_idx + self.n_ph
        y0_dyn[dyn_i_start_idx:dyn_i_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_i_start_idx:pf_i_end_idx]
        ]
        # pf:i2 -> dyn:icf
        pf_i_start_idx = self.var_offset["i"] + self.n_ph
        pf_i_end_idx = pf_i_start_idx + self.n_ph
        dyn_i_start_idx = self.var_offset_dynamic_b1["icf"]
        dyn_i_end_idx = dyn_i_start_idx + self.n_ph
        y0_dyn[dyn_i_start_idx:dyn_i_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_i_start_idx:pf_i_end_idx]
        ]
        # pf:i3 -> dyn:iinf
        pf_i_start_idx = self.var_offset["i"] + 2 * self.n_ph
        pf_i_end_idx = pf_i_start_idx + self.n_ph
        dyn_i_start_idx = self.var_offset_dynamic_b1["iinf"]
        dyn_i_end_idx = dyn_i_start_idx + self.n_ph
        y0_dyn[dyn_i_start_idx:dyn_i_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_i_start_idx:pf_i_end_idx]
        ]
        # qf = q
        pf_q_start_idx = self.var_offset["q"]
        pf_q_end_idx = pf_q_start_idx + self.n_ph
        dyn_qf_start_idx = self.var_offset_dynamic_b1["qf"]
        dyn_qf_end_idx = dyn_qf_start_idx + self.n_ph
        y0_dyn[dyn_qf_start_idx:dyn_qf_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_q_start_idx:pf_q_end_idx]
        ]
        # lamda_g = lamda1
        # lamda_in = lamda2]
        pf_lamda_start_idx = self.var_offset["lamda"]
        pf_lamda_end_idx = pf_lamda_start_idx + 2 * self.n_ph
        dyn_lamda_start_idx = self.var_offset_dynamic_b1["lamda_g"]
        dyn_lamda_end_idx = dyn_lamda_start_idx + 2 * self.n_ph
        y0_dyn[dyn_lamda_start_idx:dyn_lamda_end_idx] = [
            np.sqrt(2) * phasor_to_timedomain(val)
            for val in y_comp[pf_lamda_start_idx:pf_lamda_end_idx]
        ]

        # # compute for initialization
        # # dyn:ig_qd <- computed from igf
        # igf = y0_dyn[
        #     self.var_offset_dynamic_b1["igf"] : self.var_offset_dynamic_b1["igf"] + self.n_ph
        # ]
        # dyn_ig_start = self.var_offset_dynamic_b1["igf"]
        # dyn_ig_end = dyn_ig_start + self.n_ph
        # y0_dyn[dyn_ig_start:dyn_ig_end] = igf
        # # dyn:iq, id <- computed from igf
        # iq = (2 / 3) * (
        #     igf[0] * np.cos(0)
        #     + igf[1] * np.cos(-2 * np.pi / 3)
        #     + igf[2] * np.cos(2 * np.pi / 3)
        # )
        # id = (2 / 3) * (
        #     igf[0] * np.sin(0)
        #     + igf[1] * np.sin(-2 * np.pi / 3)
        #     + igf[2] * np.sin(2 * np.pi / 3)
        # )
        # dyn_igqd_start_idx = self.var_offset_dynamic_b1["ig_qd"]
        # dyn_igqd_end_idx = dyn_igqd_start_idx + 2
        # y0_dyn[dyn_igqd_start_idx:dyn_igqd_end_idx] = [iq, id]

        # pf_i_start = self.var_offset["i"]
        # ia = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_i_start])
        # ib = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_i_start + 1])
        # ic = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_i_start + 2])

        # dyn_ig_q_start = self.var_offset_dynamic_b1["ig_qd"]
        # y0_dyn[dyn_ig_q_start] = (2 / 3) * (
        #     ia * np.cos(0) + ib * np.cos(-2 * np.pi / 3) + ic * np.cos(2 * np.pi / 3)
        # )
        # dyn_ig_d_start = self.var_offset_dynamic_b1["ig_qd"] + 1
        # y0_dyn[dyn_ig_d_start] = (2 / 3) * (
        #     ia * np.sin(0) + ib * np.sin(-2 * np.pi / 3) + ic * np.sin(2 * np.pi / 3)
        # )

        return y0_dyn

    def _initial_guess_dynamic_b2(self, y_comp: list, wnom, y0_dyn: np.ndarray):
        # from powerflow
        # Vg = V1
        # Vc = V2
        # ig = i1
        # iin = i3

        # theta
        pf_Vc_start = self.var_offset["V"] + self.n_ph
        theta = np.angle(y_comp[pf_Vc_start])

        # Vg = V1
        pf_V_start = self.var_offset["V"]
        pf_V_end = pf_V_start + self.n_ph
        dyn_Vg_start = self.var_offset_dynamic_b2["Vg"]
        dyn_Vg_end = dyn_Vg_start + self.n_ph
        y0_dyn[dyn_Vg_start:dyn_Vg_end] = np.sqrt(2) * phasor_to_timedomain(
            y_comp[pf_V_start:pf_V_end]
        )

        # Vc = V2
        pf_V_start = self.var_offset["V"] + self.n_ph
        pf_V_end = pf_V_start + self.n_ph
        dyn_Vc_start = self.var_offset_dynamic_b2["Vc"]
        dyn_Vc_end = dyn_Vc_start + self.n_ph
        y0_dyn[dyn_Vc_start:dyn_Vc_end] = np.sqrt(2) * phasor_to_timedomain(
            y_comp[pf_V_start:pf_V_end]
        )

        # ig = i1
        pf_i_start = self.var_offset["i"]
        pf_i_end = pf_i_start + self.n_ph
        dyn_ig_start = self.var_offset_dynamic_b2["ig"]
        dyn_ig_end = dyn_ig_start + self.n_ph
        y0_dyn[dyn_ig_start:dyn_ig_end] = np.sqrt(2) * phasor_to_timedomain(
            y_comp[pf_i_start:pf_i_end]
        )

        # iin = i3
        pf_i3_start = self.var_offset["i"] + 2 * self.n_ph
        pf_i3_end = pf_i3_start + self.n_ph
        dyn_iin_start = self.var_offset_dynamic_b2["iin"]
        dyn_iin_end = dyn_iin_start + self.n_ph
        y0_dyn[dyn_iin_start:dyn_iin_end] = np.sqrt(2) * phasor_to_timedomain(
            y_comp[pf_i3_start:pf_i3_end]
        )

        # compute
        # Vg_ll
        # Vg_qd
        # Vc_qd
        # ig_qd
        # iin_qd

        # Vg_ll
        Va = np.sqrt(2) * phasor_to_timedomain(y_comp[self.var_offset["V"]])
        Vb = np.sqrt(2) * phasor_to_timedomain(y_comp[self.var_offset["V"] + 1])
        Vc = np.sqrt(2) * phasor_to_timedomain(y_comp[self.var_offset["V"] + 2])
        dyn_Vg_ab_start = self.var_offset_dynamic_b2["Vg_ll"]
        y0_dyn[dyn_Vg_ab_start] = Va - Vb

        dyn_Vg_bc_start = self.var_offset_dynamic_b2["Vg_ll"] + 1
        y0_dyn[dyn_Vg_bc_start] = Vb - Vc

        # Vg_qd
        dyn_Vg_q_start = self.var_offset_dynamic_b2["Vg_qd"]
        y0_dyn[dyn_Vg_q_start] = (2 / 3) * (
            Va * np.cos(theta)
            + Vb * np.cos(theta - 2 * np.pi / 3)
            + Vc * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_Vg_d_start = self.var_offset_dynamic_b2["Vg_qd"] + 1
        y0_dyn[dyn_Vg_d_start] = (2 / 3) * (
            Va * np.sin(theta)
            + Vb * np.sin(theta - 2 * np.pi / 3)
            + Vc * np.sin(theta + 2 * np.pi / 3)
        )

        # vcc
        pf_vcc_start = self.var_offset["v"] + 2 * self.n_ph
        pf_vcc_end = pf_vcc_start + self.n_ph
        pf_vcc = y_comp[pf_vcc_start:pf_vcc_end]
        dyn_vcc_start = self.var_offset_dynamic_b2["vcc"]
        dyn_vcc_end = dyn_vcc_start + self.n_ph
        y0_dyn[dyn_vcc_start:dyn_vcc_end] = np.sqrt(2) * phasor_to_timedomain(pf_vcc)

        # Vc_qd
        Vc_start_idx = self.var_offset["V"] + self.n_ph
        Vc_a = np.sqrt(2) * phasor_to_timedomain(y_comp[Vc_start_idx])
        Vc_b = np.sqrt(2) * phasor_to_timedomain(y_comp[Vc_start_idx + 1])
        Vc_c = np.sqrt(2) * phasor_to_timedomain(y_comp[Vc_start_idx + 2])
        dyn_Vc_q_start = self.var_offset_dynamic_b2["Vc_qd"]
        y0_dyn[dyn_Vc_q_start] = (2 / 3) * (
            Vc_a * np.cos(theta)
            + Vc_b * np.cos(theta - 2 * np.pi / 3)
            + Vc_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_Vc_d_start = self.var_offset_dynamic_b2["Vc_qd"] + 1
        y0_dyn[dyn_Vc_d_start] = (2 / 3) * (
            Vc_a * np.sin(theta)
            + Vc_b * np.sin(theta - 2 * np.pi / 3)
            + Vc_c * np.sin(theta + 2 * np.pi / 3)
        )

        print(f"Vc_q (initialized): {y0_dyn[dyn_Vc_q_start]}")
        print(f"Vc_d (initialized): {y0_dyn[dyn_Vc_d_start]}")

        # vcc_qd
        dyn_vcc_q_idx = self.var_offset_dynamic["vcc_qd"]
        dyn_vcc_d_idx = dyn_vcc_q_idx + 1
        vcc_a = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_vcc_start])
        vcc_b = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_vcc_start + 1])
        vcc_c = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_vcc_start + 2])
        y0_dyn[dyn_vcc_q_idx] = (2 / 3) * (
            vcc_a * np.cos(theta)
            + vcc_b * np.cos(theta - 2 * np.pi / 3)
            + vcc_c * np.cos(theta + 2 * np.pi / 3)
        )
        y0_dyn[dyn_vcc_d_idx] = (2 / 3) * (
            vcc_a * np.sin(theta)
            + vcc_b * np.sin(theta - 2 * np.pi / 3)
            + vcc_c * np.sin(theta + 2 * np.pi / 3)
        )

        # ig_qd
        ig_start_idx = self.var_offset["i"]
        ig_a = np.sqrt(2) * phasor_to_timedomain(y_comp[ig_start_idx])
        ig_b = np.sqrt(2) * phasor_to_timedomain(y_comp[ig_start_idx + 1])
        ig_c = np.sqrt(2) * phasor_to_timedomain(y_comp[ig_start_idx + 2])
        dyn_ig_q_start = self.var_offset_dynamic_b2["ig_qd"]
        y0_dyn[dyn_ig_q_start] = (2 / 3) * (
            ig_a * np.cos(theta)
            + ig_b * np.cos(theta - 2 * np.pi / 3)
            + ig_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_ig_d_start = dyn_ig_q_start + 1
        y0_dyn[dyn_ig_d_start] = (2 / 3) * (
            ig_a * np.sin(theta)
            + ig_b * np.sin(theta - 2 * np.pi / 3)
            + ig_c * np.sin(theta + 2 * np.pi / 3)
        )

        # iin q_d
        iin_start_idx = self.var_offset["i"] + 2 * self.n_ph
        iin_a = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx])
        iin_b = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx + 1])
        iin_c = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx + 2])

        dyn_iin_q_start = self.var_offset_dynamic_b2["iin_qd"]
        y0_dyn[dyn_iin_q_start] = (2 / 3) * (
            iin_a * np.cos(theta)
            + iin_b * np.cos(theta - 2 * np.pi / 3)
            + iin_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_iin_d_start = self.var_offset_dynamic_b2["iin_qd"] + 1
        y0_dyn[dyn_iin_d_start] = (2 / 3) * (
            iin_a * np.sin(theta)
            + iin_b * np.sin(theta - 2 * np.pi / 3)
            + iin_c * np.sin(theta + 2 * np.pi / 3)
        )

        return y0_dyn

    def _initial_guess_dynamic_b3(self, y_comp: list, wnom, y0_dyn: np.ndarray):
        # w, Vdr_qd, Vref_amp, V_refc, idr_qd, Pe, Pe_f, Qe, Qe_f, theta

        # w
        dyn_w_idx = self.var_offset_dynamic_b3["w"]
        y0_dyn[dyn_w_idx] = wnom

        # theta = angle(Vc)
        pf_Vc_start = self.var_offset["V"] + self.n_ph
        dyn_theta_start = self.var_offset_dynamic_b3["theta"]
        y0_dyn[dyn_theta_start] = np.angle(y_comp[pf_Vc_start])
        theta = y0_dyn[dyn_theta_start]

        # Vdr_qd : qd transform of Vc as we measure voltage at capacitor node
        pf_Vc_start = self.var_offset["V"] + self.n_ph
        Va = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_Vc_start])
        Vb = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_Vc_start + 1])
        Vc = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_Vc_start + 2])
        dyn_Vdr_q_start = self.var_offset_dynamic_b3["Vdr_qd"]
        y0_dyn[dyn_Vdr_q_start] = (2 / 3) * (
            Va * np.cos(theta)
            + Vb * np.cos(theta - 2 * np.pi / 3)
            + Vc * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_Vdr_d_start = self.var_offset_dynamic_b3["Vdr_qd"] + 1
        y0_dyn[dyn_Vdr_d_start] = (2 / 3) * (
            Va * np.sin(theta)
            + Vb * np.sin(theta - 2 * np.pi / 3)
            + Vc * np.sin(theta + 2 * np.pi / 3)
        )

        # # Vref_amp: to be initialized by the amplitude of voltage at Vc
        # dyn_Vref_amp_start = self.var_offset_dynamic_b3["Vref_amp"]
        # y0_dyn[dyn_Vref_amp_start] = np.sqrt(2) * np.abs(y_comp[pf_Vc_start])

        # # V_refc
        # pf_Vc_start = self.var_offset["V"] + self.n_ph
        # pf_Vc_end = pf_Vc_start + self.n_ph
        # dyn_V_refc_start = self.var_offset_dynamic_b3["V_refc"]
        # dyn_V_refc_end = dyn_V_refc_start + self.n_ph
        # y0_dyn[dyn_V_refc_start:dyn_V_refc_end] = np.sqrt(2) * phasor_to_timedomain(
        #     y_comp[pf_Vc_start:pf_Vc_end]
        # )

        # idr_qd
        ig_start_idx = self.var_offset["i"]
        ig_a = np.sqrt(2) * phasor_to_timedomain(y_comp[ig_start_idx])
        ig_b = np.sqrt(2) * phasor_to_timedomain(y_comp[ig_start_idx + 1])
        ig_c = np.sqrt(2) * phasor_to_timedomain(y_comp[ig_start_idx + 2])
        dyn_idr_q_start = self.var_offset_dynamic_b3["idr_qd"]
        y0_dyn[dyn_idr_q_start] = (2 / 3) * (
            ig_a * np.cos(theta)
            + ig_b * np.cos(theta - 2 * np.pi / 3)
            + ig_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_idr_d_start = dyn_idr_q_start + 1
        y0_dyn[dyn_idr_d_start] = (2 / 3) * (
            ig_a * np.sin(theta)
            + ig_b * np.sin(theta - 2 * np.pi / 3)
            + ig_c * np.sin(theta + 2 * np.pi / 3)
        )

        # Pe = self.Pref
        dyn_Pe_start = self.var_offset_dynamic_b3["Pe"]
        y0_dyn[dyn_Pe_start] = self.Pref_total

        # # Qe = self.Qref
        # dyn_Qe_start = self.var_offset_dynamic_b3["Qe"]
        # y0_dyn[dyn_Qe_start] = self.Qref_total

        # Pe_f
        dyn_Pe_f_start = self.var_offset_dynamic_b3["Pe_f"]
        y0_dyn[dyn_Pe_f_start] = self.Pref_total

        # # Qe_f
        # dyn_Qe_f_start = self.var_offset_dynamic_b3["Qe_f"]
        # y0_dyn[dyn_Qe_f_start] = self.Qref_total

        return y0_dyn

    def _initial_guess_dynamic_b4(self, y_comp: list, wnom, y0_dyn: np.ndarray):
        # theta
        pf_Vc_start = self.var_offset["V"] + self.n_ph
        theta = np.angle(y_comp[pf_Vc_start])

        # Vvr_qd = qd transformed Vc
        Vc_start_idx = self.var_offset["V"] + self.n_ph
        Vc_a = np.sqrt(2) * phasor_to_timedomain(y_comp[Vc_start_idx])
        Vc_b = np.sqrt(2) * phasor_to_timedomain(y_comp[Vc_start_idx + 1])
        Vc_c = np.sqrt(2) * phasor_to_timedomain(y_comp[Vc_start_idx + 2])
        dyn_Vvr_q_start = self.var_offset_dynamic_b4["Vvr_cqd"]
        y0_dyn[dyn_Vvr_q_start] = (2 / 3) * (
            Vc_a * np.cos(theta)
            + Vc_b * np.cos(theta - 2 * np.pi / 3)
            + Vc_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_Vvr_d_start = self.var_offset_dynamic_b4["Vvr_cqd"] + 1
        y0_dyn[dyn_Vvr_d_start] = (2 / 3) * (
            Vc_a * np.sin(theta)
            + Vc_b * np.sin(theta - 2 * np.pi / 3)
            + Vc_c * np.sin(theta + 2 * np.pi / 3)
        )

        # # Vvr_cref = Vc obtained from powerflow
        # pf_Vc_start = self.var_offset["V"] + self.n_ph
        # pf_Vc_end = pf_Vc_start + self.n_ph
        # dyn_Vvr_cref_start = self.var_offset_dynamic_b4["Vvr_cref"]
        # dyn_Vvr_cref_end = dyn_Vvr_cref_start + self.n_ph
        # y0_dyn[dyn_Vvr_cref_start:dyn_Vvr_cref_end] = np.sqrt(2) * phasor_to_timedomain(
        #     y_comp[pf_Vc_start:pf_Vc_end]
        # )

        # # Vcref_qd = qd transformed Vc
        # dyn_Vcref_q_start = self.var_offset_dynamic_b4["Vcref_qd"]
        # y0_dyn[dyn_Vcref_q_start] = (2 / 3) * (
        #     Vc_a * np.cos(theta)
        #     + Vc_b * np.cos(theta - 2 * np.pi / 3)
        #     + Vc_c * np.cos(theta + 2 * np.pi / 3)
        # )
        # dyn_Vcref_d_start = self.var_offset_dynamic_b4["Vcref_qd"] + 1
        # y0_dyn[dyn_Vcref_d_start] = (2 / 3) * (
        #     Vc_a * np.sin(theta)
        #     + Vc_b * np.sin(theta - 2 * np.pi / 3)
        #     + Vc_c * np.sin(theta + 2 * np.pi / 3)
        # )

        # vvr_ccqd = qd transformed vcc
        pf_vcc_start = self.var_offset["v"] + 2 * self.n_ph
        vcc_a = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_vcc_start])
        vcc_b = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_vcc_start + 1])
        vcc_c = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_vcc_start + 2])
        dyn_vvr_ccqd_q_idx = self.var_offset_dynamic_b4["vvr_ccqd"]
        dyn_vvr_ccqd_d_idx = dyn_vvr_ccqd_q_idx + 1
        y0_dyn[dyn_vvr_ccqd_q_idx] = (2 / 3) * (
            vcc_a * np.cos(theta)
            + vcc_b * np.cos(theta - 2 * np.pi / 3)
            + vcc_c * np.cos(theta + 2 * np.pi / 3)
        )
        y0_dyn[dyn_vvr_ccqd_d_idx] = (2 / 3) * (
            vcc_a * np.sin(theta)
            + vcc_b * np.sin(theta - 2 * np.pi / 3)
            + vcc_c * np.sin(theta + 2 * np.pi / 3)
        )

        # iref_qd = qd transformed inverter current iin
        iin_start_idx = self.var_offset["i"] + 2 * self.n_ph
        iin_a = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx])
        iin_b = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx + 1])
        iin_c = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx + 2])

        dyn_iref_q_start = self.var_offset_dynamic_b4["iref_qd"]
        y0_dyn[dyn_iref_q_start] = (2 / 3) * (
            iin_a * np.cos(theta)
            + iin_b * np.cos(theta - 2 * np.pi / 3)
            + iin_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_iref_d_start = self.var_offset_dynamic_b4["iref_qd"] + 1
        y0_dyn[dyn_iref_d_start] = (2 / 3) * (
            iin_a * np.sin(theta)
            + iin_b * np.sin(theta - 2 * np.pi / 3)
            + iin_c * np.sin(theta + 2 * np.pi / 3)
        )

        # iref_qd_f = iref_qd (for initialization from steady state)
        dyn_iref_q_f_start = self.var_offset_dynamic_b4["iref_qd_f"]
        y0_dyn[dyn_iref_q_f_start] = (2 / 3) * (
            iin_a * np.cos(theta)
            + iin_b * np.cos(theta - 2 * np.pi / 3)
            + iin_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_iref_d_f_start = self.var_offset_dynamic_b4["iref_qd_f"] + 1
        y0_dyn[dyn_iref_d_f_start] = (2 / 3) * (
            iin_a * np.sin(theta)
            + iin_b * np.sin(theta - 2 * np.pi / 3)
            + iin_c * np.sin(theta + 2 * np.pi / 3)
        )

        # iff_qd = i1 + i2
        i1_start = self.var_offset["i"]
        i1_a = np.sqrt(2) * phasor_to_timedomain(y_comp[i1_start])
        i1_b = np.sqrt(2) * phasor_to_timedomain(y_comp[i1_start + 1])
        i1_c = np.sqrt(2) * phasor_to_timedomain(y_comp[i1_start + 2])
        i1_q = (2 / 3) * (
            i1_a * np.cos(theta)
            + i1_b * np.cos(theta - 2 * np.pi / 3)
            + i1_c * np.cos(theta + 2 * np.pi / 3)
        )
        i1_d = (2 / 3) * (
            i1_a * np.sin(theta)
            + i1_b * np.sin(theta - 2 * np.pi / 3)
            + i1_c * np.sin(theta + 2 * np.pi / 3)
        )

        i2_start = self.var_offset["i"] + self.n_ph
        i2_a = np.sqrt(2) * phasor_to_timedomain(y_comp[i2_start])
        i2_b = np.sqrt(2) * phasor_to_timedomain(y_comp[i2_start + 1])
        i2_c = np.sqrt(2) * phasor_to_timedomain(y_comp[i2_start + 2])
        i2_q = (2 / 3) * (
            i2_a * np.cos(theta)
            + i2_b * np.cos(theta - 2 * np.pi / 3)
            + i2_c * np.cos(theta + 2 * np.pi / 3)
        )
        i2_d = (2 / 3) * (
            i2_a * np.sin(theta)
            + i2_b * np.sin(theta - 2 * np.pi / 3)
            + i2_c * np.sin(theta + 2 * np.pi / 3)
        )

        Vvr_q = y0_dyn[dyn_Vvr_q_start]
        Vvr_d = y0_dyn[dyn_Vvr_d_start]
        w = y0_dyn[self.var_offset_dynamic["w"]]
        C = self.obj.Ca
        # ivr_cq - w*C*Vvr_d = 0
        i2_q_cv = w * C * Vvr_d
        # ivr_cd +  w*C*Vvr_q = 0
        i2_d_cv = -w * C * Vvr_q

        dyn_iff_q_start = self.var_offset_dynamic_b4["iff_qd"]
        y0_dyn[dyn_iff_q_start] = i1_q + i2_q
        dyn_iff_d_start = dyn_iff_q_start + 1
        y0_dyn[dyn_iff_d_start] = i1_d + i2_d

        # # tmp:
        # y0_dyn[dyn_iff_q_start] = y0_dyn[dyn_iref_q_start]
        # y0_dyn[dyn_iff_d_start] = y0_dyn[dyn_iref_d_start]

        # assert abs(iin_a - i1_a - i2_a) < 1e-6
        # # assert iin_b == i1_b + i2_b
        # # assert iin_c == i1_c + i2_c
        # assert abs(y0_dyn[dyn_iff_q_start] - y0_dyn[dyn_iref_q_start]) < 1e-6
        # assert y0_dyn[dyn_iff_d_start] == y0_dyn[dyn_iref_d_start]

        # ivr_cqd = i2 in qd (current across capacitor i2)
        dyn_ivr_cq_start = self.var_offset_dynamic_b4["ivr_cqd"]
        y0_dyn[dyn_ivr_cq_start] = i2_q
        dyn_ivr_cd_start = self.var_offset_dynamic_b4["ivr_cqd"] + 1
        y0_dyn[dyn_ivr_cd_start] = i2_d

        # ivr_g_qd = i1 in qd
        dyn_ivr_gq_start = self.var_offset_dynamic_b4["ivr_gqd"]
        y0_dyn[dyn_ivr_gq_start] = i1_q
        dyn_ivr_gd_start = self.var_offset_dynamic_b4["ivr_gqd"] + 1
        y0_dyn[dyn_ivr_gd_start] = i1_d

        # z_qd, z_qd_f, z_qdoe error terms could be initialized to zero for initial value starting from a steady state

        print(f"iref_q (initialized): {y0_dyn[dyn_iref_q_start]}")
        print(f"iref_d (initialized): {y0_dyn[dyn_iref_d_start]}")
        print(f"iff_q (initialized): {y0_dyn[dyn_iff_q_start]}")
        print(f"iff_d (initialized): {y0_dyn[dyn_iff_d_start]}")
        print(f"ivr_cq (initialized): {y0_dyn[dyn_ivr_cq_start]}")
        print(f"ivr_cd (initialized): {y0_dyn[dyn_ivr_cd_start]}")
        print(f"ivr_gq (initialized): {y0_dyn[dyn_ivr_gq_start]}")
        print(f"inr_gd (initialized): {y0_dyn[dyn_ivr_gd_start]}")
        print(f"i2_q_cv (initialized block): {i2_q_cv}")
        print(f"i2_d_cv (initialized block): {i2_d_cv}")

        return y0_dyn

    # b5. Current Controller
    def _initial_guess_dynamic_b5(self, y_comp: list, wnom, y0_dyn: np.ndarray):
        # Vin_qd, Vin_qdo, iin_cc_qd, icc_qdref, zi_qdo

        # init after b6

        # Vin_qd
        idx_vin_qd_ref_start = self.var_offset_dynamic["Vin_qd_ref"]
        idx_vin_qd_ref_end = idx_vin_qd_ref_start + self.n_varqd
        vin_qd_ref = y0_dyn[idx_vin_qd_ref_start:idx_vin_qd_ref_end]
        idx_vin_qd_start = self.var_offset_dynamic_b5["Vin_qd"]
        idx_vin_qd_end = idx_vin_qd_start + self.n_varqd
        y0_dyn[idx_vin_qd_start:idx_vin_qd_end] = vin_qd_ref

        # Vin_qo
        idx_vin_qdo_start = self.var_offset_dynamic_b5["Vin_qdo"]
        idx_vin_qdo_end = idx_vin_qdo_start + self.n_varqd
        y0_dyn[idx_vin_qdo_start:idx_vin_qdo_end] = vin_qd_ref

        # zin_qdo
        idx_zi_qdo_start = self.var_offset_dynamic_b5["zi_qdo"]
        idx_zi_qdo_end = idx_zi_qdo_start + self.n_varqd
        y0_dyn[idx_zi_qdo_start:idx_zi_qdo_end] = vin_qd_ref

        # iin_cc_qd
        iin_start_idx = self.var_offset["i"] + 2 * self.n_ph
        iin_a = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx])
        iin_b = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx + 1])
        iin_c = np.sqrt(2) * phasor_to_timedomain(y_comp[iin_start_idx + 2])

        # theta
        pf_Vc_start = self.var_offset["V"] + self.n_ph
        theta = np.angle(y_comp[pf_Vc_start])

        dyn_iin_cc_q_start = self.var_offset_dynamic_b5["iin_cc_qd"]
        y0_dyn[dyn_iin_cc_q_start] = (2 / 3) * (
            iin_a * np.cos(theta)
            + iin_b * np.cos(theta - 2 * np.pi / 3)
            + iin_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_iin_cc_d_start = self.var_offset_dynamic_b5["iin_cc_qd"] + 1
        y0_dyn[dyn_iin_cc_d_start] = (2 / 3) * (
            iin_a * np.sin(theta)
            + iin_b * np.sin(theta - 2 * np.pi / 3)
            + iin_c * np.sin(theta + 2 * np.pi / 3)
        )

        # icc_qdref
        dyn_icc_qref_start = self.var_offset_dynamic_b5["icc_qdref"]
        y0_dyn[dyn_icc_qref_start] = (2 / 3) * (
            iin_a * np.cos(theta)
            + iin_b * np.cos(theta - 2 * np.pi / 3)
            + iin_c * np.cos(theta + 2 * np.pi / 3)
        )
        dyn_icc_dref_start = self.var_offset_dynamic_b5["icc_qdref"] + 1
        y0_dyn[dyn_icc_dref_start] = (2 / 3) * (
            iin_a * np.sin(theta)
            + iin_b * np.sin(theta - 2 * np.pi / 3)
            + iin_c * np.sin(theta + 2 * np.pi / 3)
        )

        # zi_qdo error term for PI and could be initialized as zero

        return y0_dyn

    # SVPWM
    def _initial_guess_dynamic_b6(self, y_comp: list, wnom, y0_dyn: np.ndarray):
        # Vin_qd_ref,   Eref,  Es_abc, d, do

        # do
        pf_V3_start = self.var_offset["V"] + 2 * self.n_ph
        pf_iin_start = self.var_offset["i"] + 2 * self.n_ph
        va = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start])
        vb = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start + 1])
        vc = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start + 2])
        ia = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_iin_start])
        ib = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_iin_start + 1])
        ic = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_iin_start + 2])

        Vdc = self.obj.Vdc
        vsw = self.obj.Vsw
        rsw = self.obj.rsw
        vd = self.obj.Vd
        rd = self.obj.rd

        dyn_do_a_start = self.var_offset_dynamic_b6["do"]
        dyn_do_b_start = self.var_offset_dynamic_b6["do"] + 1
        dyn_do_c_start = self.var_offset_dynamic_b6["do"] + 2

        # compute duty cycle
        # a phase:
        # if ia >= 0:
        #     numerator_a = va + vd + rd * ia
        #     denominator_a = Vdc - vsw - rsw * ia + vd + rd * ia
        # else:
        #     numerator_a = va - vsw + rsw * ia
        #     denominator_a = Vdc + vd - rd * ia - vsw + rsw * ia
        # do_a = numerator_a / denominator_a

        # do_a = 2 * va / Vdc
        do_a = va / Vdc

        # b phase
        # if ib >= 0:
        #     numerator_b = vb + vd + rd * ib
        #     denominator_b = Vdc - vsw - rsw * ib + vd + rd * ib
        # else:
        #     numerator_b = vb - vsw + rsw * ib
        #     denominator_b = Vdc + vd - rd * ib - vsw + rsw * ib
        # do_b = numerator_b / denominator_b

        # do_b = 2 * vb / Vdc
        do_b = vb / Vdc

        # c phase
        # if ic >= 0:
        #     numerator_c = vc + vd + rd * ic
        #     denominator_c = Vdc - vsw - rsw * ic + vd + rd * ic
        # else:
        #     numerator_c = vc - vsw + rsw * ic
        #     denominator_c = Vdc + vd - rd * ic - vsw + rsw * ic
        # do_c = numerator_c / denominator_c

        # do_c = 2 * vc / Vdc
        do_c = vc / Vdc

        # initialize using above computed values
        y0_dyn[dyn_do_a_start] = do_a
        y0_dyn[dyn_do_b_start] = do_b
        y0_dyn[dyn_do_c_start] = do_c

        print(f"do_a : {do_a}")
        print(f"do_b : {do_b}")
        print(f"do_c : {do_c}")

        # d: do = 1/2 + d/2
        # therefor d = 2*do-1
        # d_a = 2*do_a - 1
        # d_b = 2*do_b - 1
        # d_c = 2*do_c - 1
        d_a = do_a
        d_b = do_b
        d_c = do_c

        print(f"d_a : {d_a}")
        print(f"d_b : {d_b}")
        print(f"d_c : {d_c}")
        dyn_d_a_start = self.var_offset_dynamic_b6["d"]
        dyn_d_b_start = self.var_offset_dynamic_b6["d"] + 1
        dyn_d_c_start = self.var_offset_dynamic_b6["d"] + 2

        y0_dyn[dyn_d_a_start] = d_a
        y0_dyn[dyn_d_b_start] = d_b
        y0_dyn[dyn_d_c_start] = d_c

        # Es_abc
        # compute from the duty cycle so obtained
        # Es_abc = dabc*(Vdc/2)

        dyn_Es_a_start = self.var_offset_dynamic_b6["Es_abc"]
        dyn_Es_b_start = self.var_offset_dynamic_b6["Es_abc"] + 1
        dyn_Es_c_start = self.var_offset_dynamic_b6["Es_abc"] + 2

        # Es_a = do_a * (Vdc/2)
        # Es_b = do_b * (Vdc/2)
        # Es_c = do_c * (Vdc/2)

        Es_a = d_a * (Vdc / 2)
        Es_b = d_b * (Vdc / 2)
        Es_c = d_c * (Vdc / 2)

        y0_dyn[dyn_Es_a_start] = Es_a
        y0_dyn[dyn_Es_b_start] = Es_b
        y0_dyn[dyn_Es_c_start] = Es_c

        # tmp::
        print(f"Es_a : {Es_a}")
        print(f"Es_b : {Es_b}")
        print(f"Es_c : {Es_c}")

        # Eref
        # to compute Eref from Es_abc we need to estimate common mode injection
        # for initialization purpose we use SPWM common mode injection
        Emid_est = (Es_a + Es_b + Es_c) / 3
        # Emid = 1853.48
        Eref_a = Es_a
        Eref_b = Es_b
        Eref_c = Es_c

        dyn_Eref_a_start = self.var_offset_dynamic_b6["Eref"]
        dyn_Eref_b_start = self.var_offset_dynamic_b6["Eref"] + 1
        dyn_Eref_c_start = self.var_offset_dynamic_b6["Eref"] + 2

        y0_dyn[dyn_Eref_a_start] = Eref_a
        y0_dyn[dyn_Eref_b_start] = Eref_b
        y0_dyn[dyn_Eref_c_start] = Eref_c

        # tmp ::
        print(f"Emid_est : {Emid_est}")
        print(f"Eref_a : {Eref_a}")
        print(f"Eref_b : {Eref_b}")
        print(f"Eref_c : {Eref_c}")

        pf_Vc_start = self.var_offset["V"] + self.n_ph
        theta = np.angle(y_comp[pf_Vc_start])

        # Vin_qd_ref: obtained from current controller. Initialized using Eref obtained above.
        dyn_Vin_q_ref_start = self.var_offset_dynamic["Vin_qd_ref"]
        dyn_Vin_d_ref_start = self.var_offset_dynamic["Vin_qd_ref"] + 1
        y0_dyn[dyn_Vin_q_ref_start] = (2 / 3) * (
            Eref_a * np.cos(theta)
            + Eref_b * np.cos(theta - 2 * np.pi / 3)
            + Eref_c * np.cos(theta + 2 * np.pi / 3)
        )
        y0_dyn[dyn_Vin_d_ref_start] = (2 / 3) * (
            Eref_a * np.sin(theta)
            + Eref_b * np.sin(theta - 2 * np.pi / 3)
            + Eref_c * np.sin(theta + 2 * np.pi / 3)
        )

        # tmp ::
        print(f"Vin_q = {y0_dyn[dyn_Vin_q_ref_start]}")
        print(f"Vin_d = {y0_dyn[dyn_Vin_d_ref_start]}")

        # # To initialize Emid and Eref :
        # # Use SPWM common mode injection to estimate that for SVPWM
        # # add the same to inverter output from powerflow to estimate original ref
        # pf_V3_start = self.var_offset["V"] + 2*self.n_ph
        # pf_V3_end = pf_V3_start + self.n_ph
        # V3_a = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start])
        # V3_b = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start + 1])
        # V3_c = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start + 2])
        # Emid = (V3_a + V3_b + V3_c)/3 # estimate for common mode injection
        # # tmp::
        # print(f"V3a : {V3_a}")
        # print(f"V3b : {V3_b}")
        # print(f"V3c : {V3_c}")
        # print(f"Emid : {Emid}")

        # # Eref : Es_abc=Eref_abc - Emid
        # # Eref = Es_abc + Emid (Es_abc is initialized as inverter output voltage)
        # dyn_Erefa_start = self.var_offset_dynamic_b6["Eref"]
        # Erefa = V3_a + Emid/2
        # y0_dyn[dyn_Erefa_start] = Erefa
        # dyn_Erefb_start = self.var_offset_dynamic_b6["Eref"] + 1
        # Erefb = V3_b + Emid/2
        # y0_dyn[dyn_Erefb_start] = Erefb
        # dyn_Erefc_start = self.var_offset_dynamic_b6["Eref"] + 2
        # Erefc = V3_c + Emid/2
        # y0_dyn[dyn_Erefc_start] = Erefc

        # # Emax

        # dyn_Emax_start = self.var_offset_dynamic_b6["Emax"]
        # Emax = max(Erefa, Erefb, Erefc)
        # y0_dyn[dyn_Emax_start] = Emax

        # # Emin
        # dyn_Emin_start = self.var_offset_dynamic_b6["Emin"]
        # Emin = min(Erefa, Erefb, Erefc)
        # y0_dyn[dyn_Emin_start] = Emin

        # # Emid
        # dyn_Emid_start = self.var_offset_dynamic_b6["Emid"]
        # y0_dyn[dyn_Emid_start] = Emid

        # # Es_abc
        # dyn_Es_abc_start = self.var_offset_dynamic_b6["Es_abc"]
        # dyn_Es_abc_end = dyn_Es_abc_start + self.n_ph
        # y0_dyn[dyn_Es_abc_start:dyn_Es_abc_end] = np.sqrt(2) * phasor_to_timedomain(
        #     y_comp[pf_V3_start:pf_V3_end]
        # ) - Emid

        # # Vin_qd_ref = Eref in qd transform

        # dyn_Vin_q_ref_start = self.var_offset_dynamic_b6["Vin_qd_ref"]
        # y0_dyn[dyn_Vin_q_ref_start] = (2 / 3) * (
        #     Erefa * np.cos(0)
        #     + Erefb * np.cos(-2 * np.pi / 3)
        #     + Erefc * np.cos(2 * np.pi / 3)
        # )
        # dyn_Vin_d_ref_start = self.var_offset_dynamic_b6["Vin_qd_ref"] + 1
        # y0_dyn[dyn_Vin_d_ref_start] = (2 / 3) * (
        #     Erefa * np.sin(0)
        #     + Erefb * np.sin(-2 * np.pi / 3)
        #     + Erefc * np.sin(2 * np.pi / 3)
        # )

        # # d

        # # resume here
        # dyn_da_start = self.var_offset_dynamic_b6["d"]
        # dyn_db_start = self.var_offset_dynamic_b6["d"] + 1
        # dyn_dc_start = self.var_offset_dynamic_b6["d"] + 2
        # da = 2 * (V3_a) / self.obj.Vdc
        # db = 2 * (V3_b) / self.obj.Vdc
        # dc = 2 * (V3_c) / self.obj.Vdc

        # y0_dyn[dyn_da_start] = da
        # y0_dyn[dyn_db_start] = db
        # y0_dyn[dyn_dc_start] = dc

        # # do duty cycle output
        # dyn_do_a_start = self.var_offset_dynamic_b6["do"]
        # dyn_do_b_start = self.var_offset_dynamic_b6["do"] + 1
        # dyn_do_c_start = self.var_offset_dynamic_b6["do"] + 2
        # y0_dyn[dyn_do_a_start] = (1 / 2) + (da / 2)
        # y0_dyn[dyn_do_b_start] = (1 / 2) + (db / 2)
        # y0_dyn[dyn_do_c_start] = (1 / 2) + (dc / 2)

        return y0_dyn

    # b7. Inverter bridge
    def _initial_guess_dynamic_b7(self, y_comp: list, wnom, y0_dyn: np.ndarray):
        # E_abc, v_abc, i_abc, din, Vdc

        # E_abc = V3 inverter output voltage
        pf_V3_start = self.var_offset["V"] + 2 * self.n_ph
        pf_V3_end = pf_V3_start + self.n_ph
        dyn_E_abc_start = self.var_offset_dynamic_b7["E_abc"]
        dyn_E_abc_end = dyn_E_abc_start + self.n_ph
        y0_dyn[dyn_E_abc_start:dyn_E_abc_end] = np.sqrt(2) * phasor_to_timedomain(
            y_comp[pf_V3_start:pf_V3_end]
        )

        # v_abc = V3 inverter output voltage
        dyn_v_abc_start = self.var_offset_dynamic_b7["v_abc"]
        dyn_v_abc_end = dyn_v_abc_start + self.n_ph
        y0_dyn[dyn_v_abc_start:dyn_v_abc_end] = np.sqrt(2) * phasor_to_timedomain(
            y_comp[pf_V3_start:pf_V3_end]
        )

        # i_abc = iin
        pf_iin_start = self.var_offset["i"] + 2 * self.n_ph
        pf_iin_end = pf_iin_start + self.n_ph
        dyn_i_abc_start = self.var_offset_dynamic_b7["i_abc"]
        dyn_i_abc_end = dyn_i_abc_start + self.n_ph
        y0_dyn[dyn_i_abc_start:dyn_i_abc_end] = np.sqrt(2) * phasor_to_timedomain(
            y_comp[pf_iin_start:pf_iin_end]
        )

        # din
        # fx = (Vdc-vsw-ix*rsw)*din_x - (vd + ix*rd)*(1-din_x) for ix>=0
        # fx = (Vdc + vd+ix*rd)*din_x + (vsw-rsw*ix)*(1-din_x) for ix<0
        # using above we initialize din in accordance with the value of ix
        va = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start])
        vb = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start + 1])
        vc = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_V3_start + 2])
        ia = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_iin_start])
        ib = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_iin_start + 1])
        ic = np.sqrt(2) * phasor_to_timedomain(y_comp[pf_iin_start + 2])

        Vdc = self.obj.Vdc
        vsw = self.obj.Vsw
        rsw = self.obj.rsw
        vd = self.obj.Vd
        rd = self.obj.rd

        dyn_din_a_start = self.var_offset_dynamic["din"]
        dyn_din_b_start = self.var_offset_dynamic["din"] + 1
        dyn_din_c_start = self.var_offset_dynamic["din"] + 2

        print(f"va:{va}, vb:{vb}, vc:{vc}")
        print(f"ia:{ia}, ib:{ib}, ic:{ic}")

        # compute duty cycle
        # a phase:
        # if ia >= 0:
        #     numerator_a = va + vd + rd * ia
        #     denominator_a = Vdc - vsw - rsw * ia + vd + rd * ia

        # else:
        #     numerator_a = va - vsw + rsw * ia
        #     denominator_b = Vdc + vd + rd * ia - vsw + rsw * ia
        # da = numerator_a / denominator_a

        # a new:
        # da = 2 * va / Vdc
        da = va / Vdc

        # b phase
        # if ib >= 0:
        #     numerator_b = vb + vd + rd * ib
        #     denominator_b = Vdc - vsw - rsw * ib + vd + rd * ib

        # else:
        #     numerator_b = vb - vsw + rsw * ib
        #     denominator_b = Vdc + vd + rd * ib - vsw + rsw * ib
        # db = numerator_b / denominator_b

        # b new:
        # db = 2 * vb / Vdc
        db = vb / Vdc

        # c phase
        # if ic >= 0:
        #     numerator_c = vc + vd + rd * ic
        #     denominator_c = Vdc - vsw - rsw * ic + vd + rd * ic

        # else:
        #     numerator_c = vc - vsw + rsw * ic
        #     denominator_c = Vdc + vd + rd * ic - vsw + rsw * ic
        # dc = numerator_c / denominator_c

        # c new:
        # dc = 2 * vc / Vdc
        dc = vc / Vdc

        # initialize using above computed values
        y0_dyn[dyn_din_a_start] = da
        y0_dyn[dyn_din_b_start] = db
        y0_dyn[dyn_din_c_start] = dc

        assert -1 <= da and da <= 1, f"da = {da}"
        assert -1 <= db and db <= 1, f"db = {db}"
        assert -1 <= dc and dc <= 1, f"dc = {dc}"

        # # tmp::
        # idx_vag = self.var_offset_dynamic["Vinf"]
        # vag = y0_dyn[idx_vag]
        # idx_iag = self.var_offset_dynamic["iinf"]
        # iag = y0_dyn[idx_iag]
        # numerator = vag + self.obj.Vd + iag * self.obj.rd
        # denominator = self.obj.Vdc - self.obj.Vsw - iag * self.obj.rsw + self.obj.Vd + iag * self.obj.rd
        # dag = numerator / denominator
        # print(f"da: {da}, dag: {dag}")

        # # Vdc
        # dyn_Vdc_start = self.var_offset_dynamic_b7["Vdc"]
        # y0_dyn[dyn_Vdc_start] = Vdc

        return y0_dyn

    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn

    def initial_yp_dynamic(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        idx_theta = self.var_offset_dynamic["theta"]
        yp0_dyn[idx_theta] = wnom

        # b1: 9)-icf + d(qf)/dt=0
        idx_qf_start = self.var_offset_dynamic["qf"]
        idx_qf_end = idx_qf_start + self.n_ph
        idx_icf_start = self.var_offset_dynamic["icf"]
        idx_icf_end = idx_icf_start + self.n_ph
        yp0_dyn[idx_qf_start:idx_qf_end] = y0_dyn_comp[idx_icf_start:idx_icf_end]

        # b1: 12)-vinf+r_in*iinf + d(lamda_in)/dt=0
        vinf_start, vinf_end = get_start_end_idx(
            self.var_offset_dynamic, "vinf", self.n_ph
        )
        vinf = y0_dyn_comp[vinf_start:vinf_end]
        iinf_start, iinf_end = get_start_end_idx(
            self.var_offset_dynamic, "iinf", self.n_ph
        )
        iinf = y0_dyn_comp[iinf_start:iinf_end]
        lamda_in_start, lamda_in_end = get_start_end_idx(
            self.var_offset_dynamic, "lamda_in", self.n_ph
        )
        r_in = self.r3_mat
        yp0_dyn[lamda_in_start:lamda_in_end] = vinf - r_in @ iinf

        # b1: 3)-vgf + rg*igf + d(lamda_g)/dt=0
        vgf_start, vgf_end = get_start_end_idx(
            self.var_offset_dynamic, "vgf", self.n_ph
        )
        vgf = y0_dyn_comp[vgf_start:vgf_end]
        igf_start, igf_end = get_start_end_idx(
            self.var_offset_dynamic, "igf", self.n_ph
        )
        igf = y0_dyn_comp[igf_start:igf_end]
        lamda_g_start, lamda_g_end = get_start_end_idx(
            self.var_offset_dynamic, "lamda_g", self.n_ph
        )
        rg = self.r1_mat
        yp0_dyn[lamda_g_start:lamda_g_end] = vgf - rg @ igf

        # # b3: 8) -v + rg*ig + d(lamda_vir)/dt
        # vvir_start, vvir_end = get_start_end_idx(self.var_offset_dynamic, "v_vir", self.n_ph)
        # v_vir = y0_dyn_comp[vvir_start:vvir_end]
        # idr_g_start, idr_g_end = get_start_end_idx(self.var_offset_dynamic, "idr_g", self.n_ph)
        # idr_g = y0_dyn_comp[idr_g_start:idr_g_end]
        # rg = self.r1_mat
        # l_vir_start, l_vir_end = get_start_end_idx(self.var_offset_dynamic, "l_vir", self.n_ph)
        # yp0_dyn[l_vir_start:l_vir_end] = v_vir - rg @ idr_g

        # # b3: 11) Vrefc_p - d(Vrefc[a])/dt = 0
        # dyn_Vrefc_p = self.var_offset_dynamic["Vrefc_p"]
        # idx_Vrefc_a = self.var_offset_dynamic["V_refc"]
        # yp0_dyn[idx_Vrefc_a] = y0_dyn_comp[dyn_Vrefc_p]

        return yp0_dyn

    def initial_guess_dynamic_zero(self, y_comp: list, wnom) -> np.ndarray:
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

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

        # fix at 20KW to validate
        # self.Pref_total = 20e3 * 3
        # self.Qref_total = 0
        self.Pref_total = sum(self.Pref[i] for i in range(len(self.Pref)))
        self.Qref_total = sum(self.Qref[i] for i in range(len(self.Qref)))

        # pf_Vc_start = self.var_offset["V"] + self.n_ph
        # pf_Vc_end = pf_Vc_start + self.n_ph
        # dyn_Vc_start = self.var_offset_dynamic["Vcf"]
        # dyn_Vc_end = dyn_Vc_start + self.n_ph
        # y0_dyn[dyn_Vc_start:dyn_Vc_end] = np.sqrt(2) * phasor_to_timedomain(
        #     y_comp[pf_Vc_start:pf_Vc_end]
        # )

        # dyn_idx_Pe = self.var_offset_dynamic["Pe"]
        # y0_dyn[dyn_idx_Pe] = self.Pref_total

        # dyn_idx_Qe = self.var_offset_dynamic["Qe"]
        # y0_dyn[dyn_idx_Qe] = self.Qref_total

        # d = 0.5
        # da = d * np.cos(0)
        # db = d * np.cos(-2 * np.pi / 3)
        # dc = d * np.cos(2 * np.pi / 3)
        # dyn_idx_d_start, dyn_idx_d_end = get_start_end_idx(self.var_offset_dynamic, "d", self.n_ph)
        # y0_dyn[dyn_idx_d_start:dyn_idx_d_end] = np.array([da, db, dc])
        # # dyn_idx_do_start, dyn_idx_do_end = get_start_end_idx(self.var_offset_dynamic, "do", self.n_ph)
        # # y0_dyn[dyn_idx_do_start:dyn_idx_do_end] = np.array([da, db, dc])
        # dyn_idx_din_start, dyn_idx_din_end = get_start_end_idx(self.var_offset_dynamic, "din", self.n_ph)
        # y0_dyn[dyn_idx_din_start:dyn_idx_din_end] = np.array([da, db, dc])

        return y0_dyn

    def initial_guess_dynamic(self, y_comp: list, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

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

        self.Pref_total = 20e3 * 3
        self.Qref_total = 0
        # self.Pref_total = sum(self.Pref[i] for i in range(len(self.Pref)))
        # self.Qref_total = sum(self.Qref[i] for i in range(len(self.Qref)))

        # 2. init vars
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        # self._initial_guess_dynamic_b7(y_comp, wnom, y0_dyn)
        # self._initial_guess_dynamic_b6(y_comp, wnom, y0_dyn)
        # self._initial_guess_dynamic_b5(y_comp, wnom, y0_dyn)

        # # ---

        # self._initial_guess_dynamic_b1(y_comp, wnom, y0_dyn)
        # self._initial_guess_dynamic_b2(y_comp, wnom, y0_dyn)
        # self._initial_guess_dynamic_b3(y_comp, wnom, y0_dyn)
        # self._initial_guess_dynamic_b4(y_comp, wnom, y0_dyn)  

        ###################################

        def copy_values_pf_to_dyn(
            from_key: str,
            from_start_offset: int,
            count: int,
            to_key: str,
            to_start_offset: int,
        ):
            from_start, from_end = get_start_end_idx_with_offset(self.var_offset, from_key, from_start_offset, count)
            to_start, to_end = get_start_end_idx_with_offset(self.var_offset_dynamic, to_key, to_start_offset, count)
            y0_dyn[to_start:to_end] = np.sqrt(2) * phasor_to_timedomain(y_comp[from_start:from_end])

        def copy_values_dyn_to_dyn(
            from_key: str,
            from_start_offset: int,
            count: int,
            to_key: str,
            to_start_offset: int,
        ):
            from_start, from_end = get_start_end_idx_with_offset(self.var_offset_dynamic, from_key, from_start_offset, count)
            to_start, to_end = get_start_end_idx_with_offset(self.var_offset_dynamic, to_key, to_start_offset, count)
            y0_dyn[to_start:to_end] = y0_dyn[from_start:from_end]

        # dyn -> dyn operation
        def transform_abc_to_qd(
            from_key: str,
            from_start_offset: int,
            to_key: str,
            to_start_offset: int,
            theta: float,
            base: float
        ):
            idx_abc_start, idx_abc_end = get_start_end_idx_with_offset(self.var_offset_dynamic, from_key, from_start_offset, 3)
            abc = y0_dyn[idx_abc_start:idx_abc_end]
            qd = abc_to_qd(abc, theta)
            idx_qd_start, idx_qd_end = get_start_end_idx_with_offset(self.var_offset_dynamic, to_key, to_start_offset, 2)
            y0_dyn[idx_qd_start:idx_qd_end] = qd / base

        def dyn_var_array(key: str, start_offset: int, count: int) -> np.ndarray:
            idx_start, idx_end = get_start_end_idx_with_offset(self.var_offset_dynamic, key, start_offset, count)
            return y0_dyn[idx_start:idx_end]

        ######################################################################

        # 1. LCL filter
        # V, Vcf, Vinf, Vab, Vbc, I, Iinf, vgf, vcf, vinf, vccf, igf, icf, iinf, qf, lamda_g, lamda_in

        copy_values_pf_to_dyn("V", 0, 3 * self.n_ph, "V", 0)      # V -> V, Vcf, Vinf

        # Vab, Vbc
        Va = y0_dyn[self.var_offset_dynamic["Vinf"]]
        Vb = y0_dyn[self.var_offset_dynamic["Vinf"] + 1]
        Vc = y0_dyn[self.var_offset_dynamic["Vinf"] + 2]
        dyn_Vab_start, dyn_Vab_end = get_start_end_idx(self.var_offset_dynamic, "Vab", 1)
        dyn_Vbc_start, dyn_Vbc_end = get_start_end_idx(self.var_offset_dynamic, "Vbc", 1)
        y0_dyn[dyn_Vab_start:dyn_Vab_end] = Va - Vb
        y0_dyn[dyn_Vbc_start: dyn_Vbc_end] = Vb - Vc

        copy_values_pf_to_dyn("I", 0, self.n_ph, "I", 0)                # I -> I
        copy_values_pf_to_dyn("I", 2*self.n_ph, self.n_ph, "Iinf", 0 )  # I[6] -> Iinf

        # vgf = v1, vcf = v2(self.n_ph), vinf = v3(3*self.n_ph), vccf = vcc(2*self.n_ph)
        copy_values_pf_to_dyn("v", 0, self.n_ph, "vgf", 0)              # v1 -> vgf
        copy_values_pf_to_dyn("v", self.n_ph, self.n_ph, "vcf", 0)      # v2 -> vcf
        copy_values_pf_to_dyn("v", 3*self.n_ph, self.n_ph, "vinf", 0)      # v3 -> vinf
        copy_values_pf_to_dyn("v", 2*self.n_ph, self.n_ph, "vccf", 0 )  # vcc -> vccf

        copy_values_pf_to_dyn("i", 0, self.n_ph, "igf", 0)              # i[0] -> igf
        copy_values_pf_to_dyn("i", self.n_ph, self.n_ph, "icf", 0)      # i[3] -> icf
        copy_values_pf_to_dyn("i", 2*self.n_ph, self.n_ph, "iinf", 0)   # i[6] -> iinf

        copy_values_pf_to_dyn("q", 0, self.n_ph, "qf", 0)   # q -> qf

        copy_values_pf_to_dyn("lamda", 0, self.n_ph, "lamda_g", 0)          # lamda[0] -> lamda_g
        copy_values_pf_to_dyn("lamda", self.n_ph, self.n_ph, "lamda_in", 0) # lamda[3] -> lamda_in

        ######################################################################

        # 2. m&t block
        #[Vc,  vcc,  Vc_qd,  vcc_qd,  ig,    iin,    ig_qd,  iin_qd ]

        theta = 0

        copy_values_dyn_to_dyn("Vcf", 0, self.n_ph, "Vc", 0)    # dyn Vcf -> Vc
        copy_values_dyn_to_dyn("vccf", 0, self.n_ph, "vcc", 0)  # dyn vccf -> vcc

        transform_abc_to_qd("Vc", 0, "Vc_qd", 0, theta, self.Vqd_b)    # dyn Vc -> xform -> Vc_qd
        transform_abc_to_qd("vcc", 0, "vcc_qd", 0, theta, self.Vqd_b)  # dyn vcc -> xform -> vcc_qd
        
        copy_values_dyn_to_dyn("igf", 0, self.n_ph, "ig", 0)     # dyn igf -> ig
        copy_values_dyn_to_dyn("iinf", 0, self.n_ph, "iin", 0)   # dyn iinf -> iin

        transform_abc_to_qd("ig", 0, "ig_qd", 0, theta, self.Iqd_b)    # dyn ig -> xform -> ig_qd
        transform_abc_to_qd("iin", 0, "iin_qd", 0, theta, self.Iqd_b)  # dyn iin -> xform -> iin_qd

        ######################################################################

        # brf
        # I_qd,   Ibrf1_qd, Ibrf2_qd, Ibrf3_qd, I_qd_dc

        w_reject = 2 * const.w_nominal
        w2 = 1 / w_reject**2

        copy_values_dyn_to_dyn("ig_qd", 0, 2, "I_qd", 0) # -> I_qd
        
        # -> Ibrf1_qd
        idx_Ibrf1_qd_start = self.var_offset_dynamic["Ibrf1_qd"]
        idx_Ibrf1_qd_end = idx_Ibrf1_qd_start + self.n_varqd
        idx_igqd_start, idx_igqd_end = get_start_end_idx(self.var_offset_dynamic, "ig_qd", 2)
        y0_dyn[idx_Ibrf1_qd_start:idx_Ibrf1_qd_end] = w2 * y0_dyn[idx_igqd_start:idx_igqd_end]
        
        # 0 -> Ibrf2_qd
        # 0 -> Ibrf3_qd

        copy_values_dyn_to_dyn("ig_qd", 0, 2, "I_qd_dc", 0) # -> I_qd_dc

        ######################################################################

        # 3. pw droop
        # w,   w_pu,      Vdr_qd, idr_qd, Pe,        Pe_f,    theta           

        y0_dyn[self.var_offset_dynamic["w"]] = const.w_nominal     # nominal freq -> w
        y0_dyn[self.var_offset_dynamic["w_pu"]] = 1                  # 1 -> w_pu
        
        copy_values_dyn_to_dyn("Vc_qd", 0, 2, "Vdr_qd", 0)  # dyn Vc_qd -> Vdr_qd
        copy_values_dyn_to_dyn("I_qd_dc", 0, 2, "idr_qd", 0)  # dyn I_qd_dc -> idr_qd

        y0_dyn[self.var_offset_dynamic["Pe"]] = self.Pref_total / self.Pb     # -> Pe
        y0_dyn[self.var_offset_dynamic["Pe_f"]] = self.Pref_total / self.Pb   # -> Pe_f
        
        y0_dyn[self.var_offset_dynamic["theta"]] = 0    # 0 -> theta

        ######################################################################

        # qv droop
        # V_qvdroop_qd, i_qvdroop_qd,  Qe,       Qe_f,    Vref_qd

        copy_values_dyn_to_dyn("Vc_qd", 0, 2, "V_qvdroop_qd", 0) # V_qvdroop_qd, 
        copy_values_dyn_to_dyn("ig_qd", 0, 2, "i_qvdroop_qd", 0) # i_qvdroop_qd,  
        
        y0_dyn[self.var_offset_dynamic["Qe"]] = self.Qref_total / self.Pb       # Qe,       
        y0_dyn[self.var_offset_dynamic["Qe_f"]] = self.Qref_total / self.Pb     # Qe_f,    

        y0_dyn[self.var_offset_dynamic["Vref_qd"]] = self.nominal_voltage_rms * np.sqrt(2) / self.Vqd_b   # Vref_qd
        y0_dyn[self.var_offset_dynamic["Vref_qd"] + 1] = 0

        ######################################################################

        # 4. voltage controller
        # Vvr_cqd,   Vcref_qd,  vvr_ccqd   iref_qd, iref_qd_f, iff_qd, ivr_cqd, ivr_gqd, z_qd, z_qdf   zvr_qdo   
        
        copy_values_dyn_to_dyn("Vc_qd", 0, 2, "Vvr_cqd", 0)     # Vvr_cqd,   
        copy_values_dyn_to_dyn("Vref_qd", 0, 2, "Vcref_qd", 0)  # Vcref_qd,  
        copy_values_dyn_to_dyn("vcc_qd", 0, 2, "vvr_ccqd", 0)   # vvr_ccqd   
        
        copy_values_dyn_to_dyn("iin_qd", 0, 2, "iref_qd", 0)     # iref_qd, 
        copy_values_dyn_to_dyn("iref_qd", 0, 2, "iref_qd_f", 0)   # iref_qd_f, 
        
        copy_values_dyn_to_dyn("iin_qd", 0, 2, "iff_qd", 0)     # iff_qd, 
        
        # ivr_cqd, 
        dyn_ivrcqd_start, dyn_ivrcqd_end = get_start_end_idx(self.var_offset_dynamic, "ivr_cqd", 2)
        iin_qd = dyn_var_array("iin_qd", 0, 2)
        ig_qd = dyn_var_array("ig_qd", 0, 2)
        y0_dyn[dyn_ivrcqd_start: dyn_ivrcqd_end] = iin_qd - ig_qd

        copy_values_dyn_to_dyn("ig_qd", 0, 2, "ivr_gqd", 0) # ivr_gqd, 
        
        # 0 -> z_qd, 
        # 0 -> z_qdf 
        # 0 -> zvr_qdo         

        ######################################################################

        # 5. current controller
        #Vin_qd,   Vin_qdo,    iin_cc_qd,  icc_qdref,  zi_qdo

        # Vin_qd,
        Vinf = dyn_var_array("Vinf", 0, self.n_ph)
        Vinf_qd = abc_to_qd(Vinf, theta) / self.Vqd_b
        idx_Vinqd_start, idx_Vinqd_end = get_start_end_idx(self.var_offset_dynamic, "Vin_qd", 2)
        y0_dyn[idx_Vinqd_start:idx_Vinqd_end] = Vinf_qd

        copy_values_dyn_to_dyn("Vin_qd", 0, 2, "Vin_qdo", 0) # Vin_qdo,   

        copy_values_dyn_to_dyn("iin_qd", 0, 2, "iin_cc_qd", 0) # iin_cc_qd,  
        copy_values_dyn_to_dyn("iin_qd", 0, 2, "icc_qdref", 0) # icc_qdref,  

        copy_values_dyn_to_dyn("Vin_qd", 0, 2, "zi_qdo", 0) # zi_qdo

        ######################################################################

        # 6. svpwm
        # m_qd, m_qd_f, v_qd_ref,   m_abc

        # m_qd, 
        Vin_qdo = dyn_var_array("Vin_qdo", 0, 2)
        idx_mqd_start, idx_mqd_end = get_start_end_idx(self.var_offset_dynamic, "m_qd", 2)
        y0_dyn[idx_mqd_start:idx_mqd_end] = Vin_qdo * (1 / self.obj.Vdc) * self.Vqd_b

        copy_values_dyn_to_dyn("m_qd", 0, 2, "m_qd_f", 0) # m_qd_f, 

        copy_values_dyn_to_dyn("Vin_qdo", 0, 2, "v_qd_ref", 0) # v_qd_ref,   

        m_qd_f = dyn_var_array("m_qd_f", 0, 2) # m_abc
        m_abc = qd_to_abc(m_qd_f, theta)
        idx_mabc_start, idx_mabc_end = get_start_end_idx(self.var_offset_dynamic, "m_abc", 3)
        y0_dyn[idx_mabc_start:idx_mabc_end] = m_abc

        ######################################################################

        # 7. inverter bridge
        # E_abc, v_abc,   Eab,     Ebc,    i_abc,  din,                
        
        copy_values_dyn_to_dyn("Vinf", 0, self.n_ph, "E_abc", 0) # E_abc, 
        copy_values_dyn_to_dyn("Vinf", 0, self.n_ph, "v_abc", 0) # v_abc,   
        copy_values_dyn_to_dyn("Vab", 0, 1, "Eab", 0) # Eab,     
        copy_values_dyn_to_dyn("Vbc", 0, 1, "Eab", 0) # Ebc,    
        copy_values_dyn_to_dyn("iinf", 0, 1, "i_abc", 0) # i_abc,  
        copy_values_dyn_to_dyn("m_abc", 0 , self.n_ph, "din", 0) # din,                

        return y0_dyn

    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        return self.M_dynamic

    def get_local_idx_dynamic(
        self, var: str, ph: str | None = None, side: NodeSide | None = None
    ):
        vars = list(self.var_offset_dynamic.keys())
        assert var in self.var_offset_dynamic.keys(), (
            f"key {var} not found in self.var_offset_dynamic {vars}"
        )

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset
