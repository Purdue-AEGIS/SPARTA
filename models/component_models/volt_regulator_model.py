import numpy as np
import scipy.sparse as sps
from scipy.sparse import diags
from oodesign import Load
from const import NodeSide, StudyType
from models.model import Model, ValType
import utils
from pprint import pformat
from models.component_models.equipment_model import EquipmentModel


"""
1. This file presently models an autotransformer with a fixed tap setting.
2. This would be further developed as a full fledged voltage regulator with LDC etc.
3. This class presently represents Type B voltage regulator circuit without considering
    the leakage and winding impedances.
4. The effective turns ratio based on given voltage regulator data and active reactive load
    in the network is calculated and the model below represents an auto-transformer with that 
    effective ratio.
"""


class VoltRegulatorModel(EquipmentModel):
    def __init__(self, obj):
        super().__init__(obj)

        self.num_term = self.get_num_term()

        # book keeping of the equations
        self.num_eqns_real = 0
        # fmt: off
        self.num_eqns_complex = (
            2*self.n_ph # 1) -I + Ai = 0
            + 2*self.n_ph # 2) A'V - v = 0
            + self.n_ph # 3) voltage transformation          
            + self.n_ph # 4) current transformation  
        )
        # fmt: on
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book keeping of the variables
        self.num_vars_real = 1
        # fmt: off
        self.num_vars_complex = (
            2 * self.n_ph +
            2 * self.n_ph +
            2 * self.n_ph +
            2 * self.n_ph
        )
        # fmt: on
        self.num_vars = self.num_vars_real + self.num_vars_complex

        # create dictionaries to store the offset of each variable in the y vector
        # fmt:off
        # 1. var offset of real variables
        self.var_offset_real = { "w": 0 }
        # 2. var offset of complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": 2 * self.n_ph,
            "v": 2 * self.n_ph + 2 * self.n_ph,
            "i": 2 * self.n_ph + 2 * self.n_ph + 2 * self.n_ph,
        }
        # 3. var offset of all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + 2 * self.n_ph,
            "v": 1 + 2 * self.n_ph + 2 * self.n_ph,
            "i": 1 + 2 * self.n_ph + 2 * self.n_ph + 2 * self.n_ph,
        }
        # fmt:on
        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["i"] + 2 * self.n_ph

        # this value is calculated in the adapter file based on voltage regulator data and active reactive loads
        self.nt_effect = {
            ph: reg_ratio
            for (ph, reg_ratio) in self.obj.effective_reg_ratio.items()
            if reg_ratio is not None
        }
        print(f">> self.obj.effective_reg_ratio: {self.obj.effective_reg_ratio}")

    def get_basetype(self):
        return "volt_reg"

    def initial_guess(self, vals: dict) -> sps.coo_array:

        # y = [w, V, I, v, i]

        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        # initialise V of primary side. This should come from adapter too as
        # voltage regulator primary side voltage comes from the secondary of the transformer
        # after which it is connected

        # v_p
        idx_v_start = self.var_offset["v"]
        idx_v_end = self.var_offset["v"] + self.n_ph

        # V_dict = {}
        # phases_without_N = [ph for ph in self.get_phases() if ph != "N"]
        # for ph in phases_without_N:
        #     V_dict[ph] = self.obj.nominated_voltage[ph]

        V_phasors = utils.get_vector_phasors(self.obj.nominal_voltage)
        V_phasors = np.array(list(V_phasors.values()))
        y_0[idx_v_start:idx_v_end, 0] = V_phasors

        # v_s
        idx_vs_start = self.var_offset["v"] + self.n_ph
        idx_vs_end = self.var_offset["v"] + 2 * self.n_ph
        y_0[idx_vs_start:idx_vs_end, 0] = (
            np.array(list(self.nt_effect.values())) * V_phasors
        )

        # # I_s
        # idx__Ifrom_start = self.var_offset["I"] + self.n_ph
        # idx__Ifrom_end = self.var_offset["I"] + 2 * self.n_ph
        # y_0[idx__Ifrom_start:idx__Ifrom_end, 0] = self.obj.i_line_mag

        # Ip = np.array(
        #     [
        #         521.4219285 - 286.5874427j,
        #         -337.4391797 - 274.8730734j,
        #         -38.23376114 + 626.883616j,
        #     ]
        # )
        # idx__Ifrom_start = self.var_offset["I"]
        # idx__Ifrom_end = self.var_offset["I"] + self.n_ph
        # y_0[idx__Ifrom_start:idx__Ifrom_end, 0] = Ip

        # Is = Ip / np.array(list(self.nt_effect.values()))
        # idx__Ito_start = self.var_offset["I"] + self.n_ph
        # idx__Ito_end = self.var_offset["I"] + 2 * self.n_ph
        # y_0[idx__Ito_start:idx__Ito_end, 0] = Is

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
        ], f"to be implemented for var = {var}"

        # the NodeSide.To is only valid for "V" and "I"
        side_offset = 0
        if side == NodeSide.TO:
            side_offset = self.n_ph

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        if val_type == ValType.IMAG:
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

        I_ph = sps.identity(self.n_ph, dtype=float)
        AI = sps.bmat([[I_ph, None], [None, -I_ph]], format="coo")
        AV = sps.bmat([[I_ph, None], [None, I_ph]], format="coo")

        # Identity matrices for the equations
        # I -> [I1a, I1b, I1c, I2a, I2b, I2c]
        Id_I = sps.identity(2 * self.n_ph, format="coo")

        # v -> [vp_a, vp_b, vp_c, vs_a, vs_b, vs_c]
        Id_v = sps.identity(2 * self.n_ph, format="coo")

        nt = np.array(list(self.nt_effect.values()))
        print(f">> nt: {nt}")
        nt_mat = diags(nt, format="coo")
        nt_r = 1 / nt
        nt_r_mat = diags(nt_r, format="coo")

        print(f">> nt_mat: {nt_mat.toarray()}")

        # equations for voltage transformation
        N_v = sps.bmat([[-nt_mat.dot(I_ph), I_ph]], format="coo")

        # equations for current transformation
        N_i = sps.bmat([[-nt_r_mat.dot(I_ph), I_ph]], format="coo")

        # zero vectors for w for each equation
        Z_w1 = sps.lil_matrix((AI.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((AV.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((N_v.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((N_i.shape[0], 1), dtype=float)

        # fmt: off
        M = sps.bmat([
            #[w,    V,      I,      v,      i]
                [Z_w1, None, -Id_I,    None,   AI],   # 1) -I + Ai = 0
                [Z_w2, AV,  None,     -Id_v,  None],# 2) A'V - v = 0
                [Z_w3, None,  None,     N_v,   None], # 3)vs - nt*vp = 0
                [Z_w4, None,  None,     None,  N_i], # 4)is - nt_r*ip = 0
            ]) 

        # fmt: on

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

        # assemble the y vector
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:

        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.lil_matrix(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        )

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def calc_tap_and_update(self, vline: sps.coo_array, iline: sps.coo_array):
        print(f">> vline: {vline.toarray()}")
        print(f">> iline: {iline.toarray()}")
        input("continue?")

        # # i_line_mag = xfmr.kVA / (np.sqrt(3) * xfmr.sec_volt)
        # i_line_mag = 5000 / (np.sqrt(3) * 4.16)
        # i_line_ang = -np.arccos(pf)
        # i_line = i_line_mag * np.exp(1j * i_line_ang)

        ct_ratio = self.obj.ct_primary / self.obj.ct_secondary
        i_comp = -(iline / ct_ratio).toarray()

        # # TODO: sec_volt = 4.16 or 4.16/sqrt(3) ??
        # # v_ln = 1e3 * xfmr.sec_volt / np.sqrt(3)
        # v_ln = 1e3 * 4.16 / np.sqrt(3)
        # v_reg = v_ln / volt_reg_dict["PT_ratio"]
        v_reg = (vline / self.obj.pt_ratio).reshape(1, -1).toarray()

        # # resume-here : special case for each phase.
        r_setting = np.array(list(self.obj.r_setting.values()))
        x_setting = np.array(list(self.obj.x_setting.values()))
        R = r_setting / self.obj.ct_secondary
        X = x_setting / self.obj.ct_secondary
        Z = (R + 1j * X).reshape(-1, 1)
        print(f">> Z: {Z}")
        print(f">> i_comp: {i_comp}")
        v_drop = np.multiply(Z, i_comp)
        print(f">> v_drop: {v_drop}")
        input("continue?")

        v_reg = v_reg.reshape(1, -1).ravel()
        v_drop = v_drop.reshape(1, -1).ravel()
        print(f">> v_reg: {v_reg}")
        print(f">> v_drop: {v_drop}")
        input("continue?")

        for i, ph in enumerate(self.get_phases()):
            v_r = v_reg[i] - v_drop[i]
            print(f">> v_r: {v_r}")
            input("continue?")

            v_level = self.obj.voltage_level[ph] - (self.obj.bandwidth / 2)
            print(f">> v_level: {v_level}")
            print(f">> np.abs(v_r): {np.abs(v_r)}")
            tap_step = 0.75
            tap = int(round((v_level - np.abs(v_r)) / tap_step))
            print(f">> tap [ph: {ph}]: {tap}")
            self.obj.tap_setting[ph] = tap

        self.re_init()

    def re_init(self):
        # update self.obj.effective_reg_ratio
        # update self.nt_effect

        for ph in self.get_phases():
            tap = self.obj.tap_setting[ph]
            self.obj.effective_reg_ratio[ph] = utils.calc_effective_reg_ratio(
                tap, self.obj.reg_type
            )
            self.nt_effect[ph] = self.obj.effective_reg_ratio[ph]
