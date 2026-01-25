from oodesign import Source, GFMInverter3Ph, GFMInverter1Ph
import utils
from models.model import Model, ValType
from oodesign import Inverter
import numpy as np
import scipy.sparse as sps
from const import NodeSide, StudyType
from pprint import pformat
from utils import phasor_to_timedomain, get_start_end_idx


"""
1) class SourceModel represent any source connected to the node.
2) Each instance of these class could have different no. of phases based on the line object passed to it.
3) Source object passed to the init function here comes from the object oriented representation of the network.
4) class SourceModel has following functions:
    a)get_M_powerflow
    b)get_u_powerflow
    c)get_fy_powerflow        
5)  Presently, constant voltage source is implemented.
TODO: M, K, fy, u for dynamic simulation
TODO: Other source models to be implemented.
TODO: the model implemented here is for constant voltage source, so it should be a subclass of SourceModel.
"""


class SourceModel(Model):
    def __init__(self, source_obj: Source):
        self.obj = source_obj

        self.n_ph = self.obj.n_ph
        self.num_term = 1
        # self.at_node = self.obj.at_node
        self.nominal_voltage = self.obj.nominal_voltage

        # NOTE: each subclass should calculate and update its own number of eqns
        self.num_eqns = None

    def get_id(self):
        return self.obj.id

    def get_basetype(self):
        return "source"


class ConstantVoltageModel(SourceModel):
    def __init__(self, source_obj: Source):
        super().__init__(source_obj)

        self.num_eqns_real = 0
        self.num_eqns_complex = (
            self.n_ph  # first set: 1) -V + v = 0
            + self.n_ph  # second set: 2) -I + i = 0
            + self.n_ph  # third set: 3) equate v to the input voltage
        )
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex
        self.num_eqns_dynamic = self.num_eqns

        # book keeping for variables
        # fmt: off
        # y = [w, V, I, v, i]
        # self.vars_real = ["w"]
        # self.vars_complex = ["V", "I", "v", "i"]
        self.num_vars_real = (1)
        self.num_vars_complex = (
            self.n_ph # V
            +self.n_ph # I
            +self.n_ph # v
            +self.n_ph # i
        )
    
        self.num_vars = self.num_vars_real + self.num_vars_complex
        self.num_vars_dynamic = self.num_vars
        # fmt: on

        # calculate variable offsets
        # fmt:off
        # 1. offset for real variables
        self.var_offset_real = {
            "w": 0,
        }
        # 2. offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ph,
            "v": self.n_ph + self.n_ph,
            "i": self.n_ph + self.n_ph + self.n_ph,
                }
        # 3. offset for all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,            
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
        }
        # fmt:on

        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["i"] + self.n_ph

        self.var_offset_dynamic = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
        }

    def initial_guess(self, vals: dict) -> sps.coo_array:
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = v_phasors

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        return y_0
    
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

        return yp0_dyn
    
    def initial_guess_dynamic_zero(self, y_comp, w_nom):
        assert len(y_comp) == self.num_vars

        # pf vars: [w V I v i]
        # dyn vars: [w V I v i]
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        y0_dyn[0] = y_comp[0].real
        # V:
        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values()))
        dyn_V_idx_start, dyn_V_idx_end = get_start_end_idx(self.var_offset_dynamic, "V", self.n_ph)
        y0_dyn[dyn_V_idx_start: dyn_V_idx_end] = [np.sqrt(2) * phasor_to_timedomain(val) for val in v_phasors]

        return y0_dyn

    def initial_guess_dynamic(self, y_comp, w_nom):
        assert len(y_comp) == self.num_vars

        # pf vars: [w V I v i]
        # dyn vars: [w V I v i]
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        y0_dyn[0] = y_comp[0].real
        # rest:
        y0_dyn[1:] = [np.sqrt(2) * phasor_to_timedomain(val) for val in y_comp[1:]]

        return y0_dyn

    def get_local_idx_dynamic(
        self, var: str, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self, stage=None) -> sps.coo_array:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
           and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """

        I_ph = sps.identity(self.n_ph, format="coo")

        Z_w1 = sps.lil_matrix((I_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((I_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((I_ph.shape[0], 1), dtype=float)

        # fmt: off
        M = sps.bmat(
            [   #w,      V,     I,       v,      i]                
                [Z_w1,  -I_ph,  None,    I_ph,   None], # 1)-V + v =0
                [Z_w2,  None,   -I_ph,   None,  -I_ph], # 2)-I - i =0 
                [Z_w3,  None,   None,    -I_ph,  None], # 3)-v + [u] = 0 equate v to the input voltage
            ]
        )
        # fmt: on

        return M

    # return [comp_fy_re, comp_fy_im]
    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation.
        2. For constant voltage source, fy is zero.
        """

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

    # def get_pd_fy(self, fy: sps.coo_array, y: sps.coo_array) -> sps.coo_array:
    #     assert self.num_eqns == fy.shape[0]
    #     assert self.num_vars == y.shape[0]
    #     pd_fy = sps.lil_matrix((self.num_eqns, y.shape[0]), dtype=complex)
    #     return pd_fy

    def get_u_powerflow(self) -> sps.coo_array:
        """
        1. Source model u has constant voltage values
        """

        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=complex)

        V_phasors = utils.get_vector_phasors(self.nominal_voltage)

        # NOTE: presently the "nominal_voltage" take from input is in KV.
        # TODO: To be converted in volts, while taking data in adapter files.
        V = np.array(list(V_phasors.values()))  # to be changed once above TODO is done

        # y = [V, I, w, v, i]

        # update u for eqn 3
        # start and end index of eq3 in u vector
        # fmt: off
        idx_eq3_re_start = (           
            self.n_ph # 1) -V + v = 0
            + self.n_ph # 2) -I + i = 0
        )
        # fmt: on

        idx_eq3_re_end = idx_eq3_re_start + self.n_ph
        u[idx_eq3_re_start:idx_eq3_re_end, 0] = V.real

        idx_eq3_im_start = self.num_eqns + idx_eq3_re_start
        idx_eq3_im_end = idx_eq3_im_start + self.n_ph
        u[idx_eq3_im_start:idx_eq3_im_end, 0] = V.imag

        return u[: self.num_eqns], u[self.num_eqns :]

    def get_K_dynamic(self, stage=None):
        K = sps.lil_matrix((self.num_eqns, self.num_vars), dtype=float)
        assert K.shape[0] == self.num_eqns_dynamic
        assert K.shape[1] == self.num_vars_dynamic
        return K

    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        return self.get_M_powerflow_inner(stage)

    def get_fy_dynamic(self, t: float, y: np.ndarray, yp: np.ndarray, stage=None) -> np.ndarray:
        fy = np.zeros(self.num_eqns, dtype=float)
        # fy = sps.lil_matrix((self.num_eqns, 1), dtype=float)
        return fy

    def get_u_dynamic(self, t: float, y: np.ndarray) -> np.ndarray:
        u = np.zeros(self.num_eqns, dtype=float)

        w_idx = self.var_offset["w"]
        w = y[w_idx]

        v = []
        phases_without_n = [ph for ph in self.get_phases() if ph != "N"]
        for ph in phases_without_n:
            v_mag = np.sqrt(2) * self.nominal_voltage[ph]
            if ph == "A":
                angle = 0
            elif ph == "B":
                angle = -120
            elif ph == "C":
                angle = 120
            else:
                raise ValueError(f"unknown phase: {ph}")
            rad = np.radians(angle)
            v_val = v_mag * np.cos(w * t + rad)
            v.append(v_val)

        v = np.array(v)

        idx_eq3_start = self.n_ph + self.n_ph
        idx_eq3_end = idx_eq3_start + self.n_ph

        u[idx_eq3_start:idx_eq3_end] = v
        return u


class ConstantVoltageSeqModel(SourceModel):
    def __init__(self, source_obj: Source):
        super().__init__(source_obj)

        self.num_eqns_real = 0
        self.num_eqns_complex = (
            self.n_ph  # first set: 1) -V + v = 0
            + self.n_ph  # second set: 2) -I + i = 0
            + self.n_ph  # third set: 3) equate v to the input voltage
        )
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book keeping for variables
        # fmt: off
        # y = [w, V, I, v, i, vneg]
        # self.vars_real = ["w"]
        # self.vars_complex = ["V", "I", "v", "i"]
        self.num_vars_real = (1)
        self.num_vars_complex = (
            self.n_ph # V
            +self.n_ph # I
            +self.n_ph # v
            +self.n_ph # i
            + 1 # vneg
        )
    
        self.num_vars = self.num_vars_real + self.num_vars_complex
        # fmt: on

        # calculate variable offsets
        # fmt:off
        # 1. offset for real variables
        self.var_offset_real = {
            "w": 0,
        }
        # 2. offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ph,
            "v": self.n_ph + self.n_ph,
            "i": self.n_ph + self.n_ph + self.n_ph,
            "vneg": self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }
        # 3. offset for all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,            
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
            "vneg": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }
        # fmt:on

        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["vneg"] + 1

        # ineq
        self.num_ineq_real = 1
        self.num_ineq_complex = 0
        self.num_ineq_split = self.num_ineq_real + self.num_ineq_complex

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # [V, I, w, v, i, vneg]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = v_phasors

        idx_vneg = self.var_offset["vneg"]
        y_0[idx_vneg, 0] = y_0[idx_v_start, 0] * 0.1

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        print(f">> source: y_0 initialized to: {y_0.toarray()}")
        # input("continue1?")

        return y_0

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        if side:
            assert side == NodeSide.AT

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self) -> sps.coo_array:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
           and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """

        I_ph = sps.identity(self.n_ph, format="coo")

        Z_w1 = sps.lil_matrix((I_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((I_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((I_ph.shape[0], 1), dtype=float)

        Ineg = sps.lil_matrix((I_ph.shape[0], 1), dtype=float)
        Ineg[0] = 1
        Ineg[1] = -1 / 2
        Ineg[2] = -1 / 2

        # fmt: off
        M = sps.bmat(
            [   #w,      V,     I,       v,      i      vneg]                
                [Z_w1,  -I_ph,  None,    I_ph,   None,  None], # 1)-V + v =0
                [Z_w2,  None,   -I_ph,   None,   I_ph,  None], # 2)-I + i =0 
                [Z_w3,  None,   None,    -I_ph,  None,  Ineg], # 3)-v + [u] + vneg = 0 equate v to the input voltage
            ]
        )
        # fmt: on

        return M

    # return [comp_fy_re, comp_fy_im]
    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation.
        2. For constant voltage source, fy is zero.
        """
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        idx_vneg = self.var_offset["vneg"]

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        idx_eq3_start = self.n_ph + self.n_ph

        assert self.n_ph == 3
        fy[idx_eq3_start + 1] = 1j * np.sqrt(3) / 2 * y[idx_vneg]
        fy[idx_eq3_start + 2] = -1j * np.sqrt(3) / 2 * y[idx_vneg]

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

        eq3_start_row = self.n_ph + self.n_ph
        vneg_re_col_offset = self.var_offset["vneg"]
        vneg_im_col_offset = self.var_offset_complex["vneg"]

        # Vb
        pd_fy_split[eq3_start_row + 1, self.num_vars + vneg_im_col_offset] = (
            -np.sqrt(3) / 2
        )
        pd_fy_split[self.num_eqns + eq3_start_row + 1, vneg_re_col_offset] = (
            np.sqrt(3) / 2
        )

        # Vc
        pd_fy_split[eq3_start_row + 2, self.num_vars + vneg_im_col_offset] = (
            np.sqrt(3) / 2
        )
        pd_fy_split[self.num_eqns + eq3_start_row + 2, vneg_re_col_offset] = (
            -np.sqrt(3) / 2
        )

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_u_powerflow(self) -> sps.coo_array:
        """
        1. Source model u has constant voltage values
        """
        print(f">> in get_u_powerflow of source...")

        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=complex)

        V_phasors = utils.get_vector_phasors(self.nominal_voltage)

        # NOTE: presently the "nominal_voltage" take from input is in KV.
        # TODO: To be converted in volts, while taking data in adapter files.
        V = np.array(list(V_phasors.values()))  # to be changed once above TODO is done

        # y = [V, I, w, v, i]

        # update u for eqn 3
        # start and end index of eq3 in u vector
        # fmt: off
        idx_eq3_re_start = (           
            self.n_ph # 1) -V + v = 0
            + self.n_ph # 2) -I + i = 0
        )
        # fmt: on

        idx_eq3_re_end = idx_eq3_re_start + self.n_ph
        u[idx_eq3_re_start:idx_eq3_re_end, 0] = V.real

        idx_eq3_im_start = self.num_eqns + idx_eq3_re_start
        idx_eq3_im_end = idx_eq3_im_start + self.n_ph
        u[idx_eq3_im_start:idx_eq3_im_end, 0] = V.imag

        print(
            f">> returning u: {u[: self.num_eqns].toarray(), u[self.num_eqns :].toarray()}"
        )

        return u[: self.num_eqns], u[self.num_eqns :]

    def get_num_ineq_split(self):
        return self.num_ineq_split

    def get_num_ineq_real(self):
        return self.num_ineq_real

    def get_num_ineq_complex(self):
        return self.num_ineq_complex

    def get_hy_powerflow(
        self, y_real: sps.coo_array, y_imag: sps.coo_array, alpha=0.2
    ) -> tuple[sps.coo_array, sps.coo_array] | None:
        # v_neg_re**2 + v_neg_im**2 - alpha * u[0]_re**2 - alpha * u[0]_im**2 + s = 0       # s is separate

        nom_voltage_val = list(self.nominal_voltage.values())[0]
        u_re, u_im = nom_voltage_val, 0
        u0 = u_re + 1j * u_im
        print(f">> u0: {u0}")

        idx_vneg_re = self.var_offset["vneg"]
        idx_vneg_im = self.var_offset_complex["vneg"]
        vneg_re = y_real[idx_vneg_re, 0]
        vneg_im = y_imag[idx_vneg_im, 0]

        result = vneg_re**2 + vneg_im**2 - alpha * u0.real**2 - alpha * u0.imag**2
        hy_split = sps.lil_matrix((self.num_ineq_split, 1), dtype=float)
        assert self.num_ineq_split == 1
        print(f">> type(result): {type(result)}")
        assert type(result) == float or type(result) == np.float64
        hy_split[0, 0] = result

        print(f">> retuning hy_split: {hy_split.toarray()}")

        return hy_split[0 : self.num_ineq_real], hy_split[self.num_ineq_real :]

    def get_pd_hy_split(
        self, y_real, y_imag
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        print(f">> here!!!")
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_hy_split = sps.lil_matrix(
            (
                self.num_ineq_split,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        )

        idx_vneg_re = self.var_offset["vneg"]
        idx_vneg_im = self.var_offset_complex["vneg"]
        vneg_re = y_real[idx_vneg_re, 0]
        vneg_im = y_imag[idx_vneg_im, 0]
        pd_hy_split[0, idx_vneg_re] = 2 * vneg_re
        pd_hy_split[0, self.num_vars + idx_vneg_im] = 2 * vneg_im

        rr = pd_hy_split[0 : self.num_ineq_real, 0 : self.num_vars]
        ri = pd_hy_split[0 : self.num_ineq_real, self.num_vars :]
        ir = pd_hy_split[self.num_ineq_real :, 0 : self.num_vars]
        ii = pd_hy_split[self.num_ineq_real :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_pd_hy_split(
        self, y_real, y_imag, mu_real, mu_imag
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert (
            self.num_ineq_real == mu_real.shape[0]
        ), f"mu_real: {mu_real.shape[0]} != {self.num_ineq_real}"
        assert self.num_ineq_complex == mu_imag.shape[0]

        pd_pd_hy_split = sps.lil_matrix(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        )

        assert self.num_ineq_real == 1
        mu_re = mu_real[0, 0]
        vneg_re_row = self.var_offset["vneg"]
        vneg_re_col = vneg_re_row
        pd_pd_hy_split[vneg_re_row, vneg_re_col] = 2 * mu_re

        vneg_im_row = self.num_vars + self.var_offset_complex["vneg"]
        vneg_im_col = vneg_im_row
        pd_pd_hy_split[vneg_im_row, vneg_im_col] = 2 * mu_re

        rr = pd_pd_hy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_pd_hy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_pd_hy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_pd_hy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_objective_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        re = sps.lil_matrix((self.num_vars, 1), dtype=float)
        im = sps.lil_matrix((self.num_vars_complex, 1), dtype=float)

        return re, im

    def get_pd_pd_objective_split(
        self,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
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

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
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


"""
1)class InverterModel represent all GFL GFM inverters connected to the system
2)Presently, it only has one subclass: GFLInverterModel
"""

"""
TODO: Presently the OODesign has parent class Inverter which has subclasses 3-ph inverter and 1-ph inverter
2) These two subclasses further have two subclasses each which is based on whether they are GFL or GFM inverters
3)Now we want one parent class Inverter which has subclasses GFLInverterModel and GFMInverterModel
3)There need not be subclasses for phases as the eqns remain same but their count changes according to the number of phases
4)So that needs to be changed in OODesign and adapter_ieee_doc.py
5) The inverter has following blocks:
    a) the inverter bridge
    b) the LCL filter
    c) The phase locked loop
    d) the control block
    e) the SVPWM 
6) Out of these only the LCL filter forms part of the power flow and therefore the rest of the blocks can be written just as functions instead of making them 
    a part of powerflow.
"""

# class GFMInverter3PhDualLoopModel(GFMInverterModel):
#     def __init__(self, inverter_obj: GFMInverter3Ph):
#         super().__init__(inverter_obj)

#         # experiment flag
#         self.seq_mode = False
#         self.Eabc_derived = None

#         # TODO: add the check if it has dual loop control or single loop control here and then choose block A or block B
#         ###############################################################################################################
#         # values of inverter with sequence based control
#         # book keeping of the eqns
#         # fmt:off
#         self.num_eqns_real_seq = 0
#         self.num_eqns_complex_seq = (
#             self.n_ph   # 1) Vabc - Eabc - Eg + vabc =0
#             + self.n_ph # 1a) -vabc + r*iabc + jw*lamda = 0
#             + self.n_ph # 2) Lamda - L*iabc =0
#             + self.n_ph # 3) -Iabc + iabc = 0
#             + (self.n_ph - 1) # 4) Vln to Vll conversion
#             + self.n_ph # 5) -Eabc + u = 0
         
#         )
#         # fmt:on
#         self.num_eqns_seq = self.num_eqns_real_seq + self.num_eqns_complex_seq

#         # book keeping of variables
#         # fmt:off
#         # y = [w, Vabc, Eabc, Eg, Vabc_ll, Iabc, vabc, iabc, lamda]

#         self.num_vars_real_seq = 1 # w

#         self.num_vars_complex_seq = (
#             self.n_ph # Vabc
#             + self.n_ph # Eabc
#             # + 1 # Eg
#             + (self.n_ph - 1) # Vabc_ll
#             + self.n_ph # Iabc
#             + self.n_ph # vabc
#             + self.n_ph # iabc
#             + self.n_ph # lamda
#         )

#         self.num_vars_seq = self.num_vars_real_seq + self.num_vars_complex_seq
#         # fmt:on

#         # calculate variable offsets
#         # fmt:off
#         # 1. offset for real variables
#         self.var_offset_real_seq = {
#             "w" : 0,
#         }
#         # # 2. offset for complex variables
#         # self.var_offset_complex_seq = {
#         #     "V": 0,
#         #     "E" : self.n_ph,
#         #     "Eg": 2*self.n_ph,
#         #     "Vll" : 2*self.n_ph + 1,
#         #     "I": 3*self.n_ph - 1  + 1,
#         #     "v": 4*self.n_ph - 1  + 1,
#         #     "i": 5*self.n_ph - 1  + 1,
#         #     "lamda": 6*self.n_ph - 1 + 1
#         # }
#         # 2. offset for complex variables
#         self.var_offset_complex_seq = {
#             "V": 0,
#             "E" : self.n_ph,
#             "Vll" : 2*self.n_ph,
#             "I": 3*self.n_ph - 1,
#             "v": 4*self.n_ph - 1,
#             "i": 5*self.n_ph - 1,
#             "lamda": 6*self.n_ph - 1
#         }
#         # # 3. offset for all variables
#         # self.var_offset_seq = {
#         #     "w": 0,
#         #     "V": 1,
#         #     "E": 1 + self.n_ph,
#         #     "Eg": 1 + 2*self.n_ph + 1,
#         #     "Vll": 1 + 2*self.n_ph + 1,
#         #     "I": 1 + 3*self.n_ph - 1 + 1,
#         #     "v": 1 + 4*self.n_ph - 1 + 1,
#         #     "i": 1 + 5*self.n_ph - 1 + 1,
#         #     "lamda": 1 + 6*self.n_ph - 1 + 1
#         # }
#         # 3. offset for all variables
#         self.var_offset_seq = {
#             "w": 0,
#             "V": 1,
#             "E": 1 + self.n_ph,
#             "Vll": 1 + 2*self.n_ph,
#             "I": 1 + 3*self.n_ph - 1,
#             "v": 1 + 4*self.n_ph - 1,
#             "i": 1 + 5*self.n_ph - 1,
#             "lamda": 1 + 6*self.n_ph - 1
#         }
#         # fmt:on

#         assert len(self.var_offset_real_seq.keys()) + len(
#             self.var_offset_complex_seq.keys()
#         ) == len(self.var_offset_seq.keys())
#         assert self.num_vars_seq == self.var_offset_seq["lamda"] + self.n_ph

#         ###############################################################################################################

#         # default values for the inverter
#         # book keeping of the eqns
#         # fmt:off
#         self.num_eqns_real_def = 0
#         self.num_eqns_complex_def = (
#             self.n_ph   # 1) Vabc - Eabc + vabc =0
#             + self.n_ph # 1a) -vabc + r*iabc + jw*lamda = 0
#             + self.n_ph # 2) Lamda - L*iabc =0
#             + self.n_ph # 3) -Iabc + iabc = 0
#             + (self.n_ph - 1) # 4) Vln to Vll conversion
#             + self.n_ph # 5) -VLL + [u] = 0 this sets the voltage at the terminal to l-l nominal voltage
       
#         )
#         # fmt:on
#         self.num_eqns_def = self.num_eqns_real_def + self.num_eqns_complex_def

#         # book keeping of variables
#         # fmt:off
#         # y = [w, Vabc, Eabc, Vabc_ll, Iabc, vabc, iabc, lamda]

#         self.num_vars_real_def = 1 # w

#         self.num_vars_complex_def = (
#             self.n_ph # Vabc
#             + self.n_ph # Eabc
#             + (self.n_ph - 1) # Vabc_ll
#             + self.n_ph # Iabc
#             + self.n_ph # vabc
#             + self.n_ph # iabc
#             + self.n_ph # lamda
#         )

#         self.num_vars_def = self.num_vars_real_def + self.num_vars_complex_def
#         # fmt:on

#         # calculate variable offsets
#         # fmt:off
#         # 1. offset for real variables
#         self.var_offset_real_def = {
#             "w" : 0,
#         }
#         # 2. offset for complex variables
#         self.var_offset_complex_def = {
#             "V": 0,
#             "E" : self.n_ph,
#             "Vll" : 2*self.n_ph,
#             "I": 3*self.n_ph - 1,
#             "v": 4*self.n_ph - 1,
#             "i": 5*self.n_ph - 1,
#             "lamda": 6*self.n_ph - 1
#         }
#         # 3. offset for all variables
#         self.var_offset_def = {
#             "w": 0,
#             "V": 1,
#             "E": 1 + self.n_ph,
#             "Vll": 1 + 2*self.n_ph,
#             "I": 1 + 3*self.n_ph - 1,
#             "v": 1 + 4*self.n_ph - 1,
#             "i": 1 + 5*self.n_ph - 1,
#             "lamda": 1 + 6*self.n_ph - 1
#         }
#         # fmt:on

#         assert len(self.var_offset_real_def.keys()) + len(
#             self.var_offset_complex_def.keys()
#         ) == len(self.var_offset_def.keys())
#         assert self.num_vars_def == self.var_offset_def["lamda"] + self.n_ph

#         ###############################################################################################################

#         # by default, the inverter is in default mode
#         self.num_vars = self.num_vars_def
#         self.num_eqns = self.num_eqns_def
#         self.num_vars_real = self.num_vars_real_def
#         self.num_vars_complex = self.num_vars_complex_def
#         self.num_eqns_real = self.num_eqns_real_def
#         self.num_eqns_complex = self.num_eqns_complex_def
#         self.var_offset = self.var_offset_def
#         self.var_offset_real = self.var_offset_real_def
#         self.var_offset_complex = self.var_offset_complex_def

#     def initial_guess(self, vals: dict) -> sps.coo_array:
#         if self.seq_mode:
#             return self.initial_guess_seq(vals)
#         else:
#             return self.initial_guess_def(vals)

#     def initial_guess_def(self, vals: dict) -> sps.coo_array:
#         y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

#         v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
#         v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
#         idx_E_start = self.var_offset_def["E"]
#         idx_E_end = idx_E_start + self.n_ph
#         y_0[idx_E_start:idx_E_end, 0] = v_phasors

#         idx_w = self.var_offset_def["w"]
#         y_0[idx_w, 0] = vals["w"]

#         return y_0

#     def initial_guess_seq(self, vals: dict) -> sps.coo_array:
#         y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

#         v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
#         v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
#         idx_E_start = self.var_offset_seq["E"]
#         idx_E_end = idx_E_start + self.n_ph
#         y_0[idx_E_start:idx_E_end, 0] = v_phasors

#         idx_w = self.var_offset_seq["w"]
#         y_0[idx_w, 0] = vals["w"]

#         return y_0

#     def get_local_idx(
#         self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
#     ) -> int:
#         assert var in self.var_offset.keys()

#         # this function to be discussed as based on dual or single loop, this should fetch
#         # the index of the variable

#         if var == "w":
#             assert ph is None
#             assert side is None

#         side_offset = 0  # stc
#         phase_offset = 0 if ph is None else self.get_phases().index(ph)

#         if val_type == ValType.REAL:
#             return self.var_offset[var] + side_offset + phase_offset
#         elif val_type == ValType.IMAG:
#             return self.var_offset_complex[var] + side_offset + phase_offset

#     def get_M_powerflow_inner(self) -> sps.coo_matrix:
#         if self.seq_mode:
#             return self.get_M_powerflow_inner_seq()
#         else:
#             return self.get_M_powerflow_inner_def()

#     def get_M_powerflow_inner_def(self) -> sps.coo_matrix:
#         """
#         1)This function returns the M matrix for inverter model
#         2) For powerflow inverter is modeled as a voltage source with an output impedance
#         3) Based on whether GFM has dual control or single loop the output impedance changes as follows:
#         - For dual loop control the output impedance is just w0 * (L virtual impedance)
#         - For single loop control the output impedance is w0 * (L virtual impedance + L of LCL filter, C is negligible)
#         """

#         # create required identity, zero and coefficient matrices

#         I_ph = sps.identity(self.n_ph, format="coo")
#         Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)
#         # create R and L matrix

#         # fmt: off
#         R_mat = np.array([
#             [self.obj.raL2,    0,              0               ],
#             [0,                self.obj.rbL2,  0               ],
#             [0,                0,              self.obj.rcL2   ]
#         ])
#         L_mat = np.array([
#             [self.obj.La2,     0,              0               ],
#             [0,                self.obj.Lb2,   0               ],
#             [0,                0,              self.obj.Lc2    ]
#         ])
#         # fmt: on
#         # eq4_V = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])
#         eq4_V = np.array([[1, -1, 0], [0, 1, -1]])
#         I_eq4 = np.array([[1, 0], [0, 1]])

#         # create the M matrix
#         # fmt: off
#         M = sps.bmat([
#                #w,    Vabc,    Eabc,     Vabc_ll,    Iabc,    vabc,  iabc,   lamda
#               [Z_w,  I_ph,    -I_ph,    None,       None,    I_ph,  None,   None ], # 1) Vabc - Eabc + vabc =0
#               [Z_w,  None,    None,     None,       None,    -I_ph, R_mat,  None ], # 1a) -vabc + r*iabc + jw*Lamda =0   >> fy
#               [Z_w,  None,    None,     None,       None,    None,  -L_mat, I_ph ], # 2) Lamda - L*iabc =0
#               [Z_w,  None,    None,     None,       -I_ph,   None,  I_ph,   None ], # 3) -Iabc + iabc = 0
#               [None, eq4_V,   None,    -I_eq4,      None,    None,  None,   None ], # 4) Vln to Vll conversion
#               [Z_w,  I_ph,    None,     None,       None,    None,  None,   None ], # 5) -Vln + [u] = 0 this sets the voltage at the terminal to l-l nominal voltage
            
#              ]
#         )
#         # fmt: on
#         return M

#     def get_M_powerflow_inner_seq(self) -> sps.coo_matrix:
#         """
#         1)This function returns the M matrix for inverter model
#         2) For powerflow inverter is modeled as a voltage source with an output impedance
#         3) Based on whether GFM has dual control or single loop the output impedance changes as follows:
#         - For dual loop control the output impedance is just w0 * (L virtual impedance)
#         - For single loop control the output impedance is w0 * (L virtual impedance + L of LCL filter, C is negligible)
#         """

#         # create required identity, zero and coefficient matrices

#         I_ph = sps.identity(self.n_ph, format="coo")
#         Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)
#         # create R and L matrix

#         Eg_mat = np.ones((self.n_ph, 1))

#         # fmt: off
#         R_mat = np.array([
#             [self.obj.raL2,    0,              0               ],
#             [0,                self.obj.rbL2,  0               ],
#             [0,                0,              self.obj.rcL2   ]
#         ])
#         L_mat = np.array([
#             [self.obj.La2,     0,              0               ],
#             [0,                self.obj.Lb2,   0               ],
#             [0,                0,              self.obj.Lc2    ]
#         ])
#         # fmt: on
#         # eq4_V = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])
#         eq4_V = np.array([[1, -1, 0], [0, 1, -1]])
#         I_eq4 = np.array([[1, 0], [0, 1]])

#         # # create the M matrix
#         # # fmt: off
#         # M = sps.bmat(
#         #     [ #w,    Vabc,    Eabc,   Eg        Vabc_ll,    Iabc,    vabc,  iabc,   lamda]
#         #       [Z_w,  I_ph,    -I_ph,  Eg_mat,   None,       None,    I_ph,  None,   None ], # 1) Vabc - Eabc - Eg + vabc =0
#         #       [Z_w,  None,    None,   None,     None,       None,    -I_ph, R_mat,  None ], # 1a) -vabc + r*iabc + jw*Lamda =0   >> fy
#         #       [Z_w,  None,    None,   None,     None,       None,    None,  -L_mat, I_ph ], # 2) Lamda - L*iabc =0
#         #       [Z_w,  None,    None,   None,     None,       -I_ph,   None,  I_ph,   None ], # 3) -Iabc + iabc = 0
#         #       [None,  eq4_V,   None,  None,     -I_eq4,     None,    None,  None,   None ], # 4) Vln to Vll conversion
#         #       [Z_w,  None,    -I_ph,   None,    None,      None,    None,  None,   None ], # 5) -Eabc + [u] = 0 this sets the voltage at the terminal to l-l nominal voltage
#         #       [None, None,    None,   None,     None,       None,    None,  eq6_i,   None ],# 6) ia + ib + ic = 0
#         #      ]
#         # )
#         # # fmt: on

#         # create the M matrix
#         # fmt: off
#         M = sps.bmat(
#             [ #w,    Vabc,    Eabc,   Vabc_ll,    Iabc,    vabc,  iabc,   lamda]
#               [Z_w,  -I_ph,    I_ph,  None,       None,    -I_ph,  None,   None ], # 1) -Vabc + Eabc  - vabc =0
#               [Z_w,  None,    None,   None,       None,    -I_ph, R_mat,  None ], # 1a) -vabc + r*iabc + jw*Lamda =0   >> fy
#               [Z_w,  None,    None,   None,       None,    None,  -L_mat, I_ph ], # 2) Lamda - L*iabc =0
#               [Z_w,  None,    None,   None,       -I_ph,   None, I_ph,   None ], # 3) -Iabc + iabc = 0
#               [None,  eq4_V,   None,  -I_eq4,     None,    None,  None,   None ], # 4) Vln to Vll conversion
#               [Z_w,  -I_ph,    None,  None,      None,    None,  None,   None ], # 5) -Vabc + [u] = 0 
           
#              ]
#         )
#         # fmt: on
#         return M

#     def get_fy_powerflow(
#         self, y_re: sps.coo_array, y_im: sps.coo_array
#     ) -> tuple[sps.coo_array, sps.coo_array]:
#         """
#         1) This function returns the non-linear terms of every equation.
#         2)  For this model eqn 1 has a non-linear term jw*lamda
#         """
#         y = y_re.astype(complex)
#         y[self.num_vars_real :] += 1j * y_im

#         # create an empty matrix for fy
#         fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

#         # fy update for eqn 2
#         # start stop index for eqn 2
#         idx_eq2_start = self.n_ph
#         idx_eq2_end = idx_eq2_start + self.n_ph

#         # start and stop index for w and lamda in the y vector
#         idx_w = self.var_offset["w"]
#         idx_lamda_start = self.var_offset["lamda"]
#         idx_lamda_end = idx_lamda_start + self.n_ph

#         fy[idx_eq2_start:idx_eq2_end, 0] = (
#             1j * y[idx_w, 0] * y[idx_lamda_start:idx_lamda_end]
#         )

#         return fy.real, fy[self.num_eqns_real :].imag

#     def get_pd_fy_split(
#         self, y_real: sps.coo_array, y_imag: sps.coo_array
#     ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array]:
#         """
#         1) This function returns the partial derivative of fy with respect to y
#         """
#         # check that the variables received are of same shape as in the model
#         assert self.num_vars == y_real.shape[0]
#         assert self.num_vars_complex == y_imag.shape[0]

#         pd_fy_split = sps.coo_array(
#             (
#                 self.num_eqns + self.num_eqns_complex,
#                 self.num_vars + self.num_vars_complex,
#             ),
#             dtype=float,
#         ).tocsc()

#         # get the index of w
#         w_col_offset = self.var_offset["w"]
#         w = y_real[w_col_offset, 0]

#         # get the index of lamda
#         lamda_re_start_offset = self.var_offset["lamda"]
#         lamda_im_start_offset = self.var_offset_complex["lamda"]

#         # eq1_re i.e. update for real part of the pd of eqn 1
#         eq2_re_start_row = self.n_ph
#         for offset in range(self.n_ph):
#             row = eq2_re_start_row + offset

#             # get the index of lamda in the y vector
#             lamda_re_col_offset = lamda_re_start_offset + offset
#             lamda_im_col_offset = lamda_im_start_offset + offset
#             lamda_re = y_real[lamda_re_col_offset, 0]
#             lamda_im = y_imag[lamda_im_col_offset, 0]

#             # pd with respect to w
#             pd_fy_split[row, w_col_offset] = -lamda_im

#             # pd with respect to lamda_re : derivative  is zero

#             # pd with respect to lamda_im
#             pd_fy_split[row, lamda_im_col_offset] = w

#         # eq1_im i.e. update for imaginary part of the pd of eqn 1
#         eq2_im_start_row = self.num_eqns + eq2_re_start_row
#         for offset in range(self.n_ph):
#             row = eq2_im_start_row + offset

#             # get the index of lamda in the y vector
#             lamda_re_col_offset = lamda_re_start_offset + offset
#             lamda_im_col_offset = lamda_im_start_offset + offset
#             lamda_re = y_real[lamda_re_col_offset, 0]
#             lamda_im = y_imag[lamda_im_col_offset, 0]

#             # pd with respect to w
#             pd_fy_split[row, w_col_offset] = lamda_re

#             # pd with respect to lamda_re
#             pd_fy_split[row, lamda_re_col_offset] = w

#             # pd with respect to lamda_im : derivative is zero

#         rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
#         ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
#         ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
#         ii = pd_fy_split[self.num_eqns :, self.num_vars :]

#         return (rr, ri, ir, ii)

#     def get_u_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
#         if self.seq_mode:
#             return self.get_u_powerflow_seq()
#         else:
#             return self.get_u_powerflow_def()

#     def get_u_powerflow_def(self) -> tuple[sps.coo_array, sps.coo_array]:
#         """
#         1) This function returns the u vector for inverter model
#         2) The u vector is the input to the inverter model
#         3) The u vector is the voltage at the output of the inverter + o/p impedance i.e. nominal voltage
#         """
#         # create the u vector
#         u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=complex)

#         V_phasors = utils.get_vector_phasors(self.nominal_voltage)

#         # NOTE: presently the "nominal_voltage" take from input is in KV.
#         # TODO: To be converted in volts, while taking data in adapter files.
#         V = np.array(list(V_phasors.values()))  # to be changed once above TODO is done

#         # u vector update for eqn 5
#         # star and end index of eq5

#         # fmt:off
#         idx_eq6_re_start = (
#             self.n_ph # 1) Vabc - Eabc + vabc =0
#             + self.n_ph # 1a) -vabc + r*iabc + jw*lamda = 0
#             + self.n_ph # 2) Lamda - L*iabc =0
#             + self.n_ph # 3) -Iabc + iabc = 0
#             + (self.n_ph - 1) # 4) Vln to Vll conversion
#         )
#         # fmt:on

#         idx_eq6_re_end = idx_eq6_re_start + self.n_ph
#         u[idx_eq6_re_start:idx_eq6_re_end, 0] = V.real

#         idx_eq6_im_start = self.num_eqns + idx_eq6_re_start
#         idx_eq6_im_end = idx_eq6_im_start + self.n_ph
#         u[idx_eq6_im_start:idx_eq6_im_end, 0] = V.imag

#         return u[: self.num_eqns], u[self.num_eqns :]

#     def get_u_powerflow_seq(self) -> tuple[sps.coo_array, sps.coo_array]:
#         """
#         1) This function returns the u vector for inverter model
#         2) The u vector is the input to the inverter model
#         3) The u vector is the voltage at the output of the inverter + o/p impedance i.e. nominal voltage
#         """
#         # create the u vector
#         u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=complex)

#         # u vector update for eqn 5
#         # start and end index of eq5

#         # fmt:off
#         idx_eq6_re_start = (
#             self.n_ph # 1) Vabc - Eabc + vabc =0
#             + self.n_ph # 1a) -vabc + r*iabc + jw*lamda = 0
#             + self.n_ph # 2) Lamda - L*iabc =0
#             + self.n_ph # 3) -Iabc + iabc = 0
#             + (self.n_ph - 1) # 4) Vln to Vll conversion
#         )
#         # fmt:on

#         idx_eq6_re_end = idx_eq6_re_start + self.n_ph
#         u[idx_eq6_re_start:idx_eq6_re_end, 0] = self.Eabc_derived.real

#         idx_eq6_im_start = self.num_eqns + idx_eq6_re_start
#         idx_eq6_im_end = idx_eq6_im_start + self.n_ph
#         u[idx_eq6_im_start:idx_eq6_im_end, 0] = self.Eabc_derived.imag

#         return u[: self.num_eqns], u[self.num_eqns :]


# class GFMInverter1PhSingleLoopModel(GFMInverterModel):
#     def __init__(self, inverter_obj: GFMInverter1Ph):
#         super().__init__(inverter_obj)

#         # Bookeeping for single loop control

#         # book keeeping of eqns
#         self.num_eqns_real = 0
#         # fmt: off
#         self.num_eqns_complex = (
#             self.n_ph  # 1) Eabc - V_cabc - v_L1abc = 0
#             + self.n_ph # 2) -v_L1abc + jw*lamda_L1 + i_L1abc*r_L1 = 0
#             + self.n_ph # 3) lamda_L1 - L1*i_L1abc = 0
#             + self.n_ph # 4) -I1abc + i_L1abc = 0
#             + self.n_ph # 5) V_cabc - v_cabc = 0
#             + self.n_ph # 6) C*v_cabc - q_abc = 0 
#             + self.n_ph # 7) -i_cabc + jw*q_abc = 0
#             + self.n_ph # 8) V_cabc - V_abc - v_L2abc = 0
#             + self.n_ph # 9) -v_L2abc + jw*lamda_L2 + i_L2abc*r_L2 = 0
#             + self.n_ph # 10) lamda_L2 - L2*i_L2abc = 0
#             + self.n_ph # 11) -I2abc - i_L1abc + i_cabc + i_L2abc = 0
#             + self.n_ph # 12) - I3abc - i_L2abc = 0
#             + self.n_ph # 13) Va - Vb - V_ab = 0
#             + self.n_ph # 14) Vb - Vc - V_bc = 0
#             + self.n_ph # 15) Vc - Va  - Vca = 0
#             + self.n_ph # 16) - V_abc_ll + [u] = 0 this sets the voltage at the terminal to l-l nominal voltage

#         )
#         # fmt: on

#         # book keeping of variables
#         # y = [w, Vabc, V_cabc, E_abc, V_abc_ll, I_abc, v_abc, i_abc, lamda_L1, lamda_L2, q_abc]
#         # v_abc = [v_L1abc, v_cabc, v_L2abc]
#         # i_abc = [i_L1abc, i_cabc, i_l2abc]

#         self.num_vars_real = 1  # w
#         self.num_vars_complex = (
#             self.n_ph  # Vabc
#             + self.n_ph  # V_cabc
#             + self.n_ph  # E_abc
#             + self.n_ph  # V_abc_ll
#             + 3 * self.n_ph  # I _abc [I1abc, I2abc, I3abc]
#             + 3 * self.n_ph  # v_abc [v_L1abc, v_cabc, v_L2abc]
#             + 3 * self.n_ph  # i_abc [i_L1abc, i_cabc, i_L2abc]
#             + 2 * self.n_ph  # lamda [lamda_l1, lamda_L2]
#             + self.n_ph  # q_abc
#         )
#         # fmt: on

#         self.num_vars = self.num_vars_real + self.num_vars_complex

#         # calculate variable offsets
#         # fmt: off
#         # 1. offset for real variables
#         self.var_offset_real = {
#             "w": 0,
#         }
#         # 2. offset for complex variables
#         self.var_offset_complex = {
#             "Vabc": 0,
#             "V_cabc": self.n_ph,
#             "E_abc": 2*self.n_ph,
#             "V_abc_ll": 3*self.n_ph,
#             "I_abc": 4*self.n_ph, # I _abc [I1abc, I2abc, I3abc]
#             "v_abc": 7*self.n_ph, # v_abc [v_L1abc, v_cabc, v_L2abc]
#             "i_abc": 10*self.n_ph, # i_abc [i_L1abc, i_cabc, i_L2abc]
#             "lamda": 13*self.n_ph, # lamda [lamda_l1, lamda_L2]
#             "q_abc": 15*self.n_ph # q_abc
#         }
#         # 3. offset for all variables
#         self.var_offset = {
#             "w": 0,
#             "Vabc": 1,
#             "V_cabc": 1 + self.n_ph,
#             "E_abc": 1 + 2*self.n_ph,
#             "V_abc_ll": 1 + 3*self.n_ph,
#             "I_abc": 1 + 4*self.n_ph, # I _abc [I1abc, I2abc, I3abc]
#             "v_abc": 1 + 7*self.n_ph, # v_abc [v_L1abc, v_cabc, v_L2abc]
#             "i_abc": 1 + 10*self.n_ph, # i_abc [i_L1abc, i_cabc, i_L2abc]
#             "lamda": 1 + 13*self.n_ph, # lamda [lamda_l1, lamda_L2]
#             "q_abc": 1 + 15*self.n_ph # q_abc
#         }
#         # fmt: on

#         assert len(self.var_offset_real.keys()) + len(
#             self.var_offset_complex.keys()
#         ) == len(self.var_offset.keys())
#         assert self.num_vars == self.var_offset["q_abc"] + self.n_ph

#         ################### _init function ends here #############################

#     def initial_guess(self, vals: dict) -> sps.coo_array:
#         pass

#     def get_local_idx(
#         self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
#     ) -> int:
#         assert var in self.var_offset.keys()

#         if var == "w":
#             assert ph is None
#             assert side is None

#         side_offset = 0  # stc
#         phase_offset = 0 if ph is None else self.get_phases().index(ph)

#         if val_type == ValType.REAL:
#             return self.var_offset_real[var] + side_offset + phase_offset
#         elif val_type == ValType.IMAG:
#             return self.var_offset_complex[var] + side_offset + phase_offset

#     def get_M_powerflow_inner(self) -> sps.coo_matrix:
#         """
#         1)This function returns the M matrix for inverter model
#         2) For powerflow inverter is modeled as a voltage source with an output impedance
#         3) Based on whether GFM has dual control or single loop the output impedance changes as follows:
#         - For dual loop control the output impedance is just w0 * (L virtual impedance)
#         - For single loop control the output impedance is w0 * (L virtual impedance + L of LCL filter, C is negligible)
#         """

#         # create required identity, zero and coefficient matrices

#         I_ph = sps.identity(self.n_ph, format="coo")
#         Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=complex)
#         Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)
#         # create R and L matrix
#         R_L1 = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
#         L1_mat = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
#         R_L2 = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
#         L2_mat = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
#         C_mat = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)

#         v_L1abc = sps.bmat([[I_ph, None, None]])
#         v_cabc = sps.bmat([[None, I_ph, None]])
#         v_L2abc = sps.bmat([[None, None, I_ph]])
#         i_L1abc = sps.bmat([[I_ph, None, None]])
#         i_cabc = sps.bmat([[None, I_ph, None]])
#         i_L2abc = sps.bmat([[None, None, I_ph]])
#         I1_abc = sps.bmat([[I_ph, None, None]])
#         I2_abc = sps.bmat([[None, I_ph, None]])
#         I3_abc = sps.bmat([[None, None, I_ph]])
#         i_abc = sps.bmat([[I_ph, I_ph, I_ph]])

#         # create the M matrix
#         # fmt: off
#         M = sps.bmat(
#             [ #w,    Vabc,    V_cabc,   E_abc,   V_abc_ll,    I_abc,    v_abc,    i_abc,     lamda_L1,  lamda_L2,  q_abc]
#             [Z_w,    -I_ph,   None,     None,     None,       None,     v_L1abc,  None,      None,      None,      None ], # 1) Eabc - V_cabc - v_L1abc = 0 
#             [Z_w,    None,    None,     None,     None,       None,     -v_L1abc, R_L1,      None,      None,      None ], # 2) -v_L1abc + jw*lamda_L1 + i_L1abc*r_L1 = 0
#             [Z_w,    None,    None,     None,     None,       None,     None,     L1_mat,    I_ph,      None,      None ], # 3) lamda_L1 - L1*i_L1abc = 0
#             [Z_w,    None,    None,     None,     None,       -I1_abc,  None,     i_L1abc,   None,      None,      None ], # 4) -I1abc + i_L1abc = 0
#             [Z_w,    None,    I_ph,     None,     None,       None,     -v_cabc,  None,      None,      None,      None ], # 5) V_cabc - v_cabc = 0
#             [Z_w,    None,    None,     None,     None,       None,     C_mat,    None,      None,      None,      I_ph ], # 6) C*v_cabc - q_abc = 0
#             [Z_w,    None,    None,     None,     None,       None,     None,     -i_cabc,   None,      None,      None ], # 7) -i_cabc + jw*q_abc = 0
#             [Z_w,    -I_ph,   I_ph,     None,     None,       None,     v_L2abc,   None,     None,      None,      None ], # 8) V_cabc - V_abc - v_L2abc = 0
#             [Z_w,    None,    None,     None,     None,       None,    -v_L2abc,   R_L2,     None,      None,      None ], # 9) -v_L2abc + jw*lamda_L2 + i_L2abc*r_L2 = 0
#             [Z_w,    None,    None,      None,     None,       None,     None,      -L2_mat,  I_ph,      None,      None ], # 10) lamda_L2 - L2*i_L2abc = 0
#             [Z_w,    None,    None,     None,     None,      -I2_abc,   None,      i_abc,   None,      None,      None ], # 11) -I2abc - i_L1abc + i_cabc + i_L2abc = 0
#             [Z_w,    None,    None,     None,     None,      -I3_abc,   None,      None,     None,      None,      None ], # 12) - I3abc - i_L2abc = 0
#             []# resume here]

#         ])
#         # fmt: on
