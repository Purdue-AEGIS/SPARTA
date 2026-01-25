from models.system_model import SystemModel
import numpy as np
import scipy.sparse as sps
from collections import defaultdict
from models.model import Model
from models.model import ValType
from const import NodeSide
from models.component_models.node_model import *
from models.component_models.source_model import *

# from models.component_models.load_model import *
from models.component_models.load_model import *
from models.component_models.equipment_model.transformer_model import *
from models.component_models.line_model import *

# from models.component_models.line_model_neg import *
from models.component_models.equipment_model.capacitor_model import *

# GFMs:
# from models.component_models.equipment_model.volt_regulator_model import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV_refstudy13 import *
from models.component_models.inverters.GFMinverter_model_r0 import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_wip import *

# GFLs:
from models.component_models.inverters.GFLinverter_model_refactor import *

# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_unifi import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_unifi_init import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_vir_imp import *

from models.component_models.inverters.inverter_model import *

# import pickle
import dill as pickle

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import ipdb

# PICKLE_FILENAME = "multi_gfm_study.pkl"
PICKLE_FILENAME = "multi_gfm_study2.pkl"
# PICKLE_FILENAME = "multi_gfm_study_new.pkl"

plt.style.use("grayscale")

with open(PICKLE_FILENAME, "rb") as pkl:
    model = pickle.load(pkl)
    var_offset = pickle.load(pkl)
    t = pickle.load(pkl)
    y = pickle.load(pkl)

# wg = y[:, -1]
# plt.figure()
# plt.title("wg (Assimulo)")
# plt.plot(t, wg)
# plt.xlabel("Time (s)")
# plt.ylabel("freq")
# plt.legend("wg")
# plt.grid()
# # plt.show()

# # capactior - V
sources = model.sources

inverters = []
for source in sources:
    if isinstance(source, InverterModel):
        inverters.append(source)

assert inverters

# idx_inv = var_offset[inverter.get_id()]


# def plot_all_v():
#     inv0 = inverters[0]
#     inv1 = inverters[1]
#     inv2 = inverters[2]
#     inv3 = inverters[3]
#     inv4 = inverters[4]
#
#     idx_inv0 = var_offset[inv0.get_id()]
#     idx_inv1 = var_offset[inv1.get_id()]
#     idx_inv2 = var_offset[inv2.get_id()]
#     idx_inv3 = var_offset[inv3.get_id()]
#     idx_inv4 = var_offset[inv4.get_id()]
#
#     plt.rcParams.update({"font.size": 14})
#     plt.rcParams.update({"axes.labelsize": 14})
#     plt.rcParams.update({"axes.titlesize": 14})
#     plt.rcParams.update({"xtick.labelsize": 14})
#     plt.rcParams.update({"ytick.labelsize": 14})
#     plt.rcParams.update({"legend.fontsize": 14})
#
#     # fmt:off
#     heights = [
#         1.5, # inv0
#         1.5, # inv1
#         1.5, # inv2
#         1.5, # inv3
#         1.5, # inv4
#         1, # wg
#     ]
#     # fmt: on
#
#     fig = plt.figure(figsize=(13, 35))
#     gs = gridspec.GridSpec(
#         6,
#         1,
#         height_ratios=heights,
#         hspace=2,
#         wspace=0.8,
#     )
#     ax_inv0 = fig.add_subplot(gs[0, 0])
#     ax_inv1 = fig.add_subplot(gs[1, 0])
#     ax_inv2 = fig.add_subplot(gs[2, 0])
#     ax_inv3 = fig.add_subplot(gs[3, 0])
#     ax_inv4 = fig.add_subplot(gs[4, 0])
#     ax_wg = fig.add_subplot(gs[5, 0])
#
#     # inv0
#     idx_inv0_V_start = idx_inv0 + inv0.get_local_idx_dynamic("V", ph=None, side=None)
#     idx_inv0_V_end = idx_inv0_V_start + 3
#     inv0_V = y[:, idx_inv0_V_start:idx_inv0_V_end]
#     ax_inv0.set_title("(a)")
#     ax_inv0.plot(t, inv0_V)
#     ax_inv0.set_xlabel("Time (s)")
#     ax_inv0.set_ylabel("Inverter 1 Voltage")
#     ax_inv0.legend([f"V_{ph}" for ph in inv0.get_phases() if ph != "N"])
#
#     # inv1
#     idx_inv1_V_start = idx_inv1 + inv1.get_local_idx_dynamic("V", ph=None, side=None)
#     idx_inv1_V_end = idx_inv1_V_start + 3
#     inv1_V = y[:, idx_inv1_V_start:idx_inv1_V_end]
#     ax_inv1.set_title("(b)")
#     ax_inv1.plot(t, inv1_V)
#     ax_inv1.set_xlabel("Time (s)")
#     ax_inv1.set_ylabel("Inverter 1 Voltage")
#     ax_inv1.legend([f"V_{ph}" for ph in inv1.get_phases() if ph != "N"])
#
#     # inv2
#     idx_inv2_V_start = idx_inv2 + inv2.get_local_idx_dynamic("V", ph=None, side=None)
#     idx_inv2_V_end = idx_inv2_V_start + 3
#     inv2_V = y[:, idx_inv2_V_start:idx_inv2_V_end]
#     ax_inv2.set_title("(c)")
#     ax_inv2.plot(t, inv2_V)
#     ax_inv2.set_xlabel("Time (s)")
#     ax_inv2.set_ylabel("Inverter 1 Voltage")
#     ax_inv2.legend([f"V_{ph}" for ph in inv2.get_phases() if ph != "N"])
#
#     # inv3
#     idx_inv3_V_start = idx_inv3 + inv3.get_local_idx_dynamic("V", ph=None, side=None)
#     idx_inv3_V_end = idx_inv3_V_start + 3
#     inv3_V = y[:, idx_inv3_V_start:idx_inv3_V_end]
#     ax_inv3.set_title("(d)")
#     ax_inv3.plot(t, inv3_V)
#     ax_inv3.set_xlabel("Time (s)")
#     ax_inv3.set_ylabel("Inverter 1 Voltage")
#     ax_inv3.legend([f"V_{ph}" for ph in inv3.get_phases() if ph != "N"])
#
#     # inv4
#     idx_inv4_V_start = idx_inv4 + inv4.get_local_idx_dynamic("V", ph=None, side=None)
#     idx_inv4_V_end = idx_inv4_V_start + 3
#     inv4_V = y[:, idx_inv4_V_start:idx_inv4_V_end]
#     ax_inv4.set_title("(e)")
#     ax_inv4.plot(t, inv4_V)
#     ax_inv4.set_xlabel("Time (s)")
#     ax_inv4.set_ylabel("Inverter 1 Voltage")
#     ax_inv4.legend([f"V_{ph}" for ph in inv4.get_phases() if ph != "N"])
#
#     # wg
#     wg = y[:, -1]
#     ax_wg.set_title("(f)")
#     ax_wg.plot(t, wg)
#     ax_wg.set_xlabel("Time (s)")
#     ax_wg.set_ylabel("Network Frequency")
#     ax_wg.legend([r"$\omega_g$"])
#     ax_wg.set_ylim(376.985, 377.005)  # Set y-axis limits
#
#     # plt.savefig("multi_gfm_study.pdf", bbox_inches="tight")
#     plt.savefig("multi_gfm_study2.pdf", bbox_inches="tight")
#     # plt.savefig('gfm_study.pdf')
#     plt.show()


def plot_all():
    inv0 = inverters[0]
    inv1 = inverters[1]
    inv2 = inverters[2]
    inv3 = inverters[3]
    inv4 = inverters[4]
    inv5 = inverters[5]

    idx_inv0 = var_offset[inv0.get_id()]
    idx_inv1 = var_offset[inv1.get_id()]
    idx_inv2 = var_offset[inv2.get_id()]
    idx_inv3 = var_offset[inv3.get_id()]
    idx_inv4 = var_offset[inv4.get_id()]
    idx_inv5 = var_offset[inv5.get_id()]

    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"axes.labelsize": 14})
    plt.rcParams.update({"axes.titlesize": 14})
    plt.rcParams.update({"xtick.labelsize": 14})
    plt.rcParams.update({"ytick.labelsize": 14})
    plt.rcParams.update({"legend.fontsize": 14})

    # fmt:off
    heights = [
        1.5, # inv0
        1.5, # inv1
        1.5, # inv2
        1.5, # inv3
        1.5, # inv4
        1.5, # inv5
        # 1, # wg
    ]
    # fmt: on

    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(
        6,
        1,
        height_ratios=heights,
        hspace=0.5,
        wspace=0.6,
    )
    ax_inv0 = fig.add_subplot(gs[0, 0])
    ax_inv1 = fig.add_subplot(gs[1, 0])
    ax_inv2 = fig.add_subplot(gs[2, 0])
    ax_inv3 = fig.add_subplot(gs[3, 0])
    ax_inv4 = fig.add_subplot(gs[4, 0])
    ax_inv5 = fig.add_subplot(gs[5, 0])
    # ax_wg = fig.add_subplot(gs[6, 0])

    # inv0
    idx_inv0_I_start = idx_inv0 + inv0.get_local_idx_dynamic("I", ph=None, side=None)
    idx_inv0_I_end = idx_inv0_I_start + 3
    inv0_I = y[:, idx_inv0_I_start:idx_inv0_I_end]
    ax_inv0.set_title("(a)")
    ax_inv0.plot(t, inv0_I)
    ax_inv0.set_xlabel("Time (s)")
    ax_inv0.set_ylabel("Current (A)")
    ax_inv0.legend([f"I_{ph}" for ph in inv0.get_phases() if ph != "N"], framealpha=1)

    # inv1
    idx_inv1_I_start = idx_inv1 + inv1.get_local_idx_dynamic("I", ph=None, side=None)
    idx_inv1_I_end = idx_inv1_I_start + 3
    inv1_I = y[:, idx_inv1_I_start:idx_inv1_I_end]
    ax_inv1.set_title("(b)")
    ax_inv1.plot(t, inv1_I)
    ax_inv1.set_xlabel("Time (s)")
    ax_inv1.set_ylabel("Current (A)")
    ax_inv1.legend([f"I_{ph}" for ph in inv1.get_phases() if ph != "N"], framealpha=1)

    # inv2
    idx_inv2_I_start = idx_inv2 + inv2.get_local_idx_dynamic("I", ph=None, side=None)
    idx_inv2_I_end = idx_inv2_I_start + 3
    inv2_I = y[:, idx_inv2_I_start:idx_inv2_I_end]
    ax_inv2.set_title("(c)")
    ax_inv2.plot(t, inv2_I)
    ax_inv2.set_xlabel("Time (s)")
    ax_inv2.set_ylabel("Current (A)")
    ax_inv2.legend([f"I_{ph}" for ph in inv2.get_phases() if ph != "N"], framealpha=1)

    # inv3
    idx_inv3_I_start = idx_inv3 + inv3.get_local_idx_dynamic("I", ph=None, side=None)
    idx_inv3_I_end = idx_inv3_I_start + 3
    inv3_I = y[:, idx_inv3_I_start:idx_inv3_I_end]
    ax_inv3.set_title("(d)")
    ax_inv3.plot(t, inv3_I)
    ax_inv3.set_xlabel("Time (s)")
    ax_inv3.set_ylabel("Current (A)")
    ax_inv3.legend([f"I_{ph}" for ph in inv3.get_phases() if ph != "N"], framealpha=1)

    # inv4
    idx_inv4_I_start = idx_inv4 + inv4.get_local_idx_dynamic("I", ph=None, side=None)
    idx_inv4_I_end = idx_inv4_I_start + 3
    inv4_I = y[:, idx_inv4_I_start:idx_inv4_I_end]
    ax_inv4.set_title("(e)")
    ax_inv4.plot(t, inv4_I)
    ax_inv4.set_xlabel("Time (s)")
    ax_inv4.set_ylabel("Current (A)")
    ax_inv4.legend([f"I_{ph}" for ph in inv4.get_phases() if ph != "N"], framealpha=1)

    # inv5
    idx_inv5_I_start = idx_inv5 + inv5.get_local_idx_dynamic("I", ph=None, side=None)
    idx_inv5_I_end = idx_inv5_I_start + 3
    inv5_I = y[:, idx_inv5_I_start:idx_inv5_I_end]
    ax_inv5.set_title("(f)")
    ax_inv5.plot(t, inv5_I)
    ax_inv5.set_xlabel("Time (s)")
    ax_inv5.set_ylabel("Current (A)")
    ax_inv5.legend([f"I_{ph}" for ph in inv5.get_phases() if ph != "N"], framealpha=1)

    # # wg
    # wg = y[:, -1]
    # wg_freq = wg / (2 * np.pi)
    # ax_wg.set_title("(g)")
    # ax_wg.plot(t, wg_freq)
    # ax_wg.set_xlabel("Time (s)")
    # ax_wg.set_ylabel("Frequency")
    # ax_wg.legend([r"$f_g$"], framealpha=1)
    # # ax_wg.set_ylim(376.987, 376.999)  # Set y-axis limits
    # ax_wg.set_ylim(59.9995, 60.001)

    # plt.savefig("multi_gfm_study.pdf", bbox_inches="tight")
    # plt.savefig("multi_gfm_study2.pdf", bbox_inches="tight")
    plt.savefig("multi_gfm_study_v2.pdf", bbox_inches="tight")
    # plt.savefig('gfm_study.pdf')
    plt.show()


def plot_vrefamp():
    plt.figure()
    idx_vrefamp_start = idx_inv + inverter.get_local_idx_dynamic(
        "Vref_amp", ph=None, side=None
    )
    idx_vrefamp_end = idx_vrefamp_start + 1
    vrefamp = y[:, idx_vrefamp_start:idx_vrefamp_end] * inverter.obj.V_base
    plt.title("Vref_amp (Assimulo)")
    plt.plot(t, vrefamp)
    plt.xlabel("Time (s)")
    plt.ylabel("Vref_amp")
    # plt.ylim(570, 600)
    plt.legend(["Vref_amp"])
    plt.show()


def plot_pe_qe():
    # Pe
    idx_filter_Pe_start = idx_inv + inverter.get_local_idx_dynamic(
        "Pe", ph=None, side=None
    )
    idx_filter_Pe_end = idx_filter_Pe_start + 1
    filter_Pe = y[:, idx_filter_Pe_start:idx_filter_Pe_end]
    plt.figure()
    plt.title("Pe (Assimulo)")
    plt.plot(t, filter_Pe)
    plt.xlabel("Time (s)")
    plt.ylabel("Reactive Power")
    plt.legend(["Pe"])

    # Qe
    idx_filter_qe_start = idx_inv + inverter.get_local_idx_dynamic(
        "Qe", ph=None, side=None
    )
    idx_filter_qe_end = idx_filter_qe_start + 1
    filter_qe = y[:, idx_filter_qe_start:idx_filter_qe_end]
    plt.figure()
    plt.title("Qe (Assimulo)")
    plt.plot(t, filter_qe)
    plt.xlabel("Time (s)")
    plt.ylabel("Reactive Power")
    plt.legend(["Qe"])

    plt.show()


# plot_vrefamp()
# plot_pe_qe()
plot_all()
# plot_all_v()
