# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2017, Battelle Memorial Institute
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of the FreeBSD
# Project.
#
# This material was prepared as an account of work sponsored by an
# agency of the United States Government.  Neither the United States
# Government nor the United States Department of Energy, nor Battelle,
# nor any of their employees, nor any jurisdiction or organization that
# has cooperated in the development of these materials, makes any
# warranty, express or implied, or assumes any legal liability or
# responsibility for the accuracy, completeness, or usefulness or any
# information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights.
#
# Reference herein to any specific commercial product, process, or
# service by trade name, trademark, manufacturer, or otherwise does not
# necessarily constitute or imply its endorsement, recommendation, or
# favoring by the United States Government or any agency thereof, or
# Battelle Memorial Institute. The views and opinions of authors
# expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#
# PACIFIC NORTHWEST NATIONAL LABORATORY
# operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY
# under Contract DE-AC05-76RL01830
# }}}
import logging
from econ_dispatch.component_models import ComponentBase

_log = logging.getLogger(__name__)

DEFAULT_MFR_CHW = 50.4 #kg/s
DEFAULT_T_CHW = 7.0 #C
DEFAULT_MFR_ABW = 10.0 #kg/s
DEFAULT_T_ABW = 10.0 #C
DEFAULT_MFR_CHWBLDG = 30.0 #kg/s
DEFAULT_T_CHWBLDGRETURN = 14.0 #C
DEFAULT_T_CHWCHILLERRETURN = 15.0 #C
DEFAULT_T_CHWSTORAGESUPPLY = 7.0 #C
SPECIFIC_HEAT_WATER = 4.186 #kJ/kG-C
DENSITY_WATER = 1.000 #kg/L


class Component(ComponentBase):
    def __init__(self,
                 tank_volume=None,
                 input_temp=None,
                 output_temp=None,
                 fluid_type="water",
                 **kwargs):
        super(Component, self).__init__(**kwargs)

        tank_volume = float(tank_volume)
        input_temp = float(tank_volume)
        output_temp = float(tank_volume)

        if fluid_type not in ("water", "glycolwater30", "glycolwater50"):
            _log.warning("Unrecognized fluid type {} (must be water, glycolwater30, or glycolwater50)."
                         " Defaulting to water.".format(fluid_type))

        cp_fluid = 4.186

        if self.fluid_type == 'glycolwater30':
            cp_fluid = 3.913
        elif self.fluid_type == 'glycolwater50':
            cp_fluid = 3.558

        heat_capacity = tank_volume * cp_fluid * abs(input_temp - output_temp) / 3600.0 #convert to kWh

        self.parameters["heat_capacity"] = heat_capacity
        self.parameters["eff"] = 1.0


    def get_output_metadata(self):
        return []

    def get_input_metadata(self):
        return []

    def get_commands(self, component_loads):
        #PLACEHOLDER VARIABLE FOR DISPATCHED LOAD FROM TANK TO BUILDING
        Q_tank_bldg=1.0  # mmBTU/hr
        #PLACEHOLDER VARIABLE FOR DISPATCHED LOAD FROM CHILLERS TO TANK
        Q_chiller_tank=0   # mmBTU/hr
        #PLACEHOLDER VARIABLE FOR DISPATCHED LOAD FROM ABSORPTION CHILLERS TO TANK
        Q_abchiller_tank=0   # mmBTU/hr

        Q_tank_bldg_kW = Q_tank_bldg *1000/3.412
        #Q_chiller_tank_kW = Q_chiller_tank * 1000 / 3.412
        #Q_abchiller_tank_kW = Q_abchiller_tank * 1000 / 3.412
        mass_flow_rate_tank_bldg =  Q_tank_bldg_kW / (SPECIFIC_HEAT_WATER*(self.T_chwBldgReturn-self.T_chwStorageSupply))
        vol_flow_rate_setpoint_tank_bldg = mass_flow_rate_tank_bldg / DENSITY_WATER
        # mass_flow_rate_chiller_tank = Q_chiller_tank_kW / (SPECIFIC_HEAT_WATER * (self.T_chwChillerReturn - self.T_chW))
        # vol_flow_rate_setpoint_chiller_tank = mass_flow_rate_chiller_tank / DENSITY_WATER
        # mass_flow_rate_abchiller_tank = Q_abchiller_tank_kW / (SPECIFIC_HEAT_WATER * (self.T_chwChillerReturn - self.T_abW))
        # vol_flow_rate_setpoint_abchiller_tank = mass_flow_rate_abchiller_tank / DENSITY_WATER

        # return {"vol_flow_rate_setpoint_tank_bldg":vol_flow_rate_setpoint_tank_bldg,
        #         "vol_flow_rate_setpoint_chiller_tank": vol_flow_rate_setpoint_chiller_tank,
        #         "vol_flow_rate_setpoint_abchiller_tank": vol_flow_rate_setpoint_abchiller_tank}

        return {"flow_rate": vol_flow_rate_setpoint_tank_bldg}

    # def update_parameters(self, timestamp, inputs):
    #     self.MFR_chW = inputs.get("MFR_chW", DEFAULT_MFR_CHW)
    #     self.T_chW = inputs.get("T_chW", DEFAULT_T_CHW)
    #     self.MFR_abW = inputs.get("MFR_abW", DEFAULT_MFR_ABW)
    #     self.T_abW = inputs.get("T_abW", DEFAULT_T_ABW)
    #     self.MFR_chwBldg = inputs.get("MFR_chwBldg", DEFAULT_MFR_CHWBLDG)
    #     self.T_chwBldgReturn = inputs.get("T_chwBldgReturn", DEFAULT_T_CHWBLDGRETURN)
    #     self.T_chwChillerReturn = inputs.get("T_chwChillerReturn", DEFAULT_T_CHWCHILLERRETURN)
    #     self.T_chwStorageSupply = inputs.get("T_chwStorageSupply", DEFAULT_T_CHWSTORAGESUPPLY)
    #
    # def getNodeTemperatures(self, Nodes_tminus1):
    #     n = len(Nodes_tminus1)
    #     if self.insulated_tank:
    #         k_tank = 0.03  # thermal conductivity of tank wall and insulation, in W/m-K
    #         x_tank = 0.0254  # thickness of tank wall and insulation, in m
    #     else:
    #         k_tank = 16.0  # thermal conductivity of tank wall, in W/m-K
    #         x_tank = 0.01  # thickness of tank wall, in m
    #
    #     if self.fluid_type == 'water':
    #         cp_fluid = 4186  # j/kg-K
    #         rho_fluid = 1.000  # kg/L
    #         k_fluid = 0.609  # W/m-K
    #     elif self.fluid_type == 'glycolwater30':
    #         cp_fluid = 3913  # j/kg-K
    #         rho_fluid = 1.026  # kg/L
    #         k_fluid = 0.4707  # W/m-K
    #     elif self.fluid_type == 'glycolwater50':
    #         cp_fluid = 3558  # j/kg-K
    #         rho_fluid = 1.041  # kg/L
    #         k_fluid = 0.378  # W/m-K
    #
    #     m_fluid = rho_fluid * self.tank_volume / n
    #
    #     pi = 3.141596
    #     TankRadius = pow(((self.tank_volume / 1000) / (pi * self.tank_height)), 0.5)
    #     Circ = pi * 2 * TankRadius
    #     ExtAreaPerSegment = Circ * self.tank_height / n
    #     CrossSecArea = pi * TankRadius * TankRadius
    #     MFR_netDown = self.MFR_chwBldg - self.MFR_abW - self.MFR_chW
    #
    #     # inversion mixing flow rate, set arbitrarily at 50% of flow rate for full mixing of one node to the next in one timestep
    #     MFR_inmix = 0.50 * self.tank_volume * rho_fluid / (n * self.timestep)
    #
    #     Nodes_t = []
    #     for i in range(0, len(Nodes_tminus1)):
    #
    #         # Calculate combined conduction and convection from tank vertical walls (standby losses)
    #         L = self.tank_height  # Characteristic Length for convection
    #
    #         if self.insulated_tank:
    #             BulkAir = (self.thermal_zone_temp)
    #         else:
    #             BulkAir = (Nodes_tminus1[i] + self.thermal_zone_temp) / 2
    #
    #         rho_air = 0.0000175 * pow(BulkAir, 2) - (0.00487 * BulkAir) + 1.293
    #         mu = ((0.0049 * BulkAir) + 1.7106) * pow(10, -5)
    #         Cp_air = 1004.0
    #         k_air = 0.00007 * BulkAir + 0.0243  # Thermal conductivity of air, in W/m-K
    #         g = 9.81
    #         beta = 1 / (BulkAir + 273.15)
    #         mu_v = mu / rho_air
    #         alpha_air = k_air / (rho_air * Cp_air)
    #         Pr = Cp_air * mu / k_air  # dimensionless Prandtl Number; Approximately 0.7 to 0.8 for air and other gases
    #         Ra = g * beta * pow(pow(Nodes_tminus1[i] - self.thermal_zone_temp, 2), 0.5) * pow(L, 3) / (mu_v * alpha_air)
    #
    #         # Performing Nusselt number calculation in steps because of problems with a single calculation
    #         Nu1 = 0.387 * pow(Ra, 0.16666)
    #         Nu2 = pow((0.492 / Pr), 0.5625)
    #         Nu3 = pow(Nu2 + 1, 0.296296)
    #         Nu4 = pow((0.825 + Nu1 / Nu3),
    #                   2)  # Nu4 is non-dimensional Nusselt number for a vertical plate from Equation 9.27 in Fundamental of Heat and Mass Transfer, 6th Edition
    #
    #         h_v = Nu4 * k_air / L
    #
    #         q_conv_v = (self.thermal_zone_temp - Nodes_tminus1[i]) / ((1 / h_v) + (x_tank / k_tank))
    #         Q_conv_v = ExtAreaPerSegment * q_conv_v
    #
    #         # Calculate convection from the top of the tank (Q_conv_t)
    #         if i == 0:
    #             L = 2 * TankRadius
    #             Ra = g * beta * pow(pow(Nodes_tminus1[i] - self.thermal_zone_temp, 2), 0.5) * pow(L, 3) / (
    #             mu_v * alpha_air)
    #             Nu = 0.27 * pow(Ra, 0.25)
    #             h_h_t = Nu * k_air / L
    #             q_conv_t = (self.thermal_zone_temp - Nodes_tminus1[i]) / ((1 / h_h_t) + (x_tank / k_tank))
    #         else:
    #             q_conv_t = 0
    #         Q_conv_t = CrossSecArea * q_conv_t
    #
    #         # Calculate convection from the bottom of the tank (Q_conv_b)
    #         if i == (len(Nodes_tminus1) - 1):
    #             L = 2 * TankRadius
    #             Ra = g * beta * pow(pow(Nodes_tminus1[i] - self.thermal_zone_temp, 2), 0.5) * pow(L, 3) / (
    #             mu_v * alpha_air)
    #             Nu = 0.15 * pow(Ra, 0.3333)
    #             h_h_b = Nu * k_air / L
    #             q_conv_b = (self.thermal_zone_temp - Nodes_tminus1[i]) / ((1 / h_h_b) + (x_tank / k_tank))
    #         else:
    #             q_conv_b = 0
    #         Q_conv_b = CrossSecArea * q_conv_b
    #
    #         # Calculate conduction between current node and node above
    #         if i > 0:
    #             q_cond_nminus1 = (Nodes_tminus1[i - 1] - Nodes_tminus1[i]) * k_fluid / (self.tank_height / n)
    #             Q_cond_nminus1 = CrossSecArea * q_cond_nminus1
    #         else:
    #             Q_cond_nminus1 = 0
    #
    #         # Calculate conduction between current node and node below
    #         if i < (len(Nodes_tminus1) - 1):
    #             q_cond_nplus1 = (Nodes_tminus1[i + 1] - Nodes_tminus1[i]) * k_fluid / (self.tank_height / n)
    #             Q_cond_nplus1 = CrossSecArea * q_cond_nplus1
    #         else:
    #             Q_cond_nplus1 = 0
    #
    #         # Calculate heat from water pushing down from above
    #         if MFR_netDown > 0:
    #             if i > 0:
    #                 Q_flow_down = MFR_netDown * cp_fluid * (Nodes_tminus1[i - 1] - Nodes_tminus1[i])
    #             else:
    #                 Q_flow_down = 0
    #         else:
    #             Q_flow_down = 0
    #
    #         # Calculate heat from water pushing up from below
    #         if MFR_netDown <= 0:
    #             if i < (len(Nodes_tminus1) - 1):
    #                 Q_flow_up = (0 - MFR_netDown) * cp_fluid * (Nodes_tminus1[i + 1] - Nodes_tminus1[i])
    #             else:
    #                 Q_flow_up = 0
    #         else:
    #             Q_flow_up = 0
    #
    #         # Calculate cooling at bottom node from chiller and absorption chiller
    #         if (self.MFR_chW + self.MFR_abW > 0) and i == len(Nodes_tminus1) - 1:
    #             T_Source = (self.T_chW * self.MFR_chW + self.T_abW * self.MFR_abW) / (self.MFR_chW + self.MFR_abW)
    #             Q_Source = (self.MFR_chW + self.MFR_abW) * cp_fluid * (T_Source - Nodes_tminus1[i])
    #         else:
    #             Q_Source = 0
    #
    #         # Calculate heating at top from return building water
    #         if (self.MFR_chwBldg > 0) and i == 0:
    #             Q_Use = (self.MFR_chwBldg) * cp_fluid * (self.T_chwBldgReturn - Nodes_tminus1[i])
    #         else:
    #             Q_Use = 0
    #
    #         # Calculate inversion mixing from above
    #         if i > 0:
    #             if Nodes_tminus1[i - 1] < Nodes_tminus1[i]:
    #                 Q_inmix_above = MFR_inmix * cp_fluid * (Nodes_tminus1[i - 1] - Nodes_tminus1[i])
    #             else:
    #                 Q_inmix_above = 0
    #         else:
    #             Q_inmix_above = 0
    #
    #         # Calculate inversion mixing from below
    #         if i < (len(Nodes_tminus1) - 1):
    #             if Nodes_tminus1[i + 1] > Nodes_tminus1[i]:
    #                 Q_inmix_below = MFR_inmix * cp_fluid * (Nodes_tminus1[i + 1] - Nodes_tminus1[i])
    #             else:
    #                 Q_inmix_below = 0
    #         else:
    #             Q_inmix_below = 0
    #
    #         Q_total = Q_cond_nminus1 + Q_cond_nplus1 + Q_conv_v + Q_flow_down + Q_flow_up + Q_Source + Q_Use + Q_inmix_above + Q_inmix_below + Q_conv_b + Q_conv_t
    #
    #         Nodes_t.append(Nodes_tminus1[i] + Q_total * self.timestep / (m_fluid * cp_fluid))
    #
    #     return Nodes_t
    #
