# coding: utf-8

# In[ ]:

import numpy.matlib

def getNodeTemperatures(dt, MFR_chW, T_chW, MFR_abW, T_abW, MFR_chwBldg,T_chwBldgReturn,Nodes_tminus1,TankVolume,TankHeight,Fluid,InsulatedTank):

    # assumed constant teperature of thermal zone
    T_zone = 21

    # assumed convection heat transfer coefficient
    h = 4.00
    n = len(Nodes_tminus1)
    if InsulatedTank:
        k_tank = 0.03 #thermal conductivity of tank wall and insulation, in W/m-K
        x_tank = 0.0254 #thickness of tank wall and insulation, in m
    else:
        k_tank = 16.0 #thermal conductivity of tank wall, in W/m-K
        x_tank = 0.01 #thickness of tank wall, in m

    if Fluid == 'Water' or Fluid == 'water':
        cp_fluid = 4186 #j/kg-K
        rho_fluid = 1.000 #kg/L
        k_fluid = 0.609 # W/m-K
    elif Fluid == 'GlycolWater30' or Fluid == 'glycolwater30':
        cp_fluid = 3913 #j/kg-K
        rho_fluid = 1.026 #kg/L
        k_fluid = 0.4707 # W/m-K
    elif Fluid == 'GlycolWater50' or Fluid == 'glycolwater50':
        cp_fluid = 3558 #j/kg-K
        rho_fluid = 1.041 #kg/L
        k_fluid = 0.378 # W/m-K
    else:
        cp_fluid = 4186 #j/kg-K
        rho_fluid = 1.000 #kg/L
        k_fluid = 0.609 # W/m-K

    m_fluid = rho_fluid * TankVolume / n

    pi = 3.141596
    TankRadius = pow(((TankVolume / 1000) / (pi * TankHeight)), 0.5)
    Circ = pi * 2 * TankRadius
    ExtAreaPerSegment = Circ * TankHeight / n
    CrossSecArea = pi * TankRadius * TankRadius
    MFR_netDown = MFR_chwBldg - MFR_abW - MFR_chW

    #inversion mixing flow rate, set arbitrarily at 50% of flow rate for full mixing of one node to the next in one timestep
    MFR_inmix = 0.50 * TankVolume * rho_fluid / (n * dt)

    i = 0
    Nodes_t = []
    for i in range(0, len(Nodes_tminus1)):

        # Calculate combined conduction and convection from tank vertical walls (standby losses)
        L = TankHeight #Characteristic Length for convection

        if InsulatedTank:
            BulkAir = (T_zone)
        else:
            BulkAir = (Nodes_tminus1[i] + T_zone) / 2

        rho_air = 0.0000175 * pow(BulkAir, 2) - (0.00487 * BulkAir) + 1.293
        mu = ((0.0049 * BulkAir) + 1.7106) * pow(10, -5)
        Cp_air = 1004.0
        k_air = 0.00007 * BulkAir + 0.0243 # Thermal conductivity of air, in W/m-K
        g = 9.81
        beta = 1 / (BulkAir + 273.15)
        mu_v = mu / rho_air
        alpha_air = k_air / (rho_air * Cp_air)
        Pr = Cp_air * mu / k_air # dimensionless Prandtl Number; Approximately 0.7 to 0.8 for air and other gases
        Ra = g * beta * pow(pow(Nodes_tminus1[i] - T_zone, 2), 0.5) * pow(L, 3) / (mu_v * alpha_air)

        # Performing Nusselt number calculation in steps because of problems with a single calculation
        Nu1 = 0.387 * pow(Ra, 0.16666)
        Nu2 = pow((0.492 / Pr), 0.5625)
        Nu3 = pow(Nu2+1, 0.296296)
        Nu4 = pow((0.825 + Nu1 / Nu3), 2) # Nu4 is non-dimensional Nusselt number for a vertical plate from Equation 9.27 in Fundamental of Heat and Mass Transfer, 6th Edition

        h_v = Nu4 * k_air / L


        q_conv_v = (T_zone - Nodes_tminus1[i]) / ((1 / h_v) + (x_tank / k_tank))
        Q_conv_v = ExtAreaPerSegment * q_conv_v

        #Calculate convection from the top of the tank (Q_conv_t)
        if i == 0:
            L = 2 * TankRadius
            Ra = g * beta * pow(pow(Nodes_tminus1[i] - T_zone, 2), 0.5) * pow(L, 3) / (mu_v * alpha_air)
            Nu = 0.27 * pow(Ra, 0.25)
            h_h_t = Nu * k_air / L
            q_conv_t = (T_zone - Nodes_tminus1[i]) / ((1 / h_h_t) + (x_tank / k_tank))
        else:
            q_conv_t = 0
        Q_conv_t = CrossSecArea * q_conv_t

        #Calculate convection from the bottom of the tank (Q_conv_b)
        if i == (len(Nodes_tminus1) - 1):
            L = 2 * TankRadius
            Ra = g * beta * pow(pow(Nodes_tminus1[i]-T_zone,2),0.5) * pow(L, 3) /(mu_v * alpha_air)
            Nu = 0.15 * pow(Ra, 0.3333)
            h_h_b = Nu * k_air / L
            q_conv_b = (T_zone - Nodes_tminus1[i]) / ((1 / h_h_b) + (x_tank / k_tank))
        else:
            q_conv_b = 0
        Q_conv_b = CrossSecArea * q_conv_b

        # Calculate conduction between current node and node above
        if i > 0:
            q_cond_nminus1 = (Nodes_tminus1[i-1] - Nodes_tminus1[i]) * k_fluid / (TankHeight / n)
            Q_cond_nminus1 = CrossSecArea * q_cond_nminus1
        else:
            Q_cond_nminus1 = 0

        # Calculate conduction between current node and node below
        if i < (len(Nodes_tminus1)-1):
            q_cond_nplus1 = (Nodes_tminus1[i+1] - Nodes_tminus1[i]) * k_fluid / (TankHeight / n)
            Q_cond_nplus1 = CrossSecArea * q_cond_nplus1
        else:
            Q_cond_nplus1 = 0

        # Calculate heat from water pushing down from above
        if MFR_netDown > 0:
            if i > 0:
                Q_flow_down = MFR_netDown * cp_fluid * (Nodes_tminus1[i-1] - Nodes_tminus1[i])
            else:
                Q_flow_down = 0
        else:
            Q_flow_down = 0

        # Calculate heat from water pushing up from below
        if MFR_netDown <= 0:
            if i < (len(Nodes_tminus1) - 1):
                Q_flow_up = (0 - MFR_netDown) * cp_fluid * (Nodes_tminus1[i+1] - Nodes_tminus1[i])
            else:
                Q_flow_up = 0
        else:
            Q_flow_up = 0

        # Calculate cooling at bottom node from chiller and absorption chiller
        if (MFR_chW + MFR_abW > 0) and i == len(Nodes_tminus1) - 1:
            T_Source = (T_chW * MFR_chW + T_abW * MFR_abW) / (MFR_chW + MFR_abW)
            Q_Source = (MFR_chW + MFR_abW) * cp_fluid * (T_Source - Nodes_tminus1[i])
        else:
            Q_Source = 0

        # Calculate heating at top from return building water
        if (MFR_chwBldg > 0) and i == 0:
            Q_Use = (MFR_chwBldg) * cp_fluid * (T_chwBldgReturn - Nodes_tminus1[i])
        else:
            Q_Use = 0

        #Calculate inversion mixing from above
        if i > 0:
            if Nodes_tminus1[i-1] < Nodes_tminus1[i]:
                Q_inmix_above = MFR_inmix * cp_fluid * (Nodes_tminus1[i-1] - Nodes_tminus1[i])
            else:
                Q_inmix_above = 0
        else:
            Q_inmix_above = 0

        #Calculate inversion mixing from below
        if i < (len(Nodes_tminus1) - 1):
            if Nodes_tminus1[i+1] > Nodes_tminus1[i]:
                Q_inmix_below = MFR_inmix * cp_fluid * (Nodes_tminus1[i+1] - Nodes_tminus1[i])
            else:
                Q_inmix_below = 0
        else:
            Q_inmix_below = 0

        Q_total = Q_cond_nminus1 + Q_cond_nplus1 + Q_conv_v + Q_flow_down + Q_flow_up + Q_Source + Q_Use + Q_inmix_above + Q_inmix_below + Q_conv_b + Q_conv_t

        Nodes_t.append(Nodes_tminus1[i] + Q_total * dt / (m_fluid * cp_fluid))

    return Nodes_t
