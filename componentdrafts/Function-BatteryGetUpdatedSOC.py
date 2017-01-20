
# coding: utf-8

# In[ ]:

def getUpdatedSOC(Capacity, minimumSOC, InputPowerRequest, InputCurrent, timestep, PrevSOC,InvEFF_Charge,IR_Charge,IR_DisCharge,InvEFF_DisCharge,Idle_A,Idle_B):
    import numpy as np
    
    deltaSOC=0 #change in state of charge during this timestep. Initialization of variable
    if InputPowerRequest>0: #for batttery charging
        P2= InputPowerRequest*InvEFF_Charge-InputCurrent*InputCurrent*IR_Charge #P2 is power recieved into battery storage
        deltaSOC= P2*timestep/3600/Capacity 
    if InputPowerRequest<0:
        P2 = InputPowerRequest/InvEFF_DisCharge-InputCurrent*InputCurrent*IR_DisCharge #P2 is power released from battery storage
        deltaSOC= P2*timestep/3600/Capacity
    deltaSOC_self=(Idle_A+Idle_B*PrevSOC)*timestep/3600
    SOC = PrevSOC+deltaSOC+deltaSOC_self
    
    InputPower=InputPowerRequest
    if SOC < minimumSOC:
        if PrevSOC<minimumSOC and InputPowerRequest<0:
            InputPower=0
            SOC = PrevSOC+deltaSOC_self
        if PrevSOC>minimumSOC and InputPowerRequest<0:
            InputPower=InputPowerRequest*(PrevSOC-minimumSOC)/(PrevSOC-SOC)
            InputCurrent=InputCurrent*InputPower/InputPowerRequest
            P2 = InputPower/InvEFF_DisCharge-InputCurrent*InputCurrent*IR_DisCharge
            deltaSOC= P2*timestep/3600/Capacity
            SOC = PrevSOC+deltaSOC+deltaSOC_self
    if SOC > 1:
            InputPower=InputPowerRequest*(1-PrevSOC)/(SOC-PrevSOC)
            InputCurrent=InputCurrent*InputPower/InputPowerRequest
            P2= InputPower*InvEFF_Charge-InputCurrent*InputCurrent*IR_Charge
            deltaSOC= P2*timestep/3600/Capacity
            SOC = PrevSOC+deltaSOC+deltaSOC_self
    if SOC < 0:
            SOC=0
        
    
    Outputs = [SOC, InputPower]
    return Outputs

