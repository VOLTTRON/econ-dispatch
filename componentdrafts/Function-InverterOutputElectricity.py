
# coding: utf-8

# In[ ]:



def InverterElecOut(PLF,Efficiency, ElecIn,Cap):
    Eff= np.interp(ElecIn/Cap, PLF, Efficiency)
    return Eff*ElecIn                            
                      

