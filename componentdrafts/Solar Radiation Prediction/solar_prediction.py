import numpy as np
import pandas as pd

training_data = [25, 45, 65, 85, 95]


def training():
    #Reading radiation, cloud cover and time data from TMY3 excel file. Note
    #that TMY3 files do not contain Month variable
    data = pd.read_csv("phoenix.csv", header=0)
    # Ttmp = xlsread('phoenix_TMY3.xlsx', 'b3:b8762') * 24
    # Itmp =  xlsread('phoenix_TMY3.xlsx', 'e3:e8762')
    # CCtmp_x = xlsread('phoenix_TMY3.xlsx', 'z3:z8762')
    Ttmp = data["Time (HH:MM)"].values
    for i in range(len(Ttmp)):
        s = Ttmp[i]
        s = s[:s.index(':')]
        Ttmp[i] = float(s)

    Itmp = data["GHI (W/m^2)"].values
    CCtmp_x = data["TotCld (tenths)"].values
    means_xx = training_data

    #----------------------------------------------------------------------
    #Generating zero matrices. This way putting values in these matrices would
    #be faster in subsequent parts of the code.
    Wtmp = np.zeros(8760)
    CC1tmp = np.zeros(8760)
    CC2tmp = np.zeros(8760)
    CC24tmp = np.zeros(8760)
    I1tmp = np.zeros(8760)
    I2tmp = np.zeros(8760)
    I24tmp = np.zeros(8760)
    Id = np.zeros(8736) + 1
    # a = [1, 721, 1393, 2137, 2857, 3601, 4321, 5065, 5809, 6529, 7273, 7993]
    a = [0, 720, 1392, 2136, 2856, 3600, 4320, 5064, 5808, 6528, 7272, 7992]
    b = [720, 1392, 2136, 2856, 3600, 4320, 5064, 5808, 6528, 7272, 7992, 8736]

    #-----------------------------------------------------------------
    #calculating the daytime/night time indicator
    for i in range(8760):
        if Ttmp[i] >= 8 and Ttmp[i] <= 19:
            Wtmp[i] = 1

    #---------------------------------------------------------------------
    #Transforming cloud cove values based on the means given
    CCtmp = np.zeros(8760)
    for i in range(8760):
        absvalues = np.absolute(means_xx - 10 * CCtmp_x[i])
        minimum = np.amin(absvalues) # Nick's Notes: I don't follow the operation going on here
        index = np.argmax(absvalues == minimum)
        CCtmp[i] = means_xx[index] # Nick's Notes: Don't follow this either

    #-------------------------------------------------------------------------
    #calculating the first lag of cloud cover and radiation
    # CC1tmp(2:8760) = CCtmp(1:8759)
    # I1tmp(2:8760) = Itmp(1:8759)
    CC1tmp[1:8760] = CCtmp[0:-1]
    I1tmp[1:8760] = Itmp[0:-1]

    #-------------------------------------------------------------------
    #calculating the second lag of cloud cover and radiation
    # CC2tmp(3:8760) = CCtmp(1:8758)
    # I2tmp(3:8760) = Itmp(1:8758)
    CC2tmp[2:8760] = CCtmp[0:-2]
    I2tmp[2:8760] = Itmp[0:-2]

    #-------------------------------------------------------------------
    #calculating the seasonal lag of cloud cover and radiation
    # CC24tmp(25:8760) = CCtmp(1:8736)
    # I24tmp(25:8760) = Itmp(1:8736)
    CC24tmp[24:8760] = CCtmp[0:8736]
    I24tmp[24:8760] = Itmp[0:8736]

    #------------------------------------------------------------------
    #Removing the first 24 rows as they do not contain full lag values
    # I = Itmp(25:8760)
    # CC = CCtmp(25:8760)
    # I1 = I1tmp(25:8760)
    # I2 = I2tmp(25:8760)
    # I24 = I24tmp(25:8760)
    # CC1 = CC1tmp(25:8760)
    # CC2 = CC2tmp(25:8760)
    # CC24 = CC24tmp(25:8760)
    # W = Wtmp(25:8760)
    # T = Ttmp(25:8760)
    I = Itmp[24:]
    CC = CCtmp[24:]
    I1 = I1tmp[24:]
    I2 = I2tmp[24:]
    I24 = I24tmp[24:]
    CC1 = CC1tmp[24:]
    CC2 = CC2tmp[24:]
    CC24 = CC24tmp[24:]
    W = Wtmp[24:]
    T = Ttmp[24:]

    #------------------------------------------------------------------------
    #forming training sets
    # x1(1:8736, 1) = Id
    # x1(1:8736, 2) = T
    # x1(1:8736, 3) = CC
    # x1(1:8736, 4) = CC24
    # x1(1:8736, 5) = I24
    # x1(1:8736, 6) = CC1
    # x1(1:8736, 7) = CC2
    # x1(1:8736, 8) = I1
    # x1(1:8736, 9) = I2

    x1 = np.column_stack((Id, T, CC, CC24, I24, CC1, CC2, I1, I2))

    print "x1.dtype", x1.dtype

    # x2(1:8736, 1) = Id
    # x2(1:8736, 2) = T
    # x2(1:8736, 3) = CC
    # x2(1:8736, 4) = CC24
    # x2(1:8736, 5) = I24

    x2 = np.column_stack((Id, T, CC, CC24, I24))

    #fitting The models and finding the coefficients
    model1 = np.zeros((12, 9))
    model2 = np.zeros((12, 5))
    for i in range(12):
        # trdata1 = x1(a(i):b(i), 1:9)
        # trdata2 = x2(a(i):b(i), 1:5)
        # model1(i, 1:9) = lscov(trdata1, I(a(i):b(i)), W(a(i):b(i)))
        # model2(i, 1:5) = lscov(trdata2, I(a(i):b(i)), W(a(i):b(i)))
        W_diag = np.diag(W[a[i]:b[i]])
        W_diag = np.sqrt(W_diag)
        Bw = np.dot(I[a[i]:b[i]], W_diag)

        Aw = np.dot(W_diag, x1[a[i]:b[i]])
        X, residuals, rank, s = np.linalg.lstsq(Aw, Bw)
        model1[i] = X

        Aw = np.dot(W_diag, x2[a[i]:b[i]])
        X, residuals, rank, s = np.linalg.lstsq(Aw, Bw)
        model2[i] = X



    return model1, model2


# def deployment(model1, model2):
    # #getting the current time which determines how the prediction method should
    # #work
    # a = clock
    # t2 = a(4)
    # t2 = 10
    # #----------------------------------------------------------------------
    # #Reading input variable values from Data file
    # t = xlsread('Deployment_Data.xlsx', 'a2:a13')
    # cc = xlsread('Deployment_Data.xlsx', 'b2:b13')
    # cc24 = xlsread('Deployment_Data.xlsx', 'c2:c13')
    # I = xlsread('Deployment_Data.xlsx', 'd2:d13')
    # I24 = xlsread('Deployment_Data.xlsx', 'e2:e13')
    # pr4 =  xlsread('Deployment_Data.xlsx', 'f2:I13')
    # ci = xlsread('Deployment_Data.xlsx', 'k2:k13')
    # #Specifying the number of predictions that will be made for each time period
    # np = [2, 2, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1]

    # #-------------------------------------------------------------------
    # #Telling to model the current month. This will be used to choose model.
    # v = datevec(now)
    # month = v(2)
    # #selecting appropriate models
    # mo1 = model1(month, 1:9)
    # mo2 = model2(month, 1:5)
    # #------------------------------------------------
    # #Defining zero matrices and vectors
    # le = np.zeros(4, 2)
    # e = np.zeros(4, 4)
    # data = np.zeros(12, 9)
    # pr24 = np.zeros(12, 1)
    # #--------------------------------------------------
    # #fiiling the first column of data with 1's
    # data(1:12, 1) = np.zeros(12, 1)+1
    # #Putting input variables into the data matrix
    # data(1:12, 2) = t
    # data(1:12, 3) = cc
    # data(1:12, 4) = cc24
    # data(1:12, 5) = I24
    # #-----------------------------------------
    # #calculating and filling in the first lag of cloud covre and radiarion
    #  data(2:12, 6) = cc(1:11)
    #  data(2:12, 8) = I(1:11)
    # #calculating and filling the second lag of cloud cover and radiation
    # data(3:12, 7) = cc(1:10)
    # data(3:12, 9) = I(1:10)
    # #-------------------------------------------------------
    # #calculating 24-hours predictions
    # for i = 1:12
    #     # pr24(i) = max(data(i, 1:5) * mo2' , 0) # complex conjugate transpose?
    # end
    # #obtaining i as the index
    # i = t2-7
    # #-----------------------
    # #x = data(time-7, 1:9)
    # if t2 > = 10 && t2 < = 19
    #    #setting the number of predictions that will be made at ach time
    #    k = np(i)
    #    #-------------------------------------------------------------------
    #    for j = 0:k-1
    #        #The following lines of code updates  the last estimate for the time period of interest
    #        if j == 0
    #           ci(i) = ci(i)+1
    #           # pr4(i, ci(i)) = max(data(i, 1:9) * mo1', 0) # complex conjugate transpose?
    #        else
    #            x(1:7) = data(t2-7+j, 1:7)
    #            x(8) = pr4(i+j-1, ci(i+j-1))
    #            x(9) = pr4(i+j-2, ci(i+j-2))
    #            ci(i+j) = ci(i+j)+1
    #            # pr4(i+j, ci(i+j)) = max(x * mo1', 0) # complex conjugate transpose?
    #        end
    #    end
    # elseif t2 == 8
    #         ci(i) = ci(i)+1
    #         # pr4(i, ci(i)) = max(data(i, 1:5) * mo2', 0)
    #         ci(i+1) = ci(i+1)+1
    #         # pr4(i+1, ci(i)) = max(data(i+1, 1:5) * mo2', 0)
    #         x(1:7) = data(i+2, 1:7)
    #         x(8) = pr4(i+1, ci(i+1))
    #         x(9) = pr4(i, ci(i))
    #         ci(i+2) = ci(i+2)+1
    #         # pr4(i+2, ci(i+2)) = max(x * mo1', 0)
    #         x(1:7) = data(i+3, 1:7)
    #         x(8) = pr4(i+2, ci(i+2))
    #         x(9) = pr4(i+1, ci(i+1))
    #         ci(i+3) = ci(i+3)+1
    #         # pr4(i+3, ci(i+3)) = max(x * mo1', 0)
    #     elseif t2 == 9
    #         ci(i+1) = ci(i+1)+1
    #         # pr4(i+1, ci(i+1)) = max(data(i+1, 1:9) * mo1', 0)
    #         x(1:7) = data(i+2, 1:7)
    #         x(8) = pr4(i+1, ci(i+1))
    #         x(9) = pr4(i, ci(i+2))
    #         ci(i+2) = ci(i+2)+1
    #         # pr4(i+2, ci(i+2)) = max(x * mo1', 0)
    #         x(1:7) = data(i+3, 1:7)
    #         x(8) = pr4(i+2, ci(i+2))
    #         x(9) = pr4(i+1, ci(i+2))
    #         ci(i+3) = ci(i+3)+1
    #         # pr4(i+3, ci(i+3)) = max(x * mo1', 0)
    # end

    # print "updated deployment data"
    # print "pr4", pr4
    # print "pr24", pr24
    # print "ci", ci


if __name__ == '__main__':
    model1, model2 = training()
    print model1
    print model2
    # deployment(model1, model2)
