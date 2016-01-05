import numpy as np

def rand(s):
    rp = np.random.rand(s)
    maxVal = max(rp) 
    minVal = min(rp) 
    rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
    return lambda n: (rp[np.mod(n,s)])

def randn(s):
    rp = np.random.randn(s)
    maxVal = max(rp) 
    minVal = min(rp) 
    rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
    return lambda n: (rp[np.mod(n,s)])

def getPatterns():
    np.random.seed(None)

    patts=([])
    for i in range(75):
        patts.append(1)

    patts[0] = lambda n: np.sin(2 * np.pi * n / 10)
    patts[1] = lambda n: np.sin(2 * np.pi * n / 10)
    patts[2] = lambda n: np.sin(2 * np.pi * n / 15)
    patts[3] = lambda n: np.sin(2 * np.pi * n / 20)
    patts[4] = lambda n: +(1 == np.mod(n,20))
    patts[5] = lambda n: +(1 == np.mod(n,10))
    patts[6] = lambda n: +(1 == np.mod(n,7))
    patts[7] = lambda n: 0
    patts[8] = lambda n: 1
    patts[9] = randn(4)
    patts[10] = rand(5)
    patts[11] = rand(6)
    patts[12] = rand(7)
    patts[13] = rand(8)
    patts[14] = lambda n: 0.5* np.sin(2 * np.pi * n / 10) + 0.5
    patts[15] = lambda n: 0.2* np.sin(2 * np.pi * n / 10) + 0.7
    patts[16] = randn(3)
    patts[17] = randn(9)
    patts[18] = randn(10)
    patts[19] = lambda n: 0.8
    patts[20] = lambda n: np.sin(2 * np.pi * n / np.sqrt(27))
    patts[21] = lambda n: np.sin(2 * np.pi * n / np.sqrt(19))
    patts[22] = lambda n: np.sin(2 * np.pi * n / np.sqrt(50))
    patts[23] = lambda n: np.sin(2 * np.pi * n / np.sqrt(75))
    patts[24] = lambda n: np.sin(2 * np.pi * n / np.sqrt(10))
    patts[25] = lambda n: np.sin(2 * np.pi * n / np.sqrt(110))
    patts[26] = lambda n: 0.1 * np.sin(2 * np.pi * n / np.sqrt(75))
    patts[27] = lambda n: 0.5 * (np.sin(2 * np.pi * n / np.sqrt(20)) + np.sin(2 * np.pi * n / np.sqrt(40)))
    patts[28] = lambda n: 0.33 * np.sin(2 * np.pi * n / np.sqrt(75))
    patts[29] = lambda n: np.sin(2 * np.pi * n / np.sqrt(243))
    patts[30] = lambda n: np.sin(2 * np.pi * n / np.sqrt(150))
    patts[31] = lambda n: np.sin(2 * np.pi * n / np.sqrt(200))
    patts[32] = lambda n: np.sin(2 * np.pi * n / 10.587352723)
    patts[33] = lambda n: np.sin(2 * np.pi * n / 10.387352723)
    patts[34] = rand(7)
    patts[35] = lambda n: np.sin(2 * np.pi * n / 12)
    patts[36] = randn(5)
    patts[37] = lambda n: np.sin(2 * np.pi * n / 11)
    patts[38] = lambda n: np.sin(2 * np.pi * n / 10.17352723)
    patts[39] = lambda n: np.sin(2 * np.pi * n / 5)
    patts[40] = lambda n: np.sin(2 * np.pi * n / 6)
    patts[41] = lambda n: np.sin(2 * np.pi * n / 7)
    patts[42] = lambda n: np.sin(2 * np.pi * n / 8)
    patts[43] = lambda n: np.sin(2 * np.pi * n / 9)
    patts[44] = lambda n: np.sin(2 * np.pi * n / 12)
    patts[45] = lambda n: np.sin(2 * np.pi * n / 13)
    patts[46] = lambda n: np.sin(2 * np.pi * n / 14)
    patts[47] = lambda n: np.sin(2 * np.pi * n / 10.8342522)
    patts[48] = lambda n: np.sin(2 * np.pi * n / 11.8342522)
    patts[49] = lambda n: np.sin(2 * np.pi * n / 12.8342522)
    patts[50] = lambda n: np.sin(2 * np.pi * n / 13.1900453)
    patts[51] = lambda n: np.sin(2 * np.pi * n / 7.1900453)
    patts[52] = lambda n: np.sin(2 * np.pi * n / 7.8342522)
    patts[53] = lambda n: np.sin(2 * np.pi * n / 8.8342522)
    patts[54] = lambda n: np.sin(2 * np.pi * n / 9.8342522)
    patts[55] = lambda n: np.sin(2 * np.pi * n / 5.1900453)
    patts[56] = lambda n: np.sin(2 * np.pi * n / 5.804531)
    patts[57] = lambda n: np.sin(2 * np.pi * n / 6.4900453)
    patts[58] = lambda n: np.sin(2 * np.pi * n / 6.900453)
    patts[59] = lambda n: np.sin(2 * np.pi * n / 13.900453)
    patts[60] = randn(10)
    patts[61] = lambda n: +(1 == np.mod(n,3))
    patts[62] = lambda n: +(1 == np.mod(n,4))
    patts[63] = lambda n: +(1 == np.mod(n,5))
    patts[64] = lambda n: +(1 == np.mod(n,6))
    patts[65] = randn(4)
    patts[66] = rand(5)
    patts[67] = rand(6)
    patts[68] = rand(7)
    patts[69] = rand(8)
    patts[70] = randn(4)
    patts[71] = rand(5)
    patts[72] = rand(6)
    patts[73] = rand(7)
    patts[74] = rand(8)
    np.random.seed(None)


    return patts