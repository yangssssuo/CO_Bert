import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
import pandas as pd
import torch
from scipy.signal import find_peaks


def Gauss(x,A,xc,sigma):
    y = A/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-xc)**2/(2*sigma**2))
    return y

def Lorentz(x,A,xc,sigma):
    y = (A/np.pi)*(sigma/((x-xc)**2 + sigma**2))
    return y

def Voigt(x, y0, amp, pos, fwhm, shape = 1):
    tmp = 1/wofz(np.zeros((len(x))) + 1j*np.sqrt(np.log(2.0))*shape).real
    return y0+tmp*amp*wofz(2*np.sqrt(np.log(2.0))*(x-pos)/fwhm+1j*np.sqrt(np.log(2.0))*shape).real

class SpecBroadener():
    '''
    展宽脚本,按照voigt峰形展宽
    inp_path:输入文件的路径，第一行是所有振动模式,'72,1',1是默认的峰类型,不用修改
    spec_len:输出光谱的总长度
    resolu:分辨率,在总长上每resolu个波数取一个点
    broaden_factor:展宽因子，半峰宽的数值
    mode:光谱模式, 'Raman' 'IR'
    make_noise:给低频加随机谱峰,默认False
    bias:给所有值加上偏置,默认False
    '''
    def __init__(self,inp_path,spec_lenth,resolu,broaden_factor,mode = 'Raman',make_noise=False,bias=False) -> None:
        self.path = inp_path
        self.name = inp_path.split('.')[0]
        aaa = pd.read_csv(self.path,delimiter='\t')
        self.n_mode = int(aaa.columns.values[0])
        self.shape = int(aaa.columns.values[1])
        self.freq = list(aaa[f'{self.n_mode}'])
        self.inti = list(aaa[f'{self.shape}'])
        self.spec_length = spec_lenth
        self.resolu = resolu
        self.mode = mode
        self.bias = bias
        self.make_noise = make_noise
        self.fwhm = broaden_factor
        self.x, self.y = self._form_data()
        
        # self.noise = self._get_noise()

    def _Voigt(self,x, y0, amp, pos, fwhm, shape = 1):
        tmp = 1/wofz(np.zeros((len(x))) + 1j*np.sqrt(np.log(2.0))*shape).real
        return y0+tmp*amp*wofz(2*np.sqrt(np.log(2.0))*(x-pos)/fwhm+1j*np.sqrt(np.log(2.0))*shape).real

    def _form_data(self):
        x = np.array([i * self.resolu for i in range(int(self.spec_length / self.resolu))])
        y0 = np.array([0.0 for _ in range(int(self.spec_length / self.resolu))])
        yf = y0
        for n in range(self.n_mode):
            if self.bias:
                y = self._Voigt(x,y0,self.inti[n],self.freq[n] * np.random.choice([0.01 * i for i in range(80,95)]),self.fwhm)
            else:
                y = self._Voigt(x,y0,self.inti[n],self.freq[n],self.fwhm)
            yf = yf + y
        if self.make_noise:
            self._get_noise()
            self._gauss_noise()
            if self.mode == 'Raman':
                yf = yf + self.noise
            elif self.mode == 'IR': 
                yf = yf + 0.1 * self.noise
            # yf = yf + self.gauss
            # yf = torch.from_numpy(yf)
            # yf = torch.relu(yf).squeeze().numpy()
        return x, yf
    
    def _get_noise(self):
        noi_num = np.random.randint(6,12)
        noi_freq = list(np.random.rand(noi_num) * 1000)
        noi_inti = list(np.random.rand(noi_num) * 2)
        x = np.array([i * self.resolu for i in range(int(self.spec_length / self.resolu))])
        noi0 = np.array([0.0 for _ in range(int(self.spec_length / self.resolu))])
        for n in range(noi_num):
            noi = self._Voigt(x,noi0,noi_inti[n],noi_freq[n],20)
            noi0 = noi0 + noi
        self.noise = noi0

    def _gauss_noise(self):
        x = np.array([i * self.resolu for i in range(int(self.spec_length / self.resolu))])
        noise = torch.normal(0.0,0.1,(1,len(x)))
        noise = noise.numpy()
        self.gauss = noise

    def _make_plateau(self):
        pass

    def draw_pict(self):
        # x, y = self._form_data()
        plt.plot(self.x,self.y)
        plt.savefig(f'{self.name}.png')
        plt.cla()

    def save_data(self):
        # x, y = self._form_data()
        with open(f'{self.name}.txt','a') as f:
            for i in range(len(self.x)):
                f.writelines(str(self.x[i])+','+str(self.y[i]) + '\n')


if __name__ == '__main__':
    # clear = SpecBroadener('/home/yanggk/H2L/utils/sample.inp',100,0.5,4,'IR',make_noise=False,bias=False)
    # noise = SpecBroadener('/home/yanggk/H2L/utils/sample2.inp',100,0.5,4,'IR',make_noise=False,bias=False)
    # plt.figure(dpi=600)
    # plt.plot(noise.x,noise.y,label='Raw Spectrum',linewidth=0.8)
    # plt.plot(clear.x,clear.y,label='Ideal Spectrum',linewidth=1)
    # plt.legend()
    # plt.title('Wrong Peaks')
    # plt.savefig('fig1a.png')
    # clear.draw_pict()
    for idx in range(1,4000): 
        try:
            xxx = SpecBroadener(f'/home/yanggk/Data/H2L_Data/tsps/Au-50/Raman_inp/{idx}.inp',4000,1,8,mode='Raman',make_noise=False,bias=False)
            # xxx.draw_pict()
            xxx.save_data()
            # yyy = SpecBroadener(f'/home/yanggk/Data/HW/Raman_inp/{idx}.inp',4000,1,20,mode='Raman',make_noise=False,bias=True)
            # yyy.draw_pict()
            # yyy.save_data()
            # print(idx,'/ 4000 Done')
        except: pass