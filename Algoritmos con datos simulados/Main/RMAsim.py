# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:50:16 2019
-------------------------------------------------------------
                Algoritmo "Range Migration"(RMA)
                        (Simulated)
-------------------------------------------------------------
Descripcion:
 . El presente algoritmo asume las siguientes consideraciones:
    - Riel en el eje X.
 . Se reconstruyen tanto targets individuales como matrices de targets.
 . Pruebas con datos simulados.
 . Con Decimación.
@author: LUIS_DlCp
"""
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
import sarPrm as sp # Modulo creado para definir los parametros del sistema
import drawFigures as dF # Modulo creado para graficar
import SARdata as Sd # Modulo creado para obtner el Historico de Fase teorico(Datos simulados)
import timeit

#-----------------------LECTURA DE PARÁMETROS-----------------------------
prm = sp.get_parameters()
c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
Lx,Ly,dx,dy = prm['w'],prm['h'],prm['dw'],prm['dh'] # Dimensiones de la imagen
#dec_x,dec_y = 31,7 #62,25 #224,89 #229,91 #280,111#62,25 # Estos valores generan una resolucion de 0.25m #100,25 # Decimacion en ambos ejes
Fkx,Fky = int(Lx/dx)+1,int(Ly/dy)+1#41,41 # Numero de pixeles finales si hay decimacion
show = True

def get_SAR_data():
    """ Obtiene el histórico de fase ya sea simulado o real"""
    # Cálculo de parámetros
    It, Rt = sp.get_scalar_data2() # Coordenadas(m) y magnitud del target respectivamente
    dp=Ls/(Np-1) # Paso del riel(m)
    fi=fc-BW/2 # Frecuencia inferior(GHz)
    fs=fc+BW/2 # Frecuencia superior(GHz)

    # Cálculo de las resoluciones
    rr_r=c/(2*BW) # Resolución en rango
    rr_a=c/(2*Ls*fc) # Resolución en azimuth(Angulo)

    #-----------------VERIFICACIÓN DE CONDICIONES------------------------
    # Rango máximo
    R_max=Nf*c/(2*BW)
    # Paso del riel máximo
    dx_max=c/(fc*4*np.sin(theta*np.pi/180)) # Theta en grados sexagesimales

    print("------------------------------------------------------")
    print("--------------INFORMACIÓN IMPORTANTE------------------")
    print("------------------------------------------------------")
    print("- Resolución en rango(m) : ", rr_r)
    print("- Resolución en azimuth(rad): ", rr_a)
    print("------------------------------------------------------")
    print("- Rango máximo permitido(m): ", R_max)
    print("------------------------------------------------------")
    print("______¿Se cumplen las siguientes condiciones?_________")
    print("Rango máximo del target <= rango máximo?: ", Rt.T[1].max()<=R_max) # Ponerle un try-except
    print("Paso del riel <= paso máximo?: ", dp<=dx_max) # Evita el aliasing en el eje de azimuth
    print("------------------------------------------------------")

    #----------------OBTENCIÓN DEL HISTÓRICO DE FASE---------------------
    Ski = Sd.get_phaseH(prm,It,Rt)
    #np.save("RawData_prueba1",Ski)
    #Ski = np.load("RawData_prueba1.npy")

    #-----------------GRÁFICA DEL HISTÓRICO DE FASE-----------------------
    if show:
        dF.plotImage(Ski, x_min=fi, x_max=fs, y_min=-Ls/2, y_max=Ls/2,xlabel_name='Frecuencia(GHz)',
                 ylabel_name='Posición del riel(m)', title_name='Histórico de Fase',unit_bar='',
                 origin_n='upper') #, log=True)

    return {'Ski':Ski, 'dp':dp, 'fi':fi, 'fs':fs, 'R_max':R_max}

def RMA_Algorithm(data1):
    """ Ejecuta el algoritmo RMA"""
    # Lectura de parametros
    global dec_x, dec_y
    Ski = data1['Ski'].copy()
    dp = data1['dp']
    fi = data1['fi']
    fs = data1['fs']
    R_max = data1['R_max']
    Lista_f = np.linspace(fi, fs, Nf) # Vector de frecuencias

    # Grafica el perfil de rango para una posicion del riel(Posicion '0')
    if show:
        dF.rangeProfile(fi,BW,Ski[int(len(Ski)/2)],name_title='Perfil de rango \n(Al principio)')

    # Hamming Windows
    S1 = Ski * np.hamming(len(Ski[0])) # Multiply rows x hamming windows
    S1 = (S1.T*np.hamming(len(S1))).T # Multiply columns x hamming windows

    #----------------ETAPA PREVIA: Range Compression----------------------
    #                       s(x,w) --> s(x,r)
    #---------------------------------------------------------------------
    if show:
        # a) IFFT (eje de rango)
        ifft_len = 10*len(Ski[0])
        S0 = np.fft.ifft(S1, ifft_len, axis=1) # axis 0 representa las filas(axis 1 a las columnas), lo cual quiere decir q se sumaran todas las filas manteniendo constante las columnas
        # S0 = np.fft.fftshift(S0,axes=(1)) # to center spectrum

        dF.plotImage(S0.T, x_min=-Ls/2, x_max=Ls/2, y_min=0, y_max=R_max,xlabel_name='Azimut(m)',
                     ylabel_name='Rango(m)', title_name='Compresión en Rango al HF',unit_bar='dB',
                     log=True,origin_n='lower')

    #----------PRIMERA ETAPA: FFT respecto al eje del riel 'x'----------
    #                       s(x,w) --> s(kx,kr)
    #                           Ski -> S1
    #---------------------------------------------------------------------

    # a) Zero padding en cross range
    zpad = int(Lx/dp) # Initial length of padded signal
    if zpad > len(Ski):
        # Reajuste del Nª de elementos para la decimacion en x
        #if zpad%dec_x!=0:
        #    ad = dec_x-(zpad%dec_x) # To have a hole multiple of dec
        #    zpad += ad # Final length of padded signal
        rows = int(zpad - len(Ski)) # Length of zeros
    else: rows = 0
    S2 = np.pad(S1, [[0, rows], [0, 0]], 'constant', constant_values=0) # Aplica zero padding a ambos extremos de la matriz

    # b) FFT (eje de azimuth)
    S3 = np.fft.fft(S2,axis=0) # axis 0 representa las filas(axis 1 a las columnas), lo cual quiere decir q se sumaran todas las filas manteniendo constante las columnas
    S3 = np.fft.fftshift(S3,axes=(0)) # to center spectrum

    # c) Grafica despues del FFT
    kr=4*np.pi*Lista_f/c # Numeros de onda dependientes de la frecuencia
    kx=np.fft.fftfreq(len(S1))*2*np.pi/dp
    kx=np.fft.fftshift(kx)
    if show:
        dF.plotImage(S3.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2,xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='1D-FFT\n(Dirección de azimut)', origin_n='lower',unit_bar='dB',log=True)

    # d) Recortando las zonas donde no hay mucha señal(-60dB del maximo)
    # Hallando las zonas que cumplen la condicion de los -60dB
    if show:
        S3_test = S3.copy()
        mask = 20*np.log10(S3_test)>=20*np.log10(S3_test.max())-60
        S3_test[~mask] = 0

        dF.plotImage(S3_test.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2,xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='1D-FFT\n(Test de magnitud)', origin_n='lower',unit_bar='dB',log=True)

    # Función para hallar ni a partir de yi
    def set_y(yi,Nt,yo,yf): # yi deseado, numero de valores de y, y inicial, y final
        Yit = (Nt-1)*(yi-yo)/(yf-yo) # float
        ni = int(np.floor(round(Yit,5)))
        return ni

    #ky_inf = set_y(400,len(ky_it),ky_it[0],ky_it[-1])
    #ky_sup = set_y(50,len(ky_it),ky_it[0],ky_it[-1])
    kx_inf = set_y(-400,len(kx),kx[0],kx[-1]) # Desde kx=-400
    kx_sup = set_y(400,len(kx),kx[0],kx[-1]) # Hasta kx=400

    kx = kx[kx_inf+2:kx_sup] # Correccion manual para la decimacion en x
    #ky_it = ky_it[:]
    S3 = S3[kx_inf+2:kx_sup,:]

    # Reajusta el Nº de elementos para la decimacion en "x"

    dec_x = int(np.ceil(len(kx)/Fkx)) # Hallando el factor de decimacion en x
    adx = Fkx*dec_x-len(kx) # Cantidad de valores a agregar
    if show:
        dF.plotImage(S3.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2,xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='1D-FFT\n(Dirección de azimut - Recortado)', origin_n='lower',unit_bar='(dB)',log=True)

    # Range Compression
    if show:
        S3_n = S3/(np.sqrt(np.sum(abs(S3)**2)))
        ifft_len = 10*len(S3_n[0])#np.array([len(S3_n),10*len(S3_n[0])])
        S3_1 = np.fft.ifft(S3_n,ifft_len,axis=1)
        vmin=0
        vmax=5
        dF.plotImage(S3_1.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=0, y_max=R_max,xlabel_name='kx(1/m)',
                     ylabel_name='Rango(m)', title_name='Compresión en Rango y Azimut\n(Despues del FFT)',
                     origin_n='lower',log=True,unit_bar='(dB)')#,vmin=vmin,vmax=vmax)

    #----------------SEGUNDA ETAPA: Matched Filter----------------------
    #                  s(kx,kr) --> sm(kx,kr)
    #                       S3  ->  S4
    #-------------------------------------------------------------------

    # a) Matched Filter
    krr, kxx= np.meshgrid(kr,kx)
    Xc1 = -Ly/2 # Factor de correccion debido a la FFT en "y"
    Xc2 = -Ls/2 # Factor de correccion debido a la FFT en "x"
    Xc3 = 0 #-4

    mask = np.isnan(np.sqrt(krr**2-kxx**2)) # Obtiene las posiciones de los números complejos

    phi_m = -kxx*(Xc2+Xc3)-Xc1*np.sqrt(krr**2-kxx**2)#+Xc1*krr
    S4 = S3 * np.exp(1j*phi_m) # matrix
    S4[mask] = 0+0j

    # c) Grafica despues del Matched Filter
    if show:
        dF.plotImage(S4.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2, xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='Matched Filter', origin_n='lower',unit_bar='(dB)',log=True)

    # d) Range and azimuth compression
    if show:
        S4_n = S4/(np.sqrt(np.sum(abs(S4)**2)))
        ifft_len = 10*len(S4_n[0])#np.array([len(S4_n),10*len(S4_n[0])])
        S4_1 = np.fft.fftshift(S4_n, axes=(0))
        S4_1 = np.fft.ifft(S4_1,ifft_len,axis=1)
        S4_1 = np.fft.fftshift(S4_1, axes=(0,1))
        vmin=0
        vmax=5
        dF.plotImage(S4_1.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=0, y_max=R_max,xlabel_name='kx(1/m)',
                     ylabel_name='Rango(m)', title_name='Compresión en Rango y Azimut\n(Despues del Matched Filter)',
                     origin_n='lower',log=True,unit_bar='(dB)')#,vmin=vmin,vmax=vmax)

    #----------------TERCERA ETAPA: Stolt Interpolation-------------------
    #                  F(kxmn,kymn) --> F(kx,ky)
    #                       S4  -->  S5
    #---------------------------------------------------------------------
    # a) Redefinicion de los ejes
    #kx = kx # vector
    ky = np.sqrt(krr**2-kxx**2) # Genera una matriz, para cada valor de kx [0,....len(ku)-1]
    dk_r = (kr[1]-kr[0])
    #mask = np.isnan(ky)
    #ky[mask] = 0

    # b) Interpolación STOLT
    ky_min = np.min(ky[~(np.isnan(ky))])
    ky_max = np.max(ky[~(np.isnan(ky))])
    ky_it = np.arange(ky_min,ky_max,dk_r) # Puntos de interpolacion en el eje ky

    # Reajusta el Nº de elementos para la decimacion en "y"
    dec_y = int(np.floor(len(ky_it)/Fky)) # Hallando el factor de decimacion en y
    if len(ky_it) % Fky != 0:
        #ad = dec_y-(len(ky_it)%dec_y)
        ady = len(ky_it) % Fky #Fky*dec_y-len(ky_it)
        ky_it = ky_it[ady:]
        #ky_it = np.linspace(ky_min,ky_max,len(ky_it)+ady) # Puntos finales de interpolacion eje ky

    # Interpolacion 1D lineal
    S5 = np.zeros((len(kx),len(ky_it)), dtype=np.complex)

    for i in range(len(kx)):
        interp_fn1 = sc.interpolate.interp1d(ky[i], S1.real[i], bounds_error=False, fill_value=0)#, kind='cubic')
        interp_fn2 = sc.interpolate.interp1d(ky[i], S1.imag[i], bounds_error=False, fill_value=0)#, kind='cubic')
        S5[i] = interp_fn1(ky_it)+1j*interp_fn2(ky_it)

    S5[np.isnan(S5)] = 0+0j

    if show:
        dF.plotImage(S5.T, x_min=kx.min(), x_max=kx.max(), y_min=ky_it.min(), y_max=ky_it.max(),xlabel_name='kx(1/m)',
                     ylabel_name='ky(1/m)', title_name='Interpolación STOLT\n(Sin decimar)', unit_bar='(dB)', log=True)

    # c) Decimacion(Continuacion)
      # - En el eje "kx"
    S6 = np.zeros([int(len(S5)/dec_x),len(S5[0])],dtype=complex)
    ind1 = int(len(S5)/dec_x)

    for i in range(dec_x):
        S6 += S5[:ind1]
        S5 = S5[ind1:]
    S6 /= dec_x

    # Nuevos valores del eje "kx"
    Nkx = len(S6)
    kx_n = np.fft.fftshift(np.fft.fftfreq(Nkx)*Nkx*(kx[1]-kx[0]))
    #kx_n = (np.arange(Nkx)-(Nkx-1)/2)*(kx[1]-kx[0])

      # - En el eje "y"
    S6_aux = S6.T
    S7 = np.zeros([int(len(S6_aux)/dec_y),len(S6_aux[0])], dtype=complex)
    ind2 = int(len(S6_aux)/dec_y)

    for i in range(dec_y):
        S7 += S6_aux[:ind2]
        S6_aux = S6_aux[ind2:]
    S7 = S7.T
    S7 /= dec_y

    # Nuevos valores del eje "ky"
    Nky = len(S7[0])
    ky_n = np.arange(Nky)*(ky_it[1]-ky_it[0])+ky_it.min()

    # d) Grafica despues de la interpolación
    if show:
        dF.plotImage(S7.T, x_min=kx_n.min(), x_max=kx_n.max(), y_min=ky_n.min(), y_max=ky_n.max(), xlabel_name='kx(1/m)',
                 ylabel_name='ky(1/m)', title_name='Interpolación STOLT\n(Decimado)', unit_bar='(dB)', log=True)

    #--------CUARTA ETAPA_previa: 1D-IFFT(range compression)--------------------
    #                  F(kx,ky) --> f(kx,y)
    #----------------------------------------------------------
    if show:
        # a) 1D-IFFT
        ifft_len = 1*np.array(np.shape(S7))
        S7_1 = np.fft.ifft(S7,axis=1)
        S7_1 = np.fft.fftshift(S7_1, axes=(1))

        # b) Definicion de parametros para las graficas
        #Nx = ifft_len[0]
        Ny = ifft_len[1]
        #dk_x = (kx_n[1]-kx_n[0])
        dk_y = (ky_n[1]-ky_n[0])

        # x = (np.arange(Nx)-Nx/2)*(2*np.pi/dk_x/Nx)
        y = (np.arange(Ny)-(Ny-1)/2)*(2*np.pi/dk_y/(Ny-1))-Xc1

        # d) Grafica despues de la interpolación
        dF.plotImage(S7_1.T, x_min=kx_n.min(), x_max=kx_n.max(), y_min=y.min(), y_max=y.max(), xlabel_name='kx(1/m)',
                     ylabel_name='y(m)', title_name='Compresión en rango\n(Despues de la Interpolación)', unit_bar='')

    #-----------------CUARTA ETAPA: 2D-IFFT--------------------
    #                  F(kx,ky) --> f(x,y)
    #                     S2  -->  Sf
    #----------------------------------------------------------
    # a) 2D-IFFT
    ifft_len = np.array([1*len(S7),1*len(S7[0])])#np.array(np.shape(S7)) #: Uncomment si no quiere interpolar
    S8 = np.fft.ifftshift(S3, axes=(0)) # De-centra el eje kx
    S8 = np.fft.ifft2(S8, ifft_len)
    S8 = np.fft.fftshift(S8, axes=(0,1)) # Centra en el origen

    # b) Definicion de parametros para las graficas
    Nx = ifft_len[0]
    Ny = ifft_len[1]
    dk_x = (kx_n[1]-kx_n[0])
    dk_y = (ky_n[1]-ky_n[0])

    x = np.linspace(-np.pi/dk_x,np.pi/dk_x,Nx)-Xc3
    y = np.linspace(-np.pi/dk_y,np.pi/dk_y,Ny)-Xc1

    #x = np.fft.fftfreq(Nx+1)*2*np.pi/dk_x
    #x = np.fft.fftshift(x)-Xc3
    #y = np.fft.fftfreq(Ny)*2*np.pi/dk_y
    #y = np.fft.fftshift(y)-Xc1

    # Compensacion de fase
    y2,x2 = np.meshgrid(y,x)
    kx_c = 0 #10*np.pi #(kx_n[-1]-kx_n[0])/2 # considerar solo cuando no se hace fftshift en x
    ky_c = (kr.max()-kr.min())/2 #4*np.pi + 1.37/5 #ky_n[0]-2*np.pi/0.55-2*np.pi/2#ky_n[int(len(ky_n)/2)] #4*np.pi
    kyo = 0 #-0.57
    S8 *= np.exp(-1j*kx_c*x2+1j*(ky_c*y2+kyo))

    # Normalizando la salida
    I = S8/(np.sqrt(np.sum(abs(S8)**2)))

    if show:
        # GRAFICA DEL PERFIL DE RANGO(Magnitud)
        fig, ax = plt.subplots()
        ax.plot(y, 20*np.log10(abs(I[int(len(I)/2)])),'k')
        ax.set(xlabel='Rango(m)',ylabel='Intensidad(dB)', title='Perfil de rango\n(Después de la Interpolación)')
        ax.grid()
        plt.show()
        # GRAFICA DEL PERFIL DE FASE(Eje del rango)
        fig, ax = plt.subplots()
        ax.plot(y, np.angle(I[int(len(I)/2)]),'k')
        ax.set(xlabel='Rango(m)',ylabel='Fase(rad)', title='Perfil de fase\n(Después de la Interpolación)')
        ax.grid()
        plt.show()
        # GRAFICA DEL PERFIL DE Fase(Eje de cross range)
        fig, ax = plt.subplots()
        ax.plot(x, np.angle(I.T[int(len(I[0])/2)]),'k')
        ax.set(xlabel='Rango(m)',ylabel='Fase(rad)', title='Perfil de fase - cr\n(Después de la Interpolación)')
        ax.grid()
        plt.show()

    return {'Im':I.T,'x':x,'y':y}

def plot_image(data2):
    """ Grafica la magnitud de la imagen"""
    # a) Definicion y lectura de parametros
    Im = data2['Im'].copy()
    x = data2['x']
    y = data2['y']

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    # R_max = data2['R_max']
    Nx = len(Im[0])
    Ny = len(Im)

    if show:
        # b) Truncacion de la imagen(Eje Y) _ usarlo solo en las pruebas
        # Función para hallar ni a partir de yi
        def set_y(yi):
            return (Ny-1)*(yi-y[0])/(y[-1]-y[0]) # float

        y1,y2 = y.min(),y.max() #max(0,y.min()),min(R_max,y.max()) # Nuevos valores limites de y=[y1,y2]
        n1 = int(np.floor(round(set_y(y1),5))) # n1 = f(y1)
        n2 = int(np.ceil(round(set_y(y2),5))) # n2 = f(y2)
        # Se uso primero round, floor y ceil, para asegurarse de q n1, n2 no salga de los limites permitidos
        Im_n = Im.T[n1:n2,]

        # c) Truncacion de la imagen(Eje X)
        # Función para hallar ni a partir de xi
        def set_x(xi):
            return (Nx-1)*(xi-x[0])/(x[-1]-x[0]) # float

        x1,x2 = x.min(),x.max() #max(0,y.min()),min(R_max,y.max()) # Nuevos valores limites de y=[y1,y2]
        n3 = int(np.floor(round(set_x(x1),5))) # n1 = f(y1)
        n4 = int(np.ceil(round(set_x(x2),5))) # n2 = f(y2)
        # Se uso primero round, floor y ceil, para asegurarse de q n1, n2 no salga de los limites permitidos

    # c) Grafica final(magnitud)
    cmap="plasma"
    vmin = -100 #dB
    vmax = -20
    dF.plotImage(Im,cmap=cmap,xlabel_name='Azimut(m)',ylabel_name='Rango(m)', title_name='Resultado Algoritmo Range Migration',
                 x_min=x[0]-dx/2, x_max=x[-1]+dx/2, y_min=y[0]-dy/2, y_max=y[-1]+dy/2,unit_bar='(dB)',log=True,vmin=vmin,
                 vmax=vmax)

    return 'Ok!'

def main():
    plt.close('all') # Cerrar todas las figuras previas

    start_time = timeit.default_timer()
    datos = get_SAR_data() # Obtiene el historico de fase
    print("Tiempo de simulación(RMA): ",timeit.default_timer() - start_time,"s")

    start_time = timeit.default_timer()
    d_p = RMA_Algorithm(datos) # Implementa el algoritmo RMA
    print("Tiempo del procesamiento(RMA): ",timeit.default_timer() - start_time,"s")

    d_p['R_max'] = datos['R_max']
    plot_image(d_p) # Grafica de la magnitud

if __name__ == '__main__':
  main()
