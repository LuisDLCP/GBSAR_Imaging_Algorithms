"""
-------------------------------------------------
      LIBRERÍA CREADA PARA LA GENERACIÓN DEL
            HISTÓRICO DE FASE TEÓRICO
-------------------------------------------------
@author: LUIS
"""
import numpy as np
import sarPrm as sp
import drawFigures as dF

#------------------LECTURA Y DEFINICIÓN DE PARÁMETROS--------------------
prm = sp.get_parameters()
c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
Lx,Ly,dx,dy = prm['w'],prm['h'],prm['dw'],prm['dh'] # Dimensiones de la imagen

# Funcion para mostrar el historico de fase y varios parametros
def get_SAR_data():
    """ Obtiene el histórico de fase ya sea simulado o real"""
    # Cálculo de parámetros
    It, Rt = sp.get_matrix_data1() # Coordenadas(m) y magnitud del target respectivamente
    dp=Ls/(Np-1) # Paso del riel(m)
    df=BW/(Nf-1) # Paso en frecuencia del BW
    fi=fc-BW/2 # Frecuencia inferior(GHz)
    fs=fc+BW/2 # Frecuencia superior(GHz)

    # Cálculo de las resoluciones
    rr_r=c/(2*BW) # Resolución en rango
    rr_a=c/(2*Ls*fc) # Resolución en azimuth

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

    #----------------OBTENCIÓN DEL HISTÓRICO DE FASE----------------------

    Sr_f = get_phaseH(prm,It,Rt) # Obtiene el historico de fase
    #np.save("RawData_prueba1_2",Sr_f)
    #Sr_f = np.load("RawData_prueba1.npy")

    #-----------------GRÁFICA DEL HISTÓRICO DE FASE-----------------------
    dF.plotImage(Sr_f, x_min=fi, x_max=fs, y_min=-Ls/2, y_max=Ls/2,xlabel_name='Frecuencia(GHz)',
                 ylabel_name='Posición del riel(m)', title_name='Histórico de fase',unit_bar='', origin_n='upper')

    return {'Sr_f':Sr_f, 'dp':dp, 'df':df, 'fi':fi, 'fs':fs, 'R_max':R_max, 'rr_r':rr_r, 'It':It, 'Rt':Rt}

# Funcion para obtener el histórico de fase (riel en el eje X)
def get_phaseH(prm, I_t, rt): # Parámetros, Vector de intensidades, vector de posiciones
    # Data
    c,fc,BW,Nf,Ls,Np = prm['c'],prm['fc'],prm['BW'],prm['Nf'],prm['Ls'],prm['Np']
    fi=fc-BW/2 # Frecuencia inferior(GHz)
    fs=fc+BW/2 # Frecuencia superior(GHz)

    Lista_f = np.linspace(fi, fs, Nf) #  Vector de frecuencias(GHz)
    Lista_pos = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones del riel(m)

    """It2 = np.tile(I_t,[len(Lista_pos),len(Lista_f),1])
    rt2 = rt[:,0]+rt[:,1]*1j
    ff,pp,tt=np.meshgrid(Lista_f,Lista_pos,rt2)
    d = abs(pp-tt)
    k=2*np.pi*ff/c
    Sr_f2=It2*np.exp(-2j*k*d)
    Sr_f2=np.sum(Sr_f2,axis=2)"""

    #-------------------SCATTERED SIGNAL---------------------#

    Sr_f = np.array([sum(I_t[i]*np.exp(-1j*4*np.pi*fi*distance_nk(rt[i],xi)/c)
        for i in range(len(I_t))) for xi in Lista_pos for fi in Lista_f]) # Create a vector with value for each fi y ri
    Sr_f = np.reshape(Sr_f,(Np,Nf)) # Reshape the last vector Sr_f

    return Sr_f

# Distance vector between target and riel_k position in matrix target
def distance_nk(r_n, x_k): # punto "n", punto del riel "k"
    d=((r_n[0]-x_k)**2+(r_n[1])**2)**0.5
    return d
