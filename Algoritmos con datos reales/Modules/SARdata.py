"""
-------------------------------------------------
      LIBRERÍA CREADA PARA LA GENERACIÓN DEL
            HISTÓRICO DE FASE TEÓRICO
-------------------------------------------------
@author: LUIS
"""
import numpy as np

# Funcion para obtener el histórico de fase (riel en el eje X)
def get_phaseH(prm, I_t, rt): # Parámetros, Vector de intensidades, vector de posiciones
    # Data
    c,fc,BW,Nf,Ls,Np = prm['c'],prm['fc'],prm['BW'],prm['Nf'],prm['Ls'],prm['Np']
    fi=fc-BW/2 # Frecuencia inferior(GHz)
    fs=fc+BW/2 # Frecuencia superior(GHz)

    Lista_f = np.linspace(fi, fs, Nf) #  Vector de frecuencias(GHz)
    Lista_pos = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones del riel(m)

    #-------------------SCATTERED SIGNAL---------------------#
    Sr_f = np.array([sum(I_t[i]*np.exp(-1j*4*np.pi*fi*distance_nk(rt[i],xi)/c)
        for i in range(len(I_t))) for xi in Lista_pos for fi in Lista_f]) # Create a vector with value for each fi y ri
    Sr_f = np.reshape(Sr_f,(Np,Nf)) # Reshape the last vector Sr_f

    return Sr_f

# Distance vector between target and riel_k position in matrix target
def distance_nk(r_n, x_k): # punto "n", punto del riel "k"
    d=((r_n[0]-x_k)**2+(r_n[1])**2)**0.5
    return d
