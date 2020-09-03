import numpy as np
dict = {'fname': r'C:\Users\james\Documents\XSPL\Outreach\Project\Python\Data\BSE_AlRod11.tif'}

def func(x, a, sigma):
    return a*np.exp(-(x-219)**2/(2*sigma**2))

