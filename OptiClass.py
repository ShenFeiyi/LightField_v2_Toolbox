# -*- coding:utf-8 -*-
from Utilities import saveDict

class OptiParams:
    def __init__(self, data):
        """ data structure
        data = {
            'M_MLA': ,
            'f_MLA': ,
            'pixel': ,
            'p_MLA': ,
            'z_focus': ,
            'z_focus_bias': 0 (optional)
        }
        """
        self.data = data
        self.coeffi = {'M':1, 'MM':1e-3, 'PIXEL':self.data['pixel']}

    def M_MLA(self, title='nominal'):
        # title = 'nominal' or 'real'
        if title == 'nominal':
            M = self.data['M_MLA']
        if title == 'real':
            try:
                M = self.data['M_MLA_real']
            except KeyError:
                print("Real M_MLA haven't decided! ")
                raise
        return M

    def f_MLA(self, unit='M'):
        return self.data['f_MLA'] / self.coeffi[unit.upper()]

    def pixel(self, unit='M'):
        return self.data['pixel'] / self.coeffi[unit.upper()]

    def p_MLA(self, unit='M'):
        return self.data['p_MLA'] / self.coeffi[unit.upper()]

    def z_focus(self, unit='M'):
        return self.data['z_focus'] / self.coeffi[unit.upper()]

    def z_focus_bias(self, unit='M'):
        zfb = self.data['z_focus_bias'] if 'z_focus_bias' in self.data else 0
        return zfb / self.coeffi[unit.upper()]

    def save(self, **kwargs):
        filename = kwargs['filename'] if filename in kwargs else 'OptiParams.json'
        data = {
            'M_MLA': self.data['M_MLA'],
            'f_MLA': self.data['f_MLA'],
            'pixel': self.data['pixel'],
            'p_MLA': self.data['p_MLA'],
            'z_focus': self.data['z_focus'],
            'z_focus_bias': self.data['z_focus_bias'] if 'z_focus_bias' in self.data else 0
        }
        try:
            M = self.data['M_MLA_real']
            data['M_MLA_real'] = M
        except KeyError:
            pass
        saveDict(filename, data)

    def __repr__(self):
        data = 'Light Field Optical System\n'
        data += 'Micro Lens Array focal length = ' + str(self.f_MLA('mm')) + 'mm\n'
        data += 'Nominal M_MLA = ' + str(self.M_MLA()) + '\n'
        data += 'Camera focused at ' + str(self.z_focus('mm')+self.z_focus_bias('mm')) + 'mm\n'
        return data
