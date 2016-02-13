'''
frsky.py - Support for FrSky R/C transmitters

    Copyright (C) 2014 Simon D. Levy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

'''

from quadstick.rc import RC

class Taranis(RC):
    '''
    Class for FrSky Taranis transmitter used with mini-USB cable.  
    You should set up channel mixing such that Channel 5 maps to Switch A and Channel 6 to Switch B.
    '''
 
    def __init__(self, switch_labels):
        '''
        Creates a new Taranis object.
        '''

        RC.__init__(self, 'Taranis', switch_labels)

        # Default to Linux 
        self.pitch_axis     = 2
        self.roll_axis      = 1
        self.yaw_axis       = 3
        self.throttle_axis  = 0
        self.switch_axis    = 5

        if self.platform == 'Windows':
            self.yaw_axis    = 5
            self.switch_axis = 3
            
        elif self.platform == 'Darwin':
            self.pitch_axis    = 0
            self.roll_axis     = 3
            self.yaw_axis      = 1
            self.throttle_axis = 2
            self.switch_axis   = 4

        self.pitch_sign = +1
        self.roll_sign  = -1
        self.yaw_sign   = +1

    def _convert_axis(self, index, value):

        return value

    def _get_switchval(self):

        switch = RC._get_axis(self, self.switch_axis)

        return 0 if switch < -.5 else (1 if switch < +.5 else 2)
