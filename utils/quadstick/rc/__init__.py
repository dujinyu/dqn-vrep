'''
rc/__init__.py - Python class for polling R/C transmitters

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

import os
import pygame
import quadstick

class RC(quadstick.QuadStick):

    def __init__(self, name, switch_labels):
        '''
        Creates a new RC object.  Each subclass must implement the _convert_axis method.
        '''
        quadstick.QuadStick.__init__(self, name, switch_labels)

    def _get_pitch(self):

        return self.pitch_sign * self._get_rc_axis(self.pitch_axis)

    def _get_roll(self):
    
        return self.roll_sign * self._get_rc_axis(self.roll_axis)

    def _get_yaw(self):

        return self.yaw_sign * self._get_rc_axis(self.yaw_axis)

    def _get_throttle(self):

        return (self._get_rc_axis(self.throttle_axis) + 1) / 2

    def _get_rc_axis(self, index):
        
        return self._convert_axis(index, quadstick.QuadStick._get_axis(self, index))

    def _startup_message(self):

        return 'Please cycle throttle \nand switch to begin.'
