"""
A keyboard controller for PyQuadSim that simulates the sticks
of a radio controller. Default configuration is in Mode 3,
with arrows and qzsd keys. Switch button is left alt.

Copyright (C) Kevin P (wazari972) 2015

Configuration
-------------

to adjust Keyboard.BINDINGS/INC_RATE/AUTO_DEC_RATE/SWITCHES:

> from quadstick.keyboard import Keyboard

> Keyboard.BINDING[pygame.locals.K_key] = +/-Keyboard.THROTTLE
> Keyboard.INC_RATE[Keyboard.THROTTLE] = ...
> Keyboard.AUTO_DEC_RATE[Keyboard.THROTTLE] = ...

> Keyboard.SWITCHES[pygame.locals.K_key] = Keyboard.SWITCH_1

The slowdown ration is to reduce the increase and decrease rate.
You may have to adjust it according to the pace of the simulator.

> Keyboard.SLOWDOWN_FACTOR = 2 # keys will be twice less sensitive
"""

from __future__ import print_function

import sys
import traceback
import pygame
import pygame.locals
from platform import platform

class GenericController(object):
    def __init__(self, name, switch_labels):
        '''
        Creates a new QuadStick object.
        '''

        # Set constants
        self.BAND = 0.2 # Must be these close to neutral for hold / autopilot

        # Init pygame
        pygame.init()
        pygame.display.init()
        
        self.screen = pygame.display.set_mode((500,280), pygame.locals.RESIZABLE)
        self.font = pygame.font.SysFont('Courier', 20)
        pygame.display.set_caption('QuadStick: ' + name)

        # Supports keyboard polling
        self.keys = []

        self.name = name

        self.platform = platform()[0:platform().find('-')]

        self.row_height = 30

        self.paused = False

        # pygame.joystick.init()
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()
        # self.joystick.get_axis(0)

        self.ready = False

        self.switch_labels = switch_labels


    def __str__(self):
        '''
        Returns a string representation of this QuadStick object
        '''
        return self.name

    def _pump(self):

        #pygame.event.pump()
        pass


    def _startup(self):
        return
        if not self.ready:

            self.message(self._startup_message())

            while True:
                self._pump()
                if self._get_throttle()  > .5:
                    break

            while True:
                self._pump()
                if self._get_throttle()  < .05 and self._get_switchval() == 0:
                    break

            self.clear()

            self.ready = True

    def poll(self):

        self._pump()

        self._startup()

        demands = self._get_pitch(), self._get_roll(), self._get_yaw(), self._get_throttle()

        switchval = self._get_switchval()

        self._show_demand(demands, 0, -1, 'Pitch')
        self._show_demand(demands, 1, -1, 'Roll')
        self._show_demand(demands, 2, +1, 'Yaw')
        self._show_demand(demands, 3, +1, 'Throttle') 
     
        self._show_switch(switchval, 0)
        self._show_switch(switchval, 1)
        self._show_switch(switchval, 2)

        pygame.display.flip()

        return demands[0], demands[1], demands[2], demands[3], switchval
 
    def running(self):
        '''
        Returns True if the QuadStick is running, False otherwise. Run can be terminated by hitting
        ESC.
        '''

        self.keys = pygame.event.get()

        for event in self.keys:
            if event.type == pygame.locals.QUIT:
                return False

            elif event.type == pygame.locals.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.locals.RESIZABLE)

            elif (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_ESCAPE):
                return False

        return True

    def clear(self):
        '''
        Clears the display.
        '''
        self.screen.fill((0,0,0))
 
    def error(self):
        '''
        Displays the most recent exception as an error message, and waits for ESC to quit.
        '''
        while True:

            self.clear()
            tb = traceback.format_exc()
            self._draw_label_in_row('ERROR', 0, color=(255,0,0))

            self._display(tb)
            print(tb, file=sys.stderr)
            pygame.display.flip()

            if not self.running():
                pygame.quit()
                sys.exit()

    def message(self, msg):
        '''
        Displays a message.
        '''
        self.clear()

        self._display(msg)

    def _display(self, msg):

        row = 1
        for line in msg.split('\n'):
            self._draw_label_in_row(line, row)
            row += 1

        pygame.display.flip()

    def _show_switch(self, switchval, index):

        x = 200
        y = 180 + index * 30
        r = 10

        # Draw a white ring around the button
        pygame.draw.circle(self.screen, (255,255,255), (x, y), r, 1)

        # Draw a white or black disk inside the ring depending on switch state
        pygame.draw.circle(self.screen, (255,255,255) if switchval == index else (0,0,0), (x, y), r-3)

        self._draw_label(self.switch_labels[index], y-10)

    def _show_demand(self, demands, index, sign, label):

        # color for no-demand baseline
        color = (0, 0, 255) 

        demand = sign * demands[index]

        if demand > 0:
            color =  (0, 255, 0)

        if demand < 0:
            color =  (255, 0, 0)

        w = 100			# width of rectangel for maximum demand
        h = 20

        x = 250
        y = 20 + index * 30

        # Erase previous 
        pygame.draw.rect(self.screen, (0,0,0), (x-w-1, y, x-w/2, h))

        # Draw a white hollow rectangle to represent the limits
        pygame.draw.rect(self.screen, (255,255,255), (x-w-1, y, x-w/2, h), 1)

        # Special handling for throttle
        if index == 3:
            x -= w
            w += w

        # Draw a colorful filled rectangle to represent the demand
        pygame.draw.rect(self.screen, color, (x, y, demand*w, h))

        # Draw a label for the axis
        self._draw_label(label, y)

    def _draw_label_in_row(self, text, row, color=(255,255,255)):

        self._draw_label(text, row*self.row_height, color)

    def _draw_label(self, text, y, color=(255,255,255)):

        surface = self.font.render(text, True, color, (0, 0, 0))
        surface.set_colorkey( (0, 0, 0) )
 
        self.screen.blit(surface, (20, y))

    def _get_axis(self, k):
        return 0

    def _get_button(self, k):
        return 0


class Keyboard(GenericController):
    SLOWDOWN_FACTOR = 1

    _, THROTTLE, YAWN, ROLL, PITCH = range(5) # value 0 not used
    
    SWITCHES = SWITCH_1,SWITCH_2,SWITCH_3 = range(3)

    INC_RATE = {
        THROTTLE: 0.01,
        YAWN: 0.01,
        PITCH: 0.01,
        ROLL: 0.01
        }
    
    AUTO_DEC_RATE = {
        THROTTLE: 0,
        YAWN: 0.005,
        PITCH: 0.005,
        ROLL: 0.005
        }
    
    BINDINGS = { 
        pygame.locals.K_UP   :  THROTTLE,
        pygame.locals.K_DOWN : -THROTTLE,
        pygame.locals.K_RIGHT:  YAWN,
        pygame.locals.K_LEFT : -YAWN,
        pygame.locals.K_w : -PITCH,
        pygame.locals.K_s :  PITCH,
        pygame.locals.K_d : -ROLL,
        pygame.locals.K_a :  ROLL
        }
    
    SWITCHES = {
        pygame.locals.K_LALT : SWITCH_1,
        pygame.locals.K_LCTRL : SWITCH_2, # not used
        pygame.locals.K_LMETA : SWITCH_3, # not used
        }
    
    def __init__(self, switch_labels):
        GenericController.__init__(self, 'Keyboard Controller', switch_labels)
        self.power = [None, 0, 0, 0, 0] # value 0 not used
        self.keysdown = {}
        # Support alt/pos-hold through repeated button clicsk
        self.switch_value = 0

    def _get_switchval(self):
        return self.switch_value

    def _get_axis(self, axis_index_asked):
        keys = pygame.event.get()
        
        # collect keys up and down
        for event in keys:
            is_switch = False
            try:
                key_binding = Keyboard.BINDINGS[event.key]
            except:
                try:
                    key_binding = Keyboard.SWITCHES[event.key]
                    is_switch = True
                except:
                    continue # not an interesting key
            
            if event.type == pygame.locals.KEYDOWN:
                self.keysdown[event.key] = key_binding, is_switch
            elif event.type == pygame.locals.KEYUP:
                del self.keysdown[event.key]
            else:
                assert "Key not up nor down ??" # should not come here
                continue

        # increase keys down
        for key, (axis_index, is_switch) in self.keysdown.items():
            if is_switch:
                if axis_index is None: # switch already hit
                    continue

                self.switch_value += 1
                self.switch_value %= 3

                self.keysdown[key] = (None, True) # do not repeat switch
                continue
            
            direction = 1 if axis_index > 0 else -1
            axis_index = abs(axis_index)
            
            axis_increase = Keyboard.INC_RATE[axis_index] * direction
            
            if axis_index != Keyboard.THROTTLE:
                power = self.power[axis_index]

                #reset stick if push on the other way
                if axis_increase > 0 and power < 0:
                    self.power[axis_index] = 0
                if axis_increase < 0 and power > 0:
                    self.power[axis_index] = 0
            
            axis_increase *= Keyboard.SLOWDOWN_FACTOR
                
            self.power[axis_index] += axis_increase
            
        # decrease keys up and check boundaries
        for dec_axis_index in (Keyboard.YAWN, Keyboard.PITCH, Keyboard.ROLL):
            power = self.power[dec_axis_index]
            dec = Keyboard.AUTO_DEC_RATE[dec_axis_index]
            if power < 0:
                dec *= -1
                
            dec *= Keyboard.SLOWDOWN_FACTOR
                
            self.power[dec_axis_index] -= dec

            # check boundaries -1 < ... < 1
            self.power[dec_axis_index] = max(self.power[dec_axis_index], -1)
            self.power[dec_axis_index] = min(self.power[dec_axis_index], 1)
            
        # check throttle boundaries 0 < ... < 1
        self.power[Keyboard.THROTTLE] = max(self.power[Keyboard.THROTTLE], 0)
        self.power[Keyboard.THROTTLE] = min(self.power[Keyboard.THROTTLE], 1)
        
        return self.power[axis_index_asked]
    
    def _startup_message(self):

        return 'Please cycle throttle to begin.'

    def _get_pitch(self):
    
        return self._get_axis(Keyboard.PITCH)

    def _get_roll(self):
    
        return self._get_axis(Keyboard.ROLL)

    def _get_yaw(self):

        return self._get_axis(Keyboard.YAWN)

    def _get_throttle(self):
        return self._get_axis(Keyboard.THROTTLE)

