import pyglet
from pyglet.window import key
import numpy as np
import ising

RED = 1<<7
BLUE = 1<<23

N=128
global t, imsize
t = 0
model = ising.Ising2d(T=600,dT=-0.1,N=128)

window = pyglet.window.Window(resizable=True)
label = pyglet.text.Label('stuff',
                          font_name='Arial',
                          font_size=14,
                          multiline=True,
                          width=1,
                          anchor_x='left', anchor_y='top')

img = np.empty((N,N), dtype='uint32')
image = pyglet.image.ImageData(N, N, 'RGB', img.tostring())

@window.event
def on_draw():
    window.clear()
    image.blit(0, 0, width=imsize, height=imsize)
    label.draw()

@window.event
def on_resize(width, height):
    global imsize
    imsize = min(window.width, window.height)
    label.x = imsize + 16
    label.y = height

@window.event
def on_key_release(symbol, modifiers):
    def dTChange(delta):
        if modifiers & key.MOD_SHIFT:
            model.dT += 10 * delta
        else:        
            model.dT += delta

    def dTZero(*args):
        model.dT = 0

    keyfuncs = {
        key.PLUS:         [dTChange, 0.1],
        key.NUM_ADD :     [dTChange, 0.1],
        key.MINUS:        [dTChange, -0.1],
        key.NUM_SUBTRACT: [dTChange, -0.1],
        key.SPACE:        [dTZero, None],
    }
    
    if keyfuncs.has_key(symbol):
        func, arg = keyfuncs[symbol]
        func(arg)
        
    #func, args = keyfuncs[symbol]
    #if func:
    #    func(args)



def update(dt):
    global t
    t += dt
    model.iterate()
    label.text = 'T=%sK\ndT=%.5s\nt=%.3s' % (model.T, model.dT, t)
    img[model.lattice == 1] = RED
    img[model.lattice == -1] = BLUE
    image.set_data('RGBA', N*4, img.data.__str__())

pyglet.clock.schedule_interval(update, 0.001)

pyglet.app.run()