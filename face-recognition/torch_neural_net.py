import binascii
from subprocess import Popen, PIPE
import os
import sys

import numpy as np
import cv2

class TorchNeuralNet:
    def __init__(self, model, imgDim, cuda=False):
        assert model is not None
        assert imgDim is not None
        assert cuda is not None

        self.cmd = ['/usr/bin/env', 
                    'th', '/home/dev/Work/REPOS/pfm-videoanalysis/face_recognition/openface_server.lua',
                    '-model', model, 
                    '-imgDim', str(imgDim)]
        if cuda:
            self.cmd.append('-cuda')
        
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
        
        # Check whether the child process has successfully created or not
        rc = self.p.poll()
        if rc is not None and rc != 0:
            raise Exception("Exception: Neural Network process died !")

    def forwardPath(self, imgPath):
        assert imgPath is not None

        rc = self.p.poll()
        if rc is not None and rc != 0:
            raise Exception("Exception: Neural Network process died !")

        self.p.stdin.write(imgPath + "\n")
        output = self.p.stdout.readline()
        try:
            rep = [float(x) for x in output.strip().split(',')]
            rep = np.array(rep)
            return rep
        except Exception as e:
            self.p.kill()
            stdout, stderr = self.p.communicate()
            print("Error getting result from Torch subprocess !"+ str(e))
            sys.exit(-1)

    def forward(self, rgbImg):
        assert rgbImg is not None
        
        t = '/tmp/openface-torchwrap-{}.png'.format(binascii.b2a_hex(os.urandom(8)))
        bgrImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(t, bgrImg)
        rep = self.forwardPath(t)
        os.remove(t)
        return rep
