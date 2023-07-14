class HsvFilter:

    def __init__(self, hmin=None, smin=None, vmin=None, hmax=None, smax=None, vmax=None,
                       sadd=None, ssub=None, vadd=None, vsub=None):
        self.hmin = hmin
        self.smin = smin
        self.vmin = vmin
        self.hmax = hmax
        self.smax = smax
        self.vmax = vmax
        self.sadd = sadd
        self.ssub = ssub
        self.vadd = vadd
        self.vsub = vsub