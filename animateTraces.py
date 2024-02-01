import numpy as np
import manim as ma


class TraceAnim(ma.Animation):
    def __init__(self, path, points, maxlen, **kwargs):
        super().__init__(path, **kwargs)
        self.points = points
        self.maxlen = maxlen

    def interpolate_mobject(self, alpha):
        path = self.mobject.copy()
        if alpha == 0:
            path.set_points_as_corners([self.points[0], self.points[0]])
        else:
            trace_end = max(2, 1 + int((self.points.shape[0] - 1) * alpha))
            origidx = max(0, trace_end - self.maxlen)
            path.set_points_as_corners([*self.points[origidx:trace_end]])
        self.mobject.become(path)


class Count(ma.Animation):
    def __init__(
        self, number: ma.Integer, start: float, end: float, center_pos, **kwargs
    ) -> None:
        super().__init__(number, **kwargs)
        self.start = start
        self.end = end
        self.center_pos = center_pos

    def interpolate_mobject(self, alpha: float) -> None:
        value = int(self.start + (alpha * (self.end - self.start)))
        self.mobject.set_value(value)
        self.mobject.move_to(self.center_pos)


def strided_concat(sig):
    npts, h, ndim = sig.shape
    buf = np.zeros((npts, ndim))
    for i in range(0, npts, h):
        buf[i : i + h] = sig[i]
    rem = npts % h
    if rem > 0:
        buf[-rem:] = sig[-1, :rem]
    return buf


def incremental_concat(sig):
    npts, h, ndim = sig.shape
    buf = np.zeros((npts, ndim))
    for i in range(npts):
        buf[i] = sig[i, 0]
    return buf


def averaged_concat(sig):
    npts, h, ndim = sig.shape
    buf = np.zeros((npts, ndim))
    norm = np.zeros(npts)
    norm[:h] = np.arange(1, h + 1)
    norm[-h:] = np.arange(h, 0, -1)
    norm[h:-h] = h
    for i in range(npts - h + 1):
        buf[i : i + h] += sig[i]
    return buf / norm[:, None]


def normalize(yt, yh, include):
    ax = (0, 1)
    if include in ["both", "y_true"]:
        meanv = yt.mean(axis=ax)
        yt -= meanv
        yh -= meanv
    else:
        meanv = yh.mean(axis=ax)
        yh -= meanv

    if include == "both":
        maxt = np.abs(yt).max(axis=ax)
        maxh = np.abs(yh).max(axis=ax)
        if maxt.sum() > maxh.sum():
            maxv = maxt
        else:
            maxv = maxh
        yt /= maxv
        yh /= maxv
    elif include == "y_true":
        maxv = np.abs(yt).max(axis=ax)
        yt /= maxv
    else:
        maxv = np.abs(yh).max(axis=ax)
        yh /= maxv
    return yt, yh


class PathTrace(ma.Scene):
    def __init__(
        self,
        npyfn,
        start,
        npts,
        include="both",
        scale=5,
        path_len=30,
        run_time=10,
        concat_methods=["stride"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.npyfn = npyfn
        self.start = start
        self.npts = npts
        self.include = include
        self.scale = scale
        self.path_len = path_len
        self.run_time = run_time
        self.concat_methods = []
        if "avg" in concat_methods:
            self.concat_methods.append(averaged_concat)
        if "stride" in concat_methods:
            self.concat_methods.append(strided_concat)
        if "inc" in concat_methods:
            self.concat_methods.append(incremental_concat)

    def construct(self):
        d = np.load(self.npyfn, allow_pickle=True).item()

        yt = d["y_true"][0, self.start : self.start + self.npts]
        yh = d["y_hat"][0, self.start : self.start + self.npts]

        yt, yh = normalize(yt, yh, self.include)

        def build_path(y, clr, concat=averaged_concat):
            buf = concat(y)
            buf *= self.scale
            path = ma.VMobject(stroke_width=1, stroke_color=clr)
            path.set_points_as_corners([buf[0], buf[0]])
            return path, buf

        objs = []
        anims = []
        if self.include in ["both", "y_true"]:
            tpath, tbuf = build_path(yt, "blue", concat=strided_concat)
            objs.append(tpath)
            anims.append(TraceAnim(tpath, tbuf, self.path_len))
        if self.include in ["both", "y_hat"]:
            clrs = ["orange", "green", "yellow"]
            for i, concat in enumerate(self.concat_methods):
                hpath, hbuf = build_path(yh, clrs[i], concat=concat)
                objs.append(hpath)
                anims.append(TraceAnim(hpath, hbuf, self.path_len))

        relCoord = 3.5 * ma.UP
        relPos = ma.Integer().set_color(ma.WHITE).scale(0.5).move_to(relCoord)
        objs.append(relPos)
        anims.append(Count(relPos, 0, self.npts, relCoord))

        absCoord = 3 * ma.UP
        absPos = ma.Integer().set_color(ma.WHITE).scale(1).move_to(absCoord)
        objs.append(absPos)
        anims.append(Count(absPos, self.start, self.start + self.npts, absCoord))

        self.add(*objs)
        self.play(
            *anims,
            run_time=self.run_time,
            rate_func=ma.linear,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("npyfn", help="npy file")
    parser.add_argument(
        "--include",
        choices=["both", "y_true", "y_hat"],
        default="both",
        help="which paths to include",
    )
    parser.add_argument("--start", help="start point", default=0, type=int)
    parser.add_argument(
        "--npts", help="number of points to render", default=100, type=int
    )
    parser.add_argument(
        "-q",
        help="quality",
        choices=["low_quality", "medium_quality", "high_quality"],
        default="high_quality",
    )
    parser.add_argument("--seconds", help="length in seconds", default=10, type=int)
    parser.add_argument("--scale", help="scale factor", default=4, type=int)
    parser.add_argument(
        "--path_len", help="length of path trace to display", default=30, type=int
    )
    parser.add_argument(
        "--concat",
        nargs="+",
        choices=["avg", "stride", "inc"],
        default=["stride"],
        help="concatenation methods to render",
    )
    args = parser.parse_args()
    with ma.tempconfig({"quality": args.q, "preview": True}):
        scene = PathTrace(
            args.npyfn,
            args.start,
            args.npts,
            include=args.include,
            scale=args.scale,
            path_len=args.path_len,
            run_time=args.seconds,
            concat_methods=args.concat,
        )
        scene.render()
