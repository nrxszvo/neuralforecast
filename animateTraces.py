import numpy as np
import manim as ma


class TraceAnim(ma.Animation):
    def __init__(self, path, points, maxlen, **kwargs):
        super().__init__(path, **kwargs)
        self.points = points
        self.maxlen = maxlen

    def interpolate_mobject(self, alpha):
        path = self.mobject.copy()
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


class PathTrace(ma.Scene):
    def __init__(self, npyfn, start, npts, scale=5, path_len=30, run_time=10, **kwargs):
        super().__init__(**kwargs)
        self.npyfn = npyfn
        self.start = start
        self.npts = npts
        self.scale = scale
        self.path_len = path_len
        self.run_time = run_time

    def construct(self):
        d = np.load(self.npyfn, allow_pickle=True).item()

        def build_path(k, clr):
            maxv = max(d["y_true"].max(), d["y_hat"].max())
            sig = d[k][0]
            sig = self.scale * sig / maxv
            buf = strided_concat(sig[self.start : self.start + self.npts])
            path = ma.VMobject(stroke_width=1, stroke_color=clr)
            path.set_points_as_corners([buf[0], buf[0]])
            return path, buf

        tpath, tbuf = build_path("y_true", "blue")
        hpath, hbuf = build_path("y_hat", "orange")

        relCoord = 3.5 * ma.UP
        relPos = ma.Integer().set_color(ma.WHITE).scale(0.5).move_to(relCoord)
        absCoord = 3 * ma.UP
        absPos = ma.Integer().set_color(ma.WHITE).scale(1).move_to(absCoord)

        self.add(tpath, hpath, absPos, relPos)
        self.play(
            TraceAnim(tpath, tbuf, self.path_len),
            TraceAnim(hpath, hbuf, self.path_len),
            Count(absPos, self.start, self.start + self.npts, absCoord),
            Count(relPos, 0, self.npts, relCoord),
            run_time=self.run_time,
            rate_func=ma.linear,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("npyfn", help="npy file")
    parser.add_argument("--start", help="start point", default=0, type=int)
    parser.add_argument(
        "--npts", help="number of points to render", default=200, type=int
    )
    parser.add_argument(
        "-q",
        help="quality",
        choices=["low_quality", "medium_quality", "high_quality"],
        default="high_quality",
    )
    parser.add_argument("--seconds", help="length in seconds", default=10, type=int)
    parser.add_argument("--scale", help="scale factor", default=5, type=int)
    parser.add_argument(
        "--path_len", help="length of path trace to display", default=30, type=int
    )
    args = parser.parse_args()

    with ma.tempconfig({"quality": args.q, "preview": True}):
        scene = PathTrace(
            args.npyfn,
            args.start,
            args.npts,
            scale=args.scale,
            path_len=args.path_len,
            run_time=args.seconds,
        )
        scene.render()
