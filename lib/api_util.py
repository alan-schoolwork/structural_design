from __future__ import annotations

import sys
from contextlib import contextmanager
from types import FrameType, TracebackType
from typing import Any, Callable, Never

import wadler_lindig as wl

_active_apis: list[API] = []


class Unwind(Exception):
    def __init__(
        self,
        *,
        _f: Callable[[str], BaseException],
        _frame: FrameType,
    ):
        super().__init__()
        self._f = _f
        self._frame = _frame

    def _unwrap(self, msg: str) -> BaseException:
        ex = self._f(msg)
        if ex.__traceback__ is None:
            frame = self._frame
            ex.__traceback__ = TracebackType(
                None, frame, tb_lasti=frame.f_lasti, tb_lineno=frame.f_lineno
            )
        return ex


def throw(ex: Callable[[str], BaseException]) -> Never:
    __tracebackhide__ = True

    if len(_active_apis) == 0:
        raise ex("") from None

    raise _active_apis[-1]._ex_type(_f=ex, _frame=sys._getframe(1))


class API:
    def __init__(self):
        class _unique_Unwind(Unwind):
            pass

        self._ex_type = _unique_Unwind

    @property
    def ex(self) -> type[Unwind]:
        return self._ex_type

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: API, /
    ) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(id(self))

    def __enter__(self):
        _active_apis.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc: BaseException,
        exc_tb: TracebackType,
        /,
    ):
        __tracebackhide__ = True
        _active_apis.remove(self)
        if exc_type is self._ex_type:
            assert isinstance(exc, Unwind)
            raise exc._unwrap("") from None


def format_fcall(fn, *args, **kwargs):
    parts = [pp(x) for x in args] + wl.named_objs(kwargs.items())

    args_fmt = pp_join(*parts, sep=wl.comma).group()

    ans = pp_join(fn, wl.TextDoc("("), pp_nested(args_fmt), wl.TextDoc(")"))

    return wl.pformat(ans)


def trace_args[**P, R](cb: Callable[P, R]) -> Callable[P, R]:

    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        with API() as api:
            try:
                return cb(*args, **kwargs)
            except api.ex as e_wrapper:
                e = e_wrapper._unwrap(
                    "\nwhile calling:\n" + format_fcall(cb, *args, **kwargs)
                )
                raise e from None

    return inner


def pp(x: Any):
    return wl.pdoc(x)


def pp_join(*parts, sep: wl.AbstractDoc | None = None) -> wl.AbstractDoc:
    if sep is None:
        sep = wl.TextDoc("")
    return wl.join(sep, [pp(x) for x in parts])


def pp_nested(*parts, sep: Any = None) -> wl.AbstractDoc:
    return wl.ConcatDoc(
        (
            wl.ConcatDoc(
                wl.BreakDoc(""),
                pp_join(*parts, sep=sep),
            )
            .nest(2)
            .group()
        ),
        wl.BreakDoc(""),
    )


@trace_args
def testfn2(*_, **__):

    # BaseException
    # assert False
    throw(AssertionError)


def testfn():
    testfn2(1, z=1)
