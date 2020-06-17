import functools
import types
from collections import OrderedDict

from k_means_constrained.sklearn_import.externals.funcsigs import _NonUserDefinedCallables, _get_user_defined_method, \
    _POSITIONAL_ONLY, _VAR_POSITIONAL, _VAR_KEYWORD, Signature


def signature(obj):
    '''Get a signature object for the passed callable.'''

    if not callable(obj):
        raise TypeError('{0!r} is not a callable object'.format(obj))

    if isinstance(obj, types.MethodType):
        sig = signature(obj.__func__)
        if obj.__self__ is None:
            # Unbound method: the first parameter becomes positional-only
            if sig.parameters:
                first = sig.parameters.values()[0].replace(
                    kind=_POSITIONAL_ONLY)
                return sig.replace(
                    parameters=(first,) + tuple(sig.parameters.values())[1:])
            else:
                return sig
        else:
            # In this case we skip the first parameter of the underlying
            # function (usually `self` or `cls`).
            return sig.replace(parameters=tuple(sig.parameters.values())[1:])

    try:
        sig = obj.__signature__
    except AttributeError:
        pass
    else:
        if sig is not None:
            return sig

    try:
        # Was this function wrapped by a decorator?
        wrapped = obj.__wrapped__
    except AttributeError:
        pass
    else:
        return signature(wrapped)

    if isinstance(obj, types.FunctionType):
        return Signature.from_function(obj)

    if isinstance(obj, functools.partial):
        sig = signature(obj.func)

        new_params = OrderedDict(sig.parameters.items())

        partial_args = obj.args or ()
        partial_keywords = obj.keywords or {}
        try:
            ba = sig.bind_partial(*partial_args, **partial_keywords)
        except TypeError as ex:
            msg = 'partial object {0!r} has incorrect arguments'.format(obj)
            raise ValueError(msg)

        for arg_name, arg_value in ba.arguments.items():
            param = new_params[arg_name]
            if arg_name in partial_keywords:
                # We set a new default value, because the following code
                # is correct:
                #
                #   >>> def foo(a): print(a)
                #   >>> print(partial(partial(foo, a=10), a=20)())
                #   20
                #   >>> print(partial(partial(foo, a=10), a=20)(a=30))
                #   30
                #
                # So, with 'partial' objects, passing a keyword argument is
                # like setting a new default value for the corresponding
                # parameter
                #
                # We also mark this parameter with '_partial_kwarg'
                # flag.  Later, in '_bind', the 'default' value of this
                # parameter will be added to 'kwargs', to simulate
                # the 'functools.partial' real call.
                new_params[arg_name] = param.replace(default=arg_value,
                                                     _partial_kwarg=True)

            elif (param.kind not in (_VAR_KEYWORD, _VAR_POSITIONAL) and
                            not param._partial_kwarg):
                new_params.pop(arg_name)

        return sig.replace(parameters=new_params.values())

    sig = None
    if isinstance(obj, type):
        # obj is a class or a metaclass

        # First, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = _get_user_defined_method(type(obj), '__call__')
        if call is not None:
            sig = signature(call)
        else:
            # Now we check if the 'obj' class has a '__new__' method
            new = _get_user_defined_method(obj, '__new__')
            if new is not None:
                sig = signature(new)
            else:
                # Finally, we should have at least __init__ implemented
                init = _get_user_defined_method(obj, '__init__')
                if init is not None:
                    sig = signature(init)
    elif not isinstance(obj, _NonUserDefinedCallables):
        # An object with __call__
        # We also check that the 'obj' is not an instance of
        # _WrapperDescriptor or _MethodWrapper to avoid
        # infinite recursion (and even potential segfault)
        call = _get_user_defined_method(type(obj), '__call__', 'im_func')
        if call is not None:
            sig = signature(call)

    if sig is not None:
        # For classes and objects we skip the first parameter of their
        # __call__, __new__, or __init__ methods
        return sig.replace(parameters=tuple(sig.parameters.values())[1:])

    if isinstance(obj, types.BuiltinFunctionType):
        # Raise a nicer error message for builtins
        msg = 'no signature found for builtin function {0!r}'.format(obj)
        raise ValueError(msg)

    raise ValueError('callable {0!r} is not supported by signature'.format(obj))