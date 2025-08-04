import importlib, inspect, json, pkgutil, types
import pybis

PKG_NAME = "pybis"

def walk_package(pkg) -> dict[str, dict]:
    """
    Return {fully-qualified-name: {signature, docstring}} for every
    function, coroutine, method and classmethod defined inside *pkg*.
    """
    api = {}

    # traverse pybis and all nested packages
    for *_ , modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            # skip optional dependencies / network-bound modules
            continue

        # top-level functions
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__.startswith(PKG_NAME):
                api[f"{obj.__module__}.{obj.__qualname__}"] = {
                    "signature": str(inspect.signature(obj)),
                    "doc": inspect.getdoc(obj) or "",
                }

        # classes → methods / classmethods / staticmethods
        for _name, cls in inspect.getmembers(mod, inspect.isclass):
            if not cls.__module__.startswith(PKG_NAME):
                continue
            for attr_name, attr in inspect.getmembers(cls):
                # unwrap classmethod / staticmethod objects
                if isinstance(attr, (classmethod, staticmethod)):
                    attr = attr.__func__
                if isinstance(attr, (types.FunctionType, types.MethodType)):
                    api[f"{cls.__module__}.{cls.__qualname__}.{attr.__name__}"] = {
                        "signature": str(inspect.signature(attr)),
                        "doc": inspect.getdoc(attr) or "",
                    }
    return api


api_map = walk_package(pybis)

with open("pybis_api.json", "w") as fp:
    json.dump(api_map, fp, indent=2)