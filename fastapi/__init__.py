from __future__ import annotations

import asyncio
import inspect
import json
import re
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Pattern, Tuple, Type, Union, get_type_hints
from urllib.parse import parse_qs, urlsplit

from .responses import JSONResponse, Response

__all__ = [
    "FastAPI",
    "HTTPException",
    "Request",
    "Header",
    "Cookie",
    "Depends",
    "Form",
]

__version__ = "0.1.0-stub"


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None, headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}


class HeaderInfo:
    __slots__ = ("default", "alias")

    def __init__(self, default: Any = None, alias: Optional[str] = None) -> None:
        self.default = default
        self.alias = alias


class CookieInfo:
    __slots__ = ("default", "alias")

    def __init__(self, default: Any = None, alias: Optional[str] = None) -> None:
        self.default = default
        self.alias = alias


def Header(default: Any = None, *, alias: Optional[str] = None) -> HeaderInfo:
    return HeaderInfo(default=default, alias=alias)


def Cookie(default: Any = None, *, alias: Optional[str] = None) -> CookieInfo:
    return CookieInfo(default=default, alias=alias)


class FormInfo:
    __slots__ = ("default", "alias")

    def __init__(self, default: Any = inspect._empty, alias: Optional[str] = None) -> None:
        self.default = default
        self.alias = alias


def Form(default: Any = inspect._empty, *, alias: Optional[str] = None) -> FormInfo:
    return FormInfo(default=default, alias=alias)


class DependencyInfo:
    __slots__ = ("dependency", "use_cache")

    def __init__(self, dependency: Optional[Callable[..., Any]], *, use_cache: bool = True) -> None:
        self.dependency = dependency
        self.use_cache = use_cache


def Depends(dependency: Optional[Callable[..., Any]] = None, *, use_cache: bool = True) -> DependencyInfo:
    return DependencyInfo(dependency, use_cache=use_cache)


class Headers:
    def __init__(self, items: Iterable[Tuple[str, str]]) -> None:
        store: Dict[str, str] = {}
        for key, value in items:
            store[key.lower()] = value
        self._store = store

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._store.get(key.lower(), default)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._store

    def __getitem__(self, key: str) -> str:
        return self._store[key.lower()]

    def items(self) -> Iterable[Tuple[str, str]]:
        return self._store.items()


class RequestURL(SimpleNamespace):
    def __str__(self) -> str:  # pragma: no cover - convenience method
        query = getattr(self, "query", "")
        if query:
            query = f"?{query}"
        return f"{self.scheme}://{self.netloc}{self.path}{query}"


class Request:
    def __init__(
        self,
        method: str,
        target: str,
        headers: Iterable[Tuple[str, str]],
        body: bytes,
        client: Tuple[str, int],
        scheme: str = "http",
    ) -> None:
        parts = urlsplit(target)
        self.method = method.upper()
        self.path = parts.path or "/"
        self.query = parts.query
        self.query_params = {k: v[0] if isinstance(v, list) else v for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
        self.headers = Headers(headers)
        self._body = body
        self._json: Any = ...
        self._json_loaded = False
        self._form: Optional[Dict[str, Any]] = None
        self._form_loaded = False
        host = self.headers.get("host", "localhost")
        self.url = RequestURL(scheme=scheme, netloc=host, path=self.path, query=self.query)
        self.client = client
        self.cookies = self._parse_cookies(self.headers.get("cookie"))
        self.state = SimpleNamespace()

    @staticmethod
    def _parse_cookies(header_value: Optional[str]) -> Dict[str, str]:
        if not header_value:
            return {}
        cookies: Dict[str, str] = {}
        for chunk in header_value.split(";"):
            if "=" not in chunk:
                continue
            name, value = chunk.split("=", 1)
            cookies[name.strip()] = value.strip()
        return cookies

    async def body(self) -> bytes:
        return self._body

    async def json(self) -> Any:
        if not self._json_loaded:
            if not self._body:
                self._json = {}
            else:
                self._json = json.loads(self._body.decode("utf-8"))
            self._json_loaded = True
        return self._json

    async def text(self) -> str:
        return self._body.decode("utf-8")

    async def form(self) -> Dict[str, Any]:
        if not self._form_loaded:
            if not self._body:
                data: Dict[str, Any] = {}
            else:
                parsed = parse_qs(self._body.decode("utf-8"), keep_blank_values=True)
                data = {key: values[-1] if isinstance(values, list) else values for key, values in parsed.items()}
            self._form = data
            self._form_loaded = True
        return self._form or {}


class _Route:
    __slots__ = ("path", "methods", "endpoint", "response_class", "regex", "param_names")

    def __init__(self, path: str, methods: Tuple[str, ...], endpoint: Callable[..., Any], response_class: Optional[type]) -> None:
        self.path = path
        self.methods = tuple(m.upper() for m in methods)
        self.endpoint = endpoint
        self.response_class = response_class
        self.regex, self.param_names = self._compile_path(path)

    @staticmethod
    def _compile_path(path: str) -> Tuple[Pattern[str], List[str]]:
        param_regex = re.compile(r"{([^}/]+)}")
        idx = 0
        regex = "^"
        param_names: List[str] = []
        for match in param_regex.finditer(path):
            regex += re.escape(path[idx:match.start()])
            name = match.group(1)
            param_names.append(name)
            regex += r"(?P<%s>[^/]+)" % name
            idx = match.end()
        regex += re.escape(path[idx:])
        regex += "$"
        return re.compile(regex), param_names

    def matches(self, method: str, path: str) -> Optional[re.Match[str]]:
        if method.upper() not in self.methods:
            return None
        return self.regex.match(path)


class FastAPI:
    def __init__(self, title: str = "FastAPI", version: str = "0.1.0", lifespan: Optional[Callable[["FastAPI"], Awaitable[Any]]] = None) -> None:
        self.title = title
        self.version = version
        self._routes: List[_Route] = []
        self._mounts: List[Tuple[str, Any]] = []
        self._lifespan_factory = lifespan
        self._lifespan_cm = lifespan(self) if lifespan else None
        self._http_middleware: List[Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]] = []
        self._exception_handlers: List[Tuple[Type[BaseException], Callable[[Request, BaseException], Any]]] = []

    def mount(self, path: str, app: Any, name: Optional[str] = None) -> None:
        prefix = path.rstrip("/") or "/"
        self._mounts.append((prefix, app))

    def add_api_route(self, path: str, endpoint: Callable[..., Any], methods: Iterable[str], response_class: Optional[type] = None) -> None:
        route = _Route(path, tuple(methods), endpoint, response_class)
        self._routes.append(route)

    def get(self, path: str, *, response_class: Optional[type] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, ("GET",), response_class=response_class)
            return func

        return decorator

    def post(self, path: str, *, response_class: Optional[type] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, ("POST",), response_class=response_class)
            return func

        return decorator

    def delete(self, path: str, *, response_class: Optional[type] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, ("DELETE",), response_class=response_class)
            return func

        return decorator

    def add_middleware(self, middleware_cls: Type[Any], **options: Any) -> None:
        instance = middleware_cls(self, **options)
        dispatch = getattr(instance, "dispatch", None)
        if dispatch is None:
            dispatch = getattr(instance, "__call__", None)
        if dispatch is None:
            raise TypeError("Middleware must define a 'dispatch' or '__call__' method")

        async def middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]], _dispatch=dispatch) -> Response:
            result = _dispatch(request, call_next)
            if inspect.isawaitable(result):
                result = await result
            return result

        self._http_middleware.append(middleware)

    def middleware(self, type_: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if type_ != "http":
            raise ValueError("Only 'http' middleware is supported in the stub implementation")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            async def middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]], _func=func) -> Response:
                result = _func(request, call_next)
                if inspect.isawaitable(result):
                    result = await result
                return result

            self._http_middleware.append(middleware)
            return func

        return decorator

    def exception_handler(self, exc_class: Type[BaseException]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._exception_handlers.append((exc_class, func))
            return func

        return decorator

    async def _startup(self) -> None:
        if self._lifespan_cm is not None:
            await self._lifespan_cm.__aenter__()

    async def _shutdown(self) -> None:
        if self._lifespan_cm is not None:
            await self._lifespan_cm.__aexit__(None, None, None)

    async def _handle(self, request: Request) -> Response:
        async def endpoint(req: Request) -> Response:
            return await self._dispatch_request(req)

        handler = endpoint
        for middleware in reversed(self._http_middleware):
            handler = self._wrap_http_middleware(middleware, handler)

        try:
            return await handler(request)
        except Exception as exc:  # pragma: no cover - defensive
            return await self._handle_exception(request, exc)

    async def _dispatch_request(self, request: Request) -> Response:
        path = request.path
        for prefix, app in self._mounts:
            if prefix == "/":
                continue
            if path == prefix or path.startswith(prefix + "/"):
                sub_path = path[len(prefix) :] or "/"
                if hasattr(app, "get_response"):
                    result = app.get_response(sub_path, request)
                    if inspect.isawaitable(result):
                        result = await result
                    if not isinstance(result, Response):
                        raise HTTPException(500, "Static mount did not return a Response")
                    return result
        for route in self._routes:
            match = route.matches(request.method, path)
            if not match:
                continue
            path_params = {k: v for k, v in match.groupdict().items()}
            try:
                result = await self._call_endpoint(route, request, path_params)
            except Exception as exc:  # pragma: no cover - unexpected error path
                return await self._handle_exception(request, exc)
            return self._to_response(result, route.response_class)
        raise HTTPException(404, "Not Found")

    async def _handle_exception(self, request: Request, exc: Exception) -> Response:
        for exc_type, handler in reversed(self._exception_handlers):
            if isinstance(exc, exc_type):
                result = handler(request, exc)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, Response):
                    return result
                return self._to_response(result, None)
        if isinstance(exc, HTTPException):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
        return JSONResponse({"detail": f"Internal Server Error: {exc}"}, status_code=500)

    async def _call_endpoint(self, route: _Route, request: Request, path_params: Dict[str, str]) -> Any:
        endpoint = route.endpoint
        signature = inspect.signature(endpoint)
        try:
            type_hints = get_type_hints(endpoint)
        except Exception:  # pragma: no cover - defensive fallback
            type_hints = {}
        kwargs: Dict[str, Any] = {}
        json_body: Any = None
        json_loaded = False
        for name, param in signature.parameters.items():
            annotation = type_hints.get(name, param.annotation)
            if annotation is Request or annotation is inspect._empty and name == "request":
                kwargs[name] = request
                continue
            if name in path_params:
                kwargs[name] = path_params[name]
                continue
            default = param.default
            if isinstance(default, HeaderInfo):
                header_name = default.alias or name.replace("_", "-")
                value = request.headers.get(header_name)
                kwargs[name] = value if value is not None else default.default
                continue
            if isinstance(default, CookieInfo):
                cookie_name = default.alias or name
                value = request.cookies.get(cookie_name, default.default)
                kwargs[name] = value
                continue
            if isinstance(default, FormInfo):
                form_data = await request.form()
                field_name = default.alias or name
                if field_name in form_data:
                    kwargs[name] = form_data[field_name]
                elif default.default is not inspect._empty:
                    kwargs[name] = default.default
                else:
                    raise HTTPException(400, f"Missing required form field '{field_name}'")
                continue
            if isinstance(default, DependencyInfo):
                dependency = default.dependency
                if dependency is None:
                    kwargs[name] = None
                else:
                    value = dependency(request)
                    if inspect.isawaitable(value):
                        value = await value
                    kwargs[name] = value
                continue
            if annotation is not inspect._empty:
                base_model = _lookup_basemodel(annotation)
                if base_model is not None:
                    if not json_loaded:
                        json_body = await request.json()
                        json_loaded = True
                    kwargs[name] = base_model.parse_obj(json_body)
                    continue
            if request.method in {"POST", "PUT", "PATCH"}:
                if not json_loaded:
                    try:
                        json_body = await request.json()
                    except Exception:
                        json_body = {}
                    json_loaded = True
                if isinstance(json_body, dict) and name in json_body:
                    kwargs[name] = json_body[name]
                    continue
            if name in request.query_params:
                kwargs[name] = request.query_params[name]
                continue
            if default is not inspect._empty:
                kwargs[name] = default
                continue
            raise HTTPException(400, f"Missing required parameter '{name}'")
        result = endpoint(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _wrap_http_middleware(
        self,
        middleware: Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]],
        handler: Callable[[Request], Awaitable[Response]],
    ) -> Callable[[Request], Awaitable[Response]]:
        async def call_next(request: Request, _handler=handler) -> Response:
            return await _handler(request)

        async def wrapped(request: Request, _middleware=middleware) -> Response:
            result = _middleware(request, call_next)
            if inspect.isawaitable(result):
                result = await result
            return result

        return wrapped

    def _to_response(self, result: Any, response_class: Optional[type]) -> Response:
        if isinstance(result, Response):
            return result
        if result is None:
            return Response(b"", status_code=204)
        if response_class is not None:
            if issubclass(response_class, Response):
                return response_class(result)
        if isinstance(result, (bytes, bytearray)):
            return Response(bytes(result))
        if isinstance(result, str):
            return Response(result.encode("utf-8"))
        if isinstance(result, Response):
            return result
        return JSONResponse(result)


def _lookup_basemodel(annotation: Any) -> Optional[Type[Any]]:
    try:
        from pydantic import BaseModel  # type: ignore
    except Exception:  # pragma: no cover - pydantic missing
        return None
    try:
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            return annotation
    except Exception:
        return None
    return None
