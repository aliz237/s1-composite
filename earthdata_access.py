"""Public NASA CMR discovery and noninteractive ASF credential recovery."""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
import random
import time

import requests

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
ASF_ENDPOINT = "https://cumulus.asf.earthdatacloud.nasa.gov/s3credentials"
SEARCH_ATTEMPTS = 4


def _echo10_record(result):
    """Adapt Earthaccess UMM-G metadata to the compositor's existing schema."""
    u = result["umm"]
    polygons = u.get("SpatialExtent", {}).get("HorizontalSpatialDomain", {}).get("Geometry", {}).get("GPolygons", [])
    return {"concept-id": result["meta"]["concept-id"], "Granule": {
        "GranuleUR": u["GranuleUR"], "Temporal": u["TemporalExtent"],
        "DataGranule": u["DataGranule"],
        "OnlineAccessURLs": {"OnlineAccessURL": [{"URL": link["URL"]} for link in u.get("RelatedUrls", [])]},
        "AdditionalAttributes": {"AdditionalAttribute": [
            {"Name": attr["Name"], "Values": {"Value": attr["Values"][0] if len(attr["Values"]) == 1 else attr["Values"]}}
            for attr in u.get("AdditionalAttributes", [])]},
        "Platforms": {"Platform": u["Platforms"][0]},
        "OrbitCalculatedSpatialDomains": {"OrbitCalculatedSpatialDomain": u["OrbitCalculatedSpatialDomains"]},
        "Spatial": {"HorizontalSpatialDomain": {"Geometry": {"GPolygon": [
            {"Boundary": {"Point": [{"PointLongitude": point["Longitude"], "PointLatitude": point["Latitude"]}
                                   for point in polygon.get("Boundary", {}).get("Points", [])]}}
            for polygon in polygons]}}}}}


def _retry_wait(attempt, response=None):
    delay = min(60, 2 ** attempt * 2)
    if response is not None and response.headers.get("Retry-After"):
        value = response.headers["Retry-After"]
        try:
            delay = max(0, float(value))
        except ValueError:
            try:
                delay = max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    if delay > 300:
        raise RuntimeError("NASA CMR Retry-After exceeds the 300-second retry wait budget; retry the job later")
    return delay + random.uniform(0, 2)


def _transient_search_error(exc):
    """Earthaccess wraps HTTP failures in RuntimeError with an HTTPError cause."""
    chain = set()
    while exc is not None and id(exc) not in chain:
        chain.add(id(exc))
        if isinstance(exc, (requests.ConnectionError, requests.Timeout,
                            requests.exceptions.ChunkedEncodingError, requests.exceptions.JSONDecodeError)):
            return True, None, type(exc).__name__
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            return status in (408, 429, 500, 502, 503, 504), exc.response, f"HTTP {status}"
        exc = exc.__cause__ or exc.__context__
    return False, None, "non-transient error"


def search_granules(*, short_name, temporal, bounding_box, limit):
    """Use earthaccess.search_data for discovery/pagination; never truncate scenes.

    Retry the complete query only when the library raises a transient network
    failure. Request timing and page transport belong to Earthaccess. No login
    is needed for this public collection.
    """
    import earthaccess
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("CMR limit must be a positive integer")
    params = dict(short_name=short_name, temporal=tuple(temporal.split(",")),
                  bounding_box=tuple(float(x) for x in bounding_box.split(",")))
    for attempt in range(1, SEARCH_ATTEMPTS + 1):
        try:
            query = earthaccess.granule_query().parameters(**params)
            try:
                expected = query.hits()
            finally:
                query.session.close()
            if expected >= limit:
                raise RuntimeError(f"NASA CMR has {expected} results, reaching max_results={limit}; increase it or use smaller tiles")
            results = earthaccess.search_data(count=limit, **params)
            break
        except Exception as exc:
            transient, response, reason = _transient_search_error(exc)
            if not transient or attempt == SEARCH_ATTEMPTS:
                raise RuntimeError(f"Earthaccess NASA CMR search failed on attempt {attempt}: {reason}") from exc
            wait = _retry_wait(attempt, response)
            print(f"Earthaccess NASA CMR: {reason}; retry {attempt + 1}/{SEARCH_ATTEMPTS} in {wait:.1f}s", flush=True)
            time.sleep(wait)
    if len(results) >= limit:
        raise RuntimeError(f"NASA CMR search reached max_results={limit}; increase it or use smaller tiles")
    if len(results) != expected:
        raise RuntimeError(f"Incomplete Earthaccess CMR search: received {len(results)} of {expected} results; retry the search")
    ids = [result["meta"]["concept-id"] for result in results]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Earthaccess returned duplicate CMR granules; retry the search")
    print(f"Earthaccess NASA CMR: {len(results)} results", flush=True)
    return [_echo10_record(result) for result in results]


@contextmanager
def _earthdata_environment(values):
    """Restore the process environment after using credentials from MAAP secrets."""
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate_credentials(credentials):
    if not isinstance(credentials, dict) or not all(credentials.get(k) for k in
            ("accessKeyId", "secretAccessKey", "sessionToken")):
        raise RuntimeError("ASF returned no usable temporary S3 credentials")
    return credentials


class ASFCredentials:
    """MAAP first; fall back to Earthdata environment/netrc or MAAP secrets.

    Once recovered through Earthdata, subsequent refreshes use that authenticated
    session directly. Never prompt, persist passwords, or print credentials.
    """
    def __init__(self, maap):
        self.maap = maap
        self.auth = None
        self.use_earthdata = False

    def __call__(self):
        if not self.use_earthdata:
            try:
                return _validate_credentials(self.maap.aws.earthdata_s3_credentials(ASF_ENDPOINT))
            except Exception as exc:
                # Exception bodies from authentication services can contain tokens.
                print(f"MAAP ASF credentials unavailable ({type(exc).__name__}); trying Earthdata login", flush=True)
                self.use_earthdata = True
        if self.auth is None:
            self.auth = self._login()
        try:
            return _validate_credentials(self.auth.get_s3_credentials(endpoint=ASF_ENDPOINT))
        except Exception as exc:
            self.auth = None  # Let the next refresh reauthenticate if necessary.
            raise RuntimeError(f"Direct ASF credential request failed ({type(exc).__name__})") from None

    def _login(self):
        try:
            from earthaccess.auth import Auth
        except ImportError:
            raise RuntimeError("Earthdata credential fallback requires earthaccess; rebuild the Conda environment") from None
        for strategy in ("environment", "netrc"):
            try:
                auth = Auth()
                auth.login(strategy=strategy, persist=False)
                if auth.authenticated:
                    return auth
            except Exception:
                pass
        try:
            # Same secret names used by the HLS compositor; no credentials in DPS args.
            values = {key: self.maap.secrets.get_secret(key)
                      for key in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")}
            if not all(isinstance(v, str) and v for v in values.values()):
                raise ValueError("missing Earthdata secrets")
            with _earthdata_environment(values):
                auth = Auth()
                auth.login(strategy="environment", persist=False)
            if auth.authenticated:
                return auth
        except Exception:
            pass
        raise RuntimeError("Earthdata login failed: supply EARTHDATA_USERNAME/PASSWORD via the environment, "
                           "netrc, or MAAP secrets. No interactive login is attempted.") from None
