"""Offline discovery/credential failures must not silently remove observations."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import earthdata_access as e
import opera_seasonal_mean as core
from test_gapfill import record


def umm_record(year=2020, month="06", orbit=1):
    g=record(year,month,orbit)["Granule"]
    u=dict(GranuleUR=g["GranuleUR"],TemporalExtent=g["Temporal"],DataGranule=g["DataGranule"],
           Platforms=[g["Platforms"]["Platform"]],
           OrbitCalculatedSpatialDomains=[g["OrbitCalculatedSpatialDomains"]["OrbitCalculatedSpatialDomain"]],
           RelatedUrls=g["OnlineAccessURLs"]["OnlineAccessURL"],
           AdditionalAttributes=[dict(Name=a["Name"],Values=[a["Values"]["Value"]]) for a in g["AdditionalAttributes"]["AdditionalAttribute"]])
    return {"meta":{"concept-id":f"G{year}{month}{orbit}"},"umm":u}


def http_error(status, retry_after=None):
    response=requests.Response();response.status_code=status
    if retry_after is not None:response.headers["Retry-After"]=str(retry_after)
    cause=requests.HTTPError(response=response)
    error=RuntimeError("Earthaccess request failed");error.__cause__=cause
    return error


class SearchTests(unittest.TestCase):
    args=dict(short_name="OPERA_L2_RTC-S1_V1",temporal="2020-06-15T00:00:00Z,2020-09-01T00:00:00Z",
              bounding_box="-148,64,-145,66",limit=20000)

    def setUp(self):
        self.ea=Mock()
        self.query=self.ea.granule_query.return_value.parameters.return_value
        self.query.hits.return_value=1
        self.ea.search_data.return_value=[umm_record()]
        self.modules=patch.dict(sys.modules,{"earthaccess":self.ea})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.output=contextlib.redirect_stdout(io.StringIO());self.output.__enter__();self.addCleanup(self.output.__exit__,None,None,None)

    def test_public_search_data_handles_discovery_and_metadata_adapter(self):
        results=e.search_granules(**self.args)
        self.ea.search_data.assert_called_once_with(short_name=self.args["short_name"],count=20000,
            temporal=("2020-06-15T00:00:00Z","2020-09-01T00:00:00Z"),bounding_box=(-148.,64.,-145.,66.))
        self.ea.login.assert_not_called()
        self.query.session.close.assert_called_once()
        groups=core._records(results,core._utc("2020-06-15"),core._utc("2020-09-01"))
        self.assertEqual(sum(map(len,groups)),1)
        self.assertEqual(groups[0][0]["platform"],"Sentinel-1A")
        self.assertEqual(groups[0][0]["assets"]["VV"],"s3://test/scene_VV.tif")
        self.assertEqual(groups[0][0]["orbit"],"1")

    def test_umm_footprints_and_versions_preserve_selection_and_pruning(self):
        first=umm_record();second=umm_record()
        points=[dict(Longitude=x,Latitude=y) for x,y in [(-148,64),(-145,64),(-145,66),(-148,66),(-148,64)]]
        for result in (first,second):
            result["umm"]["SpatialExtent"]={"HorizontalSpatialDomain":{"Geometry":{"GPolygons":[{"Boundary":{"Points":points}}]}}}
        second["meta"]["concept-id"]="G-new"
        second["umm"]["GranuleUR"]="newest-burst"
        second["umm"]["DataGranule"]["ProductionDateTime"]="2021-06-21T00:00:00Z"
        self.query.hits.return_value=2;self.ea.search_data.return_value=[first,second]
        groups=core._records(e.search_granules(**self.args),core._utc("2020-06-15"),core._utc("2020-09-01"))
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0][0]["id"],"newest-burst")
        self.assertEqual(groups[0][0]["footprint_bounds"],(-148.,64.,-145.,66.))

    def test_network_error_retries_the_complete_library_query(self):
        self.ea.search_data.side_effect=[requests.ConnectTimeout(),[umm_record()]]
        with patch.object(e.time,"sleep") as sleep,patch.object(e.random,"uniform",return_value=0):
            self.assertEqual(len(e.search_granules(**self.args)),1)
        self.assertEqual(self.ea.search_data.call_count,2)
        self.assertEqual(self.query.hits.call_count,2)
        sleep.assert_called_once_with(4)

    def test_throttle_and_gateway_errors_retry_but_permanent_http_does_not(self):
        for status in (429,503,401):
            self.ea.search_data.reset_mock(side_effect=True)
            self.ea.search_data.side_effect=[http_error(status,17),[umm_record()]]
            with self.subTest(status=status),patch.object(e.time,"sleep") as sleep,patch.object(e.random,"uniform",return_value=0):
                if status==401:
                    with self.assertRaisesRegex(RuntimeError,"HTTP 401"):e.search_granules(**self.args)
                    sleep.assert_not_called()
                else:
                    self.assertEqual(len(e.search_granules(**self.args)),1)
                    sleep.assert_called_once_with(17)

    def test_exhaustion_and_hit_query_failures_are_bounded(self):
        self.query.hits.side_effect=requests.ReadTimeout()
        with patch.object(e.time,"sleep") as sleep,self.assertRaisesRegex(RuntimeError,"attempt 4: ReadTimeout"):
            e.search_granules(**self.args)
        self.assertEqual(sleep.call_count,3)
        self.ea.search_data.assert_not_called()
        self.assertEqual(self.query.session.close.call_count,4)

    def test_invalid_inputs_and_metadata_do_not_retry(self):
        for limit in (0,True,1.5):
            with self.subTest(limit=limit),self.assertRaises(ValueError):e.search_granules(**dict(self.args,limit=limit))
        self.ea.search_data.side_effect=ValueError("bad query")
        with patch.object(e.time,"sleep") as sleep,self.assertRaises(RuntimeError):e.search_granules(**self.args)
        sleep.assert_not_called()
        self.assertEqual(self.ea.search_data.call_count,1)

    def test_empty_valid_but_caps_partial_searches_and_duplicates_fail(self):
        self.ea.search_data.return_value=[];self.query.hits.return_value=0
        self.assertEqual(e.search_granules(**self.args),[])
        self.query.hits.return_value=20000
        with self.assertRaises(RuntimeError):e.search_granules(**self.args)
        self.ea.search_data.reset_mock()
        self.query.hits.return_value=2;self.ea.search_data.return_value=[umm_record()]
        with self.assertRaisesRegex(RuntimeError,"Incomplete"):e.search_granules(**self.args)
        self.ea.search_data.return_value=[umm_record(),umm_record()]
        with self.assertRaisesRegex(RuntimeError,"duplicate"):e.search_granules(**self.args)

    def test_default_compositor_uses_earthaccess_without_maap_search(self):
        maap=Mock();maap.searchGranule.side_effect=AssertionError("MAAP search must not be called")
        maap.aws.earthdata_s3_credentials.return_value=CredentialTests.credentials
        self.ea.search_data.side_effect=lambda **kw:[umm_record(month=kw["temporal"][0][5:7])]
        reader=lambda rec,**kw:(np.ones((2,*kw["inside"].shape),"float32"),kw["inside"])
        with tempfile.TemporaryDirectory() as tmp,patch.object(core,"_read_burst",side_effect=reader):
            result=core.seasonal_composites(maap,"400000,7199520,400480,7200000",2020,"library",
                outdir=tmp,crs="EPSG:32606",bbox_crs="EPSG:32606",as_of="2026-01-01",speckle_filter=False)
            import json
            meta=json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(meta["catalog_search"]["client"],"earthaccess.search_data")
            self.assertEqual([meta["seasons"][season]["retained_bursts"] for season in core.SEASONS],[1,1,1])
        self.assertEqual(self.ea.search_data.call_count,3)
        maap.searchGranule.assert_not_called()


class CredentialTests(unittest.TestCase):
    credentials = dict(accessKeyId='offline-key',secretAccessKey='offline-secret',sessionToken='offline-token')

    def setUp(self):
        self.maap = Mock()
        self.maap.aws.earthdata_s3_credentials.return_value = self.credentials
        self.auth = Mock(authenticated=True)
        self.auth.get_s3_credentials.return_value = self.credentials
        self.factory = Mock(return_value=self.auth)
        self.modules = patch.dict(sys.modules,{'earthaccess':SimpleNamespace(),
                          'earthaccess.auth':SimpleNamespace(Auth=self.factory)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_successful_maap_never_reads_secrets_or_imports_earthaccess(self):
        provider = e.ASFCredentials(self.maap)
        self.assertEqual(provider(),self.credentials)
        self.factory.assert_not_called()
        self.maap.secrets.get_secret.assert_not_called()
        self.maap.aws.earthdata_s3_credentials.assert_called_once_with(e.ASF_ENDPOINT)

    def test_environment_fallback_sticks_for_refreshes_and_uses_asf_endpoint(self):
        self.maap.aws.earthdata_s3_credentials.side_effect = requests.ConnectTimeout('sensitive-response')
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):
            provider=e.ASFCredentials(self.maap)
            self.assertEqual(provider(),self.credentials)
            self.assertEqual(provider(),self.credentials)
        self.auth.login.assert_called_once_with(strategy='environment',persist=False)
        self.assertEqual(self.auth.get_s3_credentials.call_count,2)
        self.auth.get_s3_credentials.assert_called_with(endpoint=e.ASF_ENDPOINT)
        self.maap.aws.earthdata_s3_credentials.assert_called_once()
        self.maap.secrets.get_secret.assert_not_called()
        self.assertNotIn('sensitive-response',capture.getvalue())
        self.assertNotIn('offline-secret',capture.getvalue())

    def test_netrc_and_maap_secret_fallback_never_prompt_or_persist(self):
        self.maap.aws.earthdata_s3_credentials.side_effect = requests.ConnectTimeout()
        unavailable=Mock()
        unavailable.login.side_effect=RuntimeError('unavailable')
        self.factory.side_effect=[unavailable,unavailable,self.auth]
        self.maap.secrets.get_secret.side_effect=lambda key: {'EARTHDATA_USERNAME':'test-user','EARTHDATA_PASSWORD':'test-password'}[key]
        def login(**kwargs):
            self.assertEqual(os.environ['EARTHDATA_PASSWORD'],'test-password')
        self.auth.login.side_effect=login
        with patch.dict(os.environ,{'EARTHDATA_USERNAME':'previous'},clear=True),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.ASFCredentials(self.maap)(),self.credentials)
            self.assertEqual(os.environ.get('EARTHDATA_USERNAME'),'previous')
            self.assertNotIn('EARTHDATA_PASSWORD',os.environ)
        self.assertEqual([c.kwargs['strategy'] for c in unavailable.login.call_args_list],['environment','netrc'])
        self.auth.login.assert_called_once_with(strategy='environment',persist=False)

    def test_failed_auth_or_missing_credentials_fail_explicitly_without_secret_details(self):
        self.maap.aws.earthdata_s3_credentials.return_value={}
        self.auth.login.side_effect=RuntimeError('sensitive response')
        self.maap.secrets.get_secret.side_effect=RuntimeError('secret store down')
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'Earthdata login failed') as error:
            e.ASFCredentials(self.maap)()
        self.assertNotIn('sensitive',str(error.exception))
        self.auth.login.side_effect=None
        self.auth.get_s3_credentials.return_value={}
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'Direct ASF credential request failed'):
            e.ASFCredentials(self.maap)()


if __name__=='__main__':unittest.main()
