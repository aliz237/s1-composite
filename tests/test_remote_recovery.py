"""Bounded retries recover remote reads without inventing missing observations."""
import contextlib
from datetime import datetime,timedelta,timezone
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import opera_seasonal_mean as m
from test_gapfill import FakeMAAP


class RemoteRecoveryTests(unittest.TestCase):
    record={'id':'test-burst','assets':{'VV':'s3://bucket/vv.tif','VH':'s3://bucket/vh.tif','mask':'s3://bucket/mask.tif'}}

    def test_transient_throttle_reopens_without_refreshing_credentials(self):
        operation=Mock(side_effect=[RasterioIOError('HTTP response code: 503'), 'success'])
        refresh=Mock()
        with patch.object(m.time,'sleep') as sleep,patch.object(m.random,'uniform',return_value=0),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m._retry_remote_read(operation,self.record,refresh),'success')
        self.assertEqual(operation.call_count,2)
        refresh.assert_not_called()
        sleep.assert_called_once_with(5)

    def test_generic_tiff_failure_refreshes_once_and_has_bounded_backoff(self):
        operation=Mock(side_effect=[RasterioIOError('Error fetching directory link'),RasterioIOError('Read failed'),'ok'])
        refresh=Mock()
        with patch.object(m.time,'sleep') as sleep,patch.object(m.random,'uniform',return_value=0),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m._retry_remote_read(operation,self.record,refresh),'ok')
        self.assertEqual(operation.call_count,3)
        refresh.assert_called_once()
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[5,15])

    def test_missing_object_local_errors_and_validation_errors_are_not_retried(self):
        for error,record in [(RasterioIOError('HTTP response code: 404'),self.record),
                             (RasterioIOError('NoSuchKey'),self.record),
                             (RasterioIOError('corrupt local file'),{'id':'local','assets':{'VV':'local.tif'}}),
                             (ValueError('grid mismatch'),self.record)]:
            with self.subTest(error=str(error)):
                operation=Mock(side_effect=error)
                with self.assertRaises(type(error)),patch.object(m.time,'sleep') as sleep:
                    m._retry_remote_read(operation,record,Mock())
                self.assertEqual(operation.call_count,1)
                sleep.assert_not_called()

    def test_persistent_authentication_failure_refreshes_only_once(self):
        operation=Mock(side_effect=RasterioIOError('ExpiredToken'))
        refresh=Mock()
        with patch.object(m.time,'sleep'),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RasterioIOError,'after 2 attempts'):
            m._retry_remote_read(operation,self.record,refresh)
        refresh.assert_called_once()
        self.assertEqual(operation.call_count,2)

    def test_expiration_is_used_without_logging_credentials(self):
        expiration=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat()
        lifetime=m._credential_lifetime({'expiration':expiration})
        self.assertTrue(479<lifetime<=480)
        self.assertEqual(m._credential_lifetime({}),2700)
        with self.assertRaisesRegex(RuntimeError,'expired credentials'):
            m._credential_lifetime({'expiration':(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat()})

    def test_error_chain_preserves_http_cause(self):
        error=RasterioIOError('Read failed')
        error.__cause__=RasterioIOError('HTTP error code: 503')
        self.assertEqual(m._remote_error_kind(error),'transient_http')

    def test_exhausted_reads_fail_job_without_final_rasters_or_manifest(self):
        maap=FakeMAAP()
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'_read_burst',side_effect=RasterioIOError('Error fetching directory link')) as read,patch.object(m.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RasterioIOError,'after 3 attempts'):
                m.seasonal_composites(maap,'400000,7199520,400480,7200000',2020,'failed',
                    outdir=tmp,crs='EPSG:32606',bbox_crs='EPSG:32606',as_of='2026-01-01',speckle_filter=False, search=maap.searchGranule)
            self.assertEqual(read.call_count,3)
            self.assertFalse(list(Path(tmp).glob('*.tif')))
            self.assertFalse(list(Path(tmp).glob('*_manifest.json')))

    def test_recovered_burst_is_counted_once_and_cache_policy_covers_all_assets(self):
        class Credentials:
            calls=0
            def earthdata_s3_credentials(self,endpoint):
                self.calls+=1
                return dict(accessKeyId='offline',secretAccessKey='offline',sessionToken='offline')
        maap=FakeMAAP()
        maap.aws=Credentials()
        attempts=[]
        def reader(record,**kwargs):
            attempts.append(record['id'])
            config=rasterio.env.getenv()
            paths=config['CPL_VSIL_CURL_NON_CACHED']
            self.assertTrue(all('/vsis3/'+url[5:] in paths for url in record['assets'].values()))
            if len(attempts)==1:
                raise RasterioIOError('TIFF directory read failed')
            return np.ones((2,*kwargs['inside'].shape),'float32'),kwargs['inside']
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'_read_burst',side_effect=reader),patch.object(m.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            result=m.seasonal_composites(maap,'400000,7199520,400480,7200000',2020,'recover',
                outdir=tmp,crs='EPSG:32606',bbox_crs='EPSG:32606',as_of='2026-01-01',
                speckle_filter=False,write_meta=True, search=maap.searchGranule)
            self.assertEqual(maap.aws.calls,2)
            self.assertEqual(attempts[0],attempts[1])
            with rasterio.open(result['meta_cog']) as src:
                fields=m.decode_meta(src.read(1),2020)
                self.assertTrue((fields['vv_frozen']['source_count']==1).all())


if __name__=='__main__':
    unittest.main()
