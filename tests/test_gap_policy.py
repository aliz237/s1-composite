"""Eligibility prevents donor work for specks; substantial patches remain fillable."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin, array_bounds

from test_gapfill import FakeMAAP
import opera_seasonal_mean as m


class GapPolicyTests(unittest.TestCase):
    def test_component_area_and_missing_core_are_both_required(self):
        missing=np.zeros((120,120),bool)
        missing[5:37,5:37]=True  # Exactly 1024 pixels.
        missing[60:91,60:93]=True  # 1023, despite having a large core.
        missing[115,115]=True
        eligible,stats=m._eligible_gap_mask(missing,1024,7)
        self.assertEqual(stats['eligible_pixels'],1024)
        self.assertEqual(stats['eligible_patches'],1)
        self.assertEqual(stats['skipped_pixels'],1024)
        np.testing.assert_array_equal(eligible[:40,:40],missing[:40,:40])
        self.assertFalse(eligible[60:,:].any())

    def test_many_isolated_pixels_do_not_add_up_to_a_large_patch(self):
        missing=np.zeros((320,320),bool)
        missing[::8,::8]=True
        self.assertGreater(missing.sum(),1024)
        eligible,stats=m._eligible_gap_mask(missing,1024,7)
        self.assertFalse(eligible.any())
        self.assertEqual(stats['largest_patch_pixels'],1)

    def test_thin_strips_are_skipped_and_all_gap_mode_is_available(self):
        missing=np.zeros((20,2100),bool)
        missing[10,:]=True
        eligible,stats=m._eligible_gap_mask(missing,1024,7)
        self.assertFalse(eligible.any())
        self.assertEqual(stats['largest_patch_pixels'],2100)
        eligible,_=m._eligible_gap_mask(missing,1,1)
        np.testing.assert_array_equal(eligible,missing)

    def test_full_missing_tile_and_patch_edges_are_retained(self):
        missing=np.ones((100,100),bool)
        eligible,stats=m._eligible_gap_mask(missing,1024,7)
        np.testing.assert_array_equal(eligible,missing)
        self.assertEqual(stats['eligible_patches'],1)
        self.assertEqual(stats['eligible_pixels'],10000)

    def test_diagonally_touching_patches_use_eight_connectivity(self):
        missing=np.zeros((80,100),bool)
        missing[10:26,10:42]=True
        missing[26:42,42:74]=True
        eligible,stats=m._eligible_gap_mask(missing,1024,7)
        np.testing.assert_array_equal(eligible,missing)
        self.assertEqual(stats['eligible_patches'],1)

    def test_invalid_policy_is_rejected_before_authentication(self):
        import main
        for extra in [['--gap-min-pixels=0'],['--gap-core-size=2']]:
            with self.subTest(arguments=extra),tempfile.TemporaryDirectory() as tmp:
                args=main.parser().parse_args(['--bbox=1,2,3,4','--year=2020','--identifier=bad',
                                              '--output-dir='+tmp,*extra])
                with self.assertRaises(ValueError):
                    main.run_job(args,maap=object())

    def run_job(self,folder,masks):
        shape=masks.shape[-2:]
        transform=from_origin(400000,7200000,30,30)
        maap=FakeMAAP()
        def read(rec,**kw):
            year=int(rec['acquired'][:4])
            col,row=(~transform)*(kw['affine'].c,kw['affine'].f)
            row,col=int(round(row)),int(round(col))
            values=np.full((2,kw['height'],kw['width']),year/1000,dtype='float32')
            if year==2020:
                season={'01':0,'06':1,'09':2}[rec['acquired'][5:7]]
                bad=masks[season,:,row:row+kw['height'],col:col+kw['width']]
                values[bad]=np.nan
            return values,kw['inside'].copy()
        with patch.object(m,'_read_burst',side_effect=read),contextlib.redirect_stdout(io.StringIO()):
            result=m.seasonal_composites(maap,','.join(map(str,array_bounds(*shape,transform))),2020,'policy',
                outdir=folder,crs='EPSG:32606',bbox_crs='EPSG:32606',gap_fill=True,
                write_meta=True,as_of='2026-01-01',speckle_filter=False)
        return result,maap

    def test_413_and_418_pixel_seasons_make_zero_donor_queries(self):
        masks=np.zeros((3,2,96,96),bool)
        masks[0,:,5:12,5:64]=True
        masks[1,:,40:59,2:24]=True
        with tempfile.TemporaryDirectory() as tmp:
            result,maap=self.run_job(tmp,masks)
            self.assertEqual(len(maap.searches),3)
            self.assertTrue(all(s['temporal'].startswith('2020') for s in maap.searches))
            manifest=json.loads(Path(result['manifest']).read_text())
            for season,size in [('frozen',413),('summer',418)]:
                for pol in ('VV','VH'):
                    stats=manifest['seasons'][season]['gap_eligibility'][pol]
                    self.assertEqual(stats['eligible_pixels'],0)
                    self.assertEqual(stats['skipped_pixels'],size)
                self.assertEqual(manifest['seasons'][season]['donors'],[])
            with rasterio.open(result['cog']) as src:
                self.assertTrue((src.read(1)[masks[0,0]] == -9999).all())
            with rasterio.open(result['meta_cog']) as src:
                fields=m.decode_meta(src.read(1),2020)
                self.assertTrue((fields['vv_summer']['source_count'][masks[1,0]]==0).all())
                self.assertFalse(fields['vv_summer']['gap_filled'].any())

    def test_large_patch_fills_but_specks_and_other_polarization_do_not(self):
        masks=np.zeros((3,2,96,96),bool)
        masks[0,0,10:42,10:42]=True
        masks[0,:,80,80]=True
        with tempfile.TemporaryDirectory() as tmp:
            result,maap=self.run_job(tmp,masks)
            self.assertEqual(len(maap.searches),5)
            with rasterio.open(result['cog']) as src:
                vv,vh=src.read((1,2))
                self.assertTrue((vv[10:42,10:42]==np.float32(2.021)).all())
                self.assertTrue((vh[10:42,10:42]==np.float32(2.020)).all())
                self.assertEqual(vv[80,80],-9999)
                self.assertEqual(vh[80,80],-9999)
            with rasterio.open(result['meta_cog']) as src:
                fields=m.decode_meta(src.read(1),2020)
                self.assertEqual(int(fields['vv_frozen']['gap_filled'].sum()),1024)
                self.assertFalse(fields['vh_frozen']['gap_filled'].any())
                self.assertTrue((fields['vv_frozen']['source_count'][10:42,10:42]==3).all())


if __name__=='__main__':
    unittest.main()
