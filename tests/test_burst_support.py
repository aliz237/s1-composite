"""Real raster checks for gap support, compact reads and equal-pass equivalence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import transform_bounds

from test_gapfill import FakeMAAP, record
import opera_seasonal_mean as m


def assets(folder, name, shape, affine, crs, seed=1):
    rng = np.random.default_rng(seed)
    intensity = rng.gamma(3, .03, size=shape).astype('float32')
    flags = np.zeros(shape, 'uint8')
    flags[31:35, 40:49] = 2
    vh = intensity * .4
    vh[50:54, 48:52] = np.nan
    paths = {}
    for band, values in [('VV', intensity), ('VH', vh), ('mask', flags)]:
        path = folder / f'{name}_{band}.tif'
        with rasterio.open(path, 'w', driver='GTiff', count=1, width=shape[1], height=shape[0],
                           crs=crs, transform=affine, dtype=values.dtype, tiled=True,
                           nodata=255 if band == 'mask' else -9999) as dst:
            dst.write(values, 1)
        paths[band] = str(path)
    return paths


class BurstSupportTests(unittest.TestCase):
    def test_support_keeps_filter_neighbors_and_excludes_outside_tile(self):
        needed = np.zeros((30, 30), bool)
        needed[10, 10] = True
        inside = np.ones_like(needed)
        inside[7, 7] = False
        support = m._support_mask(needed, inside, 3)
        self.assertEqual(int(support.sum()), 48)
        self.assertTrue(support[7:14, 7:14][inside[7:14, 7:14]].all())
        self.assertFalse(support[7, 7])
        np.testing.assert_array_equal(m._support_mask(needed, inside, 0), needed)

    def test_catalog_envelope_prunes_actual_gap_but_keeps_unknown_and_halo(self):
        affine = from_origin(400000, 7200000, 30, 30)
        support = np.zeros((1000, 1000), bool)
        support[50:150, 50:150] = True
        support[850:950, 850:950] = True
        def bounds(top, bottom, left, right):
            return transform_bounds('EPSG:32606', 'EPSG:4326',
                *array_bounds(bottom-top, right-left, affine*rasterio.Affine.translation(left, top)), densify_pts=41)
        group = [dict(id='gap', footprint_bounds=bounds(50,150,50,150)),
                 dict(id='middle', footprint_bounds=bounds(450,550,450,550)),
                 dict(id='halo', footprint_bounds=bounds(150,151,50,150)),
                 dict(id='unknown'), dict(id='dateline', footprint_bounds=(-179,50,179,51)),
                 dict(id='polar', footprint_bounds=(0,86,1,87))]
        kept = m._support_groups([group], support, 'EPSG:32606', affine)
        self.assertEqual([r['id'] for r in kept[0]], ['gap','halo','unknown','dateline','polar'])
        with patch('rasterio.warp.transform_bounds', side_effect=ValueError('ambiguous')):
            self.assertEqual(m._support_groups([group],support,'EPSG:32606',affine), [group])

    def test_compact_reads_reduce_source_pixels_and_skip_empty_without_data_reads(self):
        shape = (512, 512)
        affine = from_origin(400000, 7200000, 30, 30)
        with tempfile.TemporaryDirectory() as tmp:
            paths = assets(Path(tmp), 'burst', shape, affine, 'EPSG:32606')
            rec = dict(id='burst', assets=paths)
            kw = dict(crs='EPSG:32606', affine=affine, width=512, height=512,
                      inside=np.ones(shape,bool), resampling=Resampling.nearest)
            open_dataset = rasterio.open
            sizes = []
            class TracedDataset:
                def __init__(self, *a, **k): self.dataset = open_dataset(*a, **k)
                def __getattr__(self, name): return getattr(self.dataset, name)
                def __enter__(self): return self
                def __exit__(self, *args): self.dataset.close()
                def read(self, *a, **k):
                    data = self.dataset.read(*a, **k)
                    sizes.append(data.size)
                    return data
            support = np.zeros(shape,bool)
            support[200:240,210:260] = True
            with patch('rasterio.open', side_effect=TracedDataset):
                full = m._read_burst(rec, **kw)
                baseline_pixels = sum(sizes); sizes.clear()
                cropped = m._read_burst(rec, support=support, **kw)
                cropped_pixels = sum(sizes)
            self.assertLess(cropped_pixels, baseline_pixels / 20)
            rows, cols = cropped[2]
            np.testing.assert_allclose(cropped[0], full[0][:,rows,cols], equal_nan=True, rtol=0, atol=0)
            np.testing.assert_array_equal(cropped[1], full[1][rows,cols])
            # This source is far outside the support rectangle. Headers only.
            outside = dict(kw, affine=affine*rasterio.Affine.translation(-700, 0))
            with patch('rasterio.open', side_effect=TracedDataset):
                sizes.clear()
                empty = m._read_burst(rec, support=support, **outside)
            self.assertEqual(sizes, [])
            self.assertEqual(empty[0].shape, (2,0,0))
            with self.assertRaisesRegex(ValueError, 'nearest'):
                m._read_burst(rec, support=support, **dict(kw,resampling=Resampling.bilinear))

    def test_compact_cross_crs_reads_preserve_overlap_weighting_counts_and_filter(self):
        shape = (320, 340)
        affine = from_origin(400000, 7200000, 30, 30)
        projected = transform_bounds('EPSG:32606', 'EPSG:3338', *array_bounds(*shape, affine), densify_pts=101)
        native = from_origin(projected[0]-60, projected[3]+60, 20, 20)
        native_shape = (int(np.ceil((projected[3]-projected[1])/20))+6,
                        int(np.ceil((projected[2]-projected[0])/20))+6)
        needed = np.zeros(shape, bool)
        needed[70:145,95:175] = True
        needed[150:230,165:240] = True
        inside = np.ones(shape, bool)
        inside[:4] = False
        support = m._support_mask(needed,inside,3)
        with tempfile.TemporaryDirectory() as tmp:
            a = assets(Path(tmp),'a',native_shape,native,'EPSG:3338',11)
            b = assets(Path(tmp),'b',native_shape,native,'EPSG:3338',12)
            groups = [[dict(id='a',assets=a),dict(id='b',assets=b)], [dict(id='a2',assets=a)]]
            kw = dict(crs='EPSG:32606',affine=affine,width=shape[1],height=shape[0],
                      inside=inside,resampling=Resampling.nearest)
            reproject = rasterio.warp.reproject
            for memory in (0, 1):  # Include forced small GDAL chunks.
                with self.subTest(warp_mem_limit=memory):
                    def warp(*args, **kwargs):
                        return reproject(*args, **kwargs, warp_mem_limit=memory)
                    with patch('rasterio.warp.reproject',side_effect=warp), contextlib.redirect_stdout(io.StringIO()):
                        full, counts = m._composite(groups,shape,lambda rec:m._read_burst(rec,**kw))
                        compact, compact_counts = m._composite(groups,shape,
                            lambda rec:m._read_burst(rec,support=support,**kw))
                    np.testing.assert_array_equal(counts[:,support],compact_counts[:,support])
                    np.testing.assert_allclose(full[:,support],compact[:,support],equal_nan=True,rtol=0,atol=0)
                    full = m.refined_lee(full,valid_mask=(counts>0)&inside[None],tile_size=64)
                    compact = m.refined_lee(compact,valid_mask=(compact_counts>0)&inside[None],tile_size=64)
                    np.testing.assert_allclose(full[:,needed],compact[:,needed],rtol=3e-6,atol=1e-7)

    def test_burst_that_only_supplies_filter_halo_is_preserved(self):
        shape = (256,256)
        affine = from_origin(400000,7200000,30,30)
        inside = np.ones(shape,bool)
        needed = np.zeros(shape,bool)
        needed[100:120,100:120] = True
        support = m._support_mask(needed,inside,3)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            left = assets(folder,'left',(256,100),affine,'EPSG:32606',7)
            right = assets(folder,'right',(256,156),affine*rasterio.Affine.translation(100,0),'EPSG:32606',8)
            groups = [[dict(id='left',assets=left),dict(id='right',assets=right)]]
            kw = dict(crs='EPSG:32606',affine=affine,width=256,height=256,inside=inside,
                      resampling=Resampling.nearest)
            with contextlib.redirect_stdout(io.StringIO()):
                full,counts = m._composite(groups,shape,lambda rec:m._read_burst(rec,**kw))
                cropped,nobs = m._composite(groups,shape,lambda rec:m._read_burst(rec,support=support,**kw))
            self.assertTrue((nobs[:,100:120,97:100] == 1).all())
            np.testing.assert_array_equal(counts[:,support],nobs[:,support])
            filtered = m.refined_lee(full,valid_mask=counts>0)
            compact = m.refined_lee(cropped,valid_mask=nobs>0)
            np.testing.assert_allclose(filtered[:,needed],compact[:,needed],rtol=3e-6,atol=1e-7)
            self.assertFalse(np.allclose(compact[:,100:120,100],full[:,100:120,100]))

    def test_full_job_matches_rc5_and_reports_pruned_reads_and_stage_timings(self):
        shape = (512,512)
        affine = from_origin(400000,7200000,30,30)
        needed = np.zeros(shape,bool)
        needed[56:456,16:96] = True
        needed[56:456,416:496] = True
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = {'target': assets(folder,'target',shape,affine,'EPSG:32606',10)}
            footprints = {}
            for name,left,right in [('left',0,128),('right',384,512),('middle',200,312)]:
                transform = affine*rasterio.Affine.translation(left,0)
                source[name] = assets(folder,name,(512,right-left),transform,'EPSG:32606',20+left)
                footprints[name] = transform_bounds('EPSG:32606','EPSG:4326',
                    *array_bounds(512,right-left,transform),densify_pts=101)
            class Catalog(FakeMAAP):
                def searchGranule(self, **kwargs):
                    self.searches.append(kwargs)
                    year,month=int(kwargs['temporal'][:4]),kwargs['temporal'][5:7]
                    results=[]
                    for name in (['target'] if year==2020 else ['left','right','middle']):
                        result=record(year,month,0)
                        g=result['Granule'];g['GranuleUR']+='-'+name
                        for attr in g['AdditionalAttributes']['AdditionalAttribute']:
                            if attr['Name']=='OPERA_BURST_ID': attr['Values']['Value']=name
                        if name in footprints:
                            west,south,east,north=footprints[name]
                            g['Spatial']={'HorizontalSpatialDomain':{'Geometry':{'GPolygon':{'Boundary':{'Point':[
                                {'PointLongitude':x,'PointLatitude':y} for x,y in
                                [(west,south),(east,south),(east,north),(west,north)]]}}}}}
                        results.append(result)
                    return results
            native_read=m._read_burst
            calls={}
            def run(name, optimized):
                maap=Catalog();calls[name]=[]
                def reader(rec,**kw):
                    if not optimized: kw.pop('support',None)
                    result=native_read(dict(rec,assets=source[rec['burst']]),**kw)
                    values,clear=result[:2]
                    if rec['acquired'].startswith('2020-01'):
                        clear &= ~needed
                        values[:,~clear]=np.nan
                    if rec['acquired'].startswith('2019'):
                        calls[name].append((rec['burst'],values.shape[-2]*values.shape[-1]))
                    return values,clear,*result[2:]
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(m,'_read_burst',side_effect=reader))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if not optimized: stack.enter_context(patch.object(m,'_support_groups',side_effect=lambda g,*a:g))
                    output=m.seasonal_composites(maap,','.join(map(str,array_bounds(*shape,affine))),2020,name,
                        outdir=folder/name,crs='EPSG:32606',bbox_crs='EPSG:32606',gap_fill=True,
                        write_meta=True,as_of='2026-01-01',filter_tile_size=64)
                return output
            baseline=run('rc5',False);optimized=run('rc6',True)
            for key in ('cog','meta_cog'):
                with rasterio.open(baseline[key]) as a,rasterio.open(optimized[key]) as b:
                    if key=='meta_cog': np.testing.assert_array_equal(a.read(),b.read())
                    else: np.testing.assert_allclose(a.read(),b.read(),rtol=3e-6,atol=1e-7)
            self.assertEqual([n for n,_ in calls['rc5']],['left','middle','right'])
            self.assertEqual([n for n,_ in calls['rc6']],['left','right'])
            self.assertLess(sum(n for _,n in calls['rc6']),sum(n for _,n in calls['rc5'])/2)
            manifest=json.loads(Path(optimized['manifest']).read_text())
            donor=manifest['seasons']['frozen']['donors'][0]
            self.assertEqual(donor['read_burst_windows'],2)
            self.assertEqual(donor['skipped_burst_windows'],1)
            self.assertEqual(donor['burst_returned_pixels'],sum(n for _,n in calls['rc6']))
            for key in ('search_seconds','burst_read_seconds','composite_seconds','filter_seconds','other_seconds'):
                self.assertGreaterEqual(donor[key],0)
            self.assertAlmostEqual(sum(donor[k] for k in ('search_seconds','burst_read_seconds',
                'composite_seconds','filter_seconds','other_seconds')),donor['elapsed_seconds'],delta=.003)


if __name__ == '__main__':
    unittest.main()
