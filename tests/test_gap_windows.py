"""Sparse/dense donor equivalence with real raster windows and reprojection."""
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
from rasterio.warp import transform_bounds

from test_gapfill import FakeMAAP
import opera_seasonal_mean as m


class GapWindowsTests(unittest.TestCase):
    def test_planner_handles_sparse_dense_and_empty_masks(self):
        shape = (3000, 3000)
        mask = np.zeros(shape, bool)
        self.assertEqual(m._gap_windows(mask), [])
        mask[100, 100] = True
        windows = m._gap_windows(mask)
        self.assertEqual(windows, [(slice(97,104), slice(97,104))])
        mask[2900, 2800] = True
        windows = m._gap_windows(mask)
        self.assertEqual(len(windows), 2)
        self.assertEqual(sum((r.stop-r.start)*(c.stop-c.start) for r,c in windows),98)
        mask[:] = True
        self.assertEqual(m._gap_windows(mask), [(slice(0,3000),slice(0,3000))])
        mask[:] = False
        mask[200:2500, 400:2000] = True
        self.assertEqual(m._gap_windows(mask), [(slice(197,2503),slice(397,2003))])

    def test_footprint_pruning_is_conservative_and_optional(self):
        polygon = {'Spatial': {'HorizontalSpatialDomain': {'Geometry': {'GPolygon': {
            'Boundary': {'Point': [{'PointLongitude': x, 'PointLatitude': y}
                                   for x,y in [(0,50),(1,50),(1,51),(0,51)]]}}}}}}
        self.assertEqual(m._footprint_bounds(polygon), (0,50,1,51))
        self.assertIsNone(m._footprint_bounds({}))
        records = [[{'id':'near','footprint_bounds':(0,50,1,51)},
                    {'id':'unknown'}, {'id':'edge','footprint_bounds':(1.005,50,2,51)},
                    {'id':'far','footprint_bounds':(10,50,11,51)}]]
        retained = m._window_groups(records,(0,50,1,51))
        self.assertEqual([r['id'] for r in retained[0]], ['near','unknown','edge'])

    def test_farther_years_only_receive_unresolved_pixels(self):
        n = np.zeros((2,1,3),dtype='uint32')
        requested = {}
        def load(year, *, donor, needed):
            requested[year] = needed.copy()
            counts = np.zeros_like(n)
            if year == 2019: counts[:,:,0] = 2
            if year == 2021: counts[:,:,1] = 3
            if year == 2018: counts[:,:,2] = 4
            return np.ones(n.shape,'float32'),counts,{'year':year,'nominal_doy':46}
        m._fill_season(np.full(n.shape,np.nan,'float32'),n,2020,46,np.ones((1,3),bool),2,load)
        np.testing.assert_array_equal(requested[2019], [[True,True,True]])
        np.testing.assert_array_equal(requested[2021], [[False,True,True]])
        np.testing.assert_array_equal(requested[2018], [[False,False,True]])
        self.assertNotIn(2022, requested)

    def test_real_raster_sparse_and_dense_donors_match_full_tile(self):
        shape = (320,340)
        transform = from_origin(400000,7200000,30,30)
        native_bounds = transform_bounds('EPSG:32606','EPSG:3338',
                                        *array_bounds(*shape, transform),densify_pts=41)
        native_transform = from_origin(native_bounds[0]-60,native_bounds[3]+60,20,20)
        rng = np.random.default_rng(77)
        native_shape = (int(np.ceil((native_bounds[3]-native_bounds[1])/20))+6,
                        int(np.ceil((native_bounds[2]-native_bounds[0])/20))+6)
        intensity = rng.gamma(2, .1, size=native_shape).astype('float32')
        flags = np.zeros(native_shape,'uint8')
        flags[200:204,210:217] = 2
        native_read = m._read_burst
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            assets = {}
            for name, values in [('VV',intensity),('VH',intensity*.4),('mask',flags)]:
                path=folder/(name+'.tif')
                with rasterio.open(path,'w',driver='GTiff',count=1,width=native_shape[1],height=native_shape[0],
                                   crs='EPSG:3338',transform=native_transform,dtype=values.dtype,
                                   nodata=255 if name=='mask' else -9999) as dst:
                    dst.write(values,1)
                assets[name]=str(path)
            for mode,kind in [('nearest','sparse'),('bilinear','sparse'),('average','dense'),('nearest','full')]:
                with self.subTest(resampling=mode,gaps=kind):
                    gaps=np.zeros(shape,bool)
                    if kind=='sparse':
                        for row,col in [(40,45),(44,49),(178,185),(280,320),(0,0),(319,339)]: gaps[row,col]=True
                    elif kind=='dense': gaps[64:272,35:295]=True
                    else: gaps[:]=True
                    calls=[]
                    def reader(rec, **kw):
                        acquired=int(rec['acquired'][:4])
                        calls.append((acquired,kw['height'],kw['width']))
                        values,clear=native_read(dict(rec,assets=assets),**kw)
                        if acquired==2020 and rec['acquired'][5:7]=='01':
                            col,row=(~transform)*(kw['affine'].c,kw['affine'].f)
                            row,col=int(round(row)),int(round(col))
                            clear &= ~gaps[row:row+kw['height'],col:col+kw['width']]
                            values[:,~clear]=np.nan
                        values *= (acquired-2010)/10
                        return values,clear
                    common=dict(bbox=','.join(map(str,array_bounds(*shape,transform))),year=2020,
                                identifier=kind+'_'+mode,crs='EPSG:32606',bbox_crs='EPSG:32606',gap_fill=True,
                                write_meta=True,as_of='2026-01-01',resampling=mode, gap_min_pixels=1, gap_core_size=1)
                    with patch.object(m,'_read_burst',side_effect=reader), contextlib.redirect_stdout(io.StringIO()):
                        sparse=m.seasonal_composites(FakeMAAP(),**common,outdir=folder/(kind+mode+'small'))
                        sparse_calls=calls.copy();calls.clear()
                        with patch.object(m,'_gap_windows',side_effect=lambda mask,halo:[(slice(0,shape[0]),slice(0,shape[1]))]):
                            full=m.seasonal_composites(FakeMAAP(),**common,outdir=folder/(kind+mode+'full'))
                    with rasterio.open(sparse['cog']) as a,rasterio.open(full['cog']) as b:
                        np.testing.assert_allclose(a.read(),b.read(),rtol=3e-6,atol=1e-7)
                    with rasterio.open(sparse['meta_cog']) as a,rasterio.open(full['meta_cog']) as b:
                        np.testing.assert_array_equal(a.read(),b.read())
                    meta=json.loads(Path(sparse['manifest']).read_text())
                    used=[d for d in meta['seasons']['frozen']['donors'] if d['status']=='processed']
                    self.assertEqual(len(used),1)
                    if kind=='sparse' and mode=='nearest':
                        self.assertTrue(all(d['processed_grid_pixels'] < shape[0]*shape[1] for d in used))
                        self.assertTrue(all(h*w < shape[0]*shape[1] for year,h,w in sparse_calls if year!=2020))
                    if mode != 'nearest':
                        self.assertTrue(all(d['processing_strategy']=='full_tile_resampling_compatibility' for d in used))
                    if kind=='full':
                        self.assertTrue(all(len(d['processing_windows'])==1 for d in used))
                        self.assertTrue(all(d['processed_grid_pixels']==shape[0]*shape[1] for d in used))


if __name__ == '__main__':
    unittest.main()
