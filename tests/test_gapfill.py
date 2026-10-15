"""Offline selection/boundary checks and actual Rasterio COG integration."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import opera_seasonal_mean as m
import main
from refined_lee import filter_geotiff


def record(year, month, orbit, platform="Sentinel-1A"):
    attrs = {"PRODUCT_VERSION": "1.0", "OPERA_BURST_ID": "test-burst",
             "ASCENDING_DESCENDING": "ASCENDING", "PATH_NUMBER": "135"}
    return {"Granule": {
        "GranuleUR": f"test-{platform}-{year}-{month}-{orbit}",
        "Temporal": {"RangeDateTime": {"BeginningDateTime": f"{year}-{month}-20T12:00:{orbit:02d}Z"}},
        "DataGranule": {"ProductionDateTime": f"{year}-{month}-21T00:00:00Z"},
        "OnlineAccessURLs": {"OnlineAccessURL": [{"URL": f"s3://test/scene_{p}.tif"}
                                                  for p in ("VV", "VH", "mask")]},
        "AdditionalAttributes": {"AdditionalAttribute": [{"Name": k, "Values": {"Value": v}}
                                                          for k, v in attrs.items()]},
        "Platforms": {"Platform": {"ShortName": platform}},
        "OrbitCalculatedSpatialDomains": {"OrbitCalculatedSpatialDomain": {"OrbitNumber": str(orbit)}}}}


class FakeAWS:
    def earthdata_s3_credentials(self, endpoint):
        return dict(accessKeyId="offline", secretAccessKey="offline", sessionToken="offline")


class FakeMAAP:
    aws = FakeAWS()

    def __init__(self):
        self.searches = []

    def searchGranule(self, **kwargs):
        self.searches.append(kwargs)
        year, month = kwargs["temporal"][:4], kwargs["temporal"][5:7]
        n = {"2019": 2, "2020": 1, "2021": 3}.get(year, 0)
        return [record(int(year), month, orbit) for orbit in range(n)]


class GapFillTests(unittest.TestCase):
    def test_nearest_year_quality_tie_and_immutable_counts(self):
        counts = np.array([[[1, 0, 0, 0]], [[1, 0, 0, 0]]], dtype="uint32")
        before = counts.copy()
        mean = np.where(counts, 10, np.nan).astype("float32")
        loaded = []
        def load(year, **kwargs):
            loaded.append(year)
            n = np.zeros_like(counts)
            if year == 2019:
                n[..., :3] = 2
            elif year == 2021:
                n[..., :3] = 2
                n[0, 0, 1] = 3  # Following year wins for VV, preceding for VH.
            elif year == 2018:
                n[...] = 4
            else:
                n[...] = 3
            return np.full_like(mean, year), n, {"year": year, "nominal_doy": 45}
        out, years, doy, used, attempts = m._fill_season(mean, counts, 2020, 46,
                                                        np.ones((1, 4), bool), 3, load)
        np.testing.assert_array_equal(counts, before)
        np.testing.assert_array_equal(years, [[[2020, 2021, 2019, 2018]], [[2020, 2019, 2019, 2018]]])
        self.assertEqual(out[0, 0, 0], 10)
        self.assertEqual(used[0, 0, 1], 3)
        self.assertEqual(doy[0, 0, 0], 46)
        self.assertEqual(loaded, [2019, 2021, 2018, 2022])

    def test_no_donor_work_when_target_complete(self):
        n = np.ones((2, 2, 2), dtype="uint32")
        with patch("builtins.print"), patch.object(m, "_calendar_window"):
            def load(*args, **kwargs):
                self.fail("No donor should be loaded")
            _, years, _, _, attempts = m._fill_season(n.astype("float32"), n, 2020, 46,
                                                      np.ones((2, 2), bool), 5, load)
        self.assertTrue((years == 2020).all())
        self.assertEqual(attempts, [])

    def test_archive_start_current_season_and_leap_boundaries(self):
        cutoff = m._utc("2026-10-14")
        self.assertEqual(m._season_availability(2016, ("01-01", "04-01"), cutoff, donor=True)["status"],
                         "before_archive_start")
        partial = m._season_availability(2016, ("03-01", "05-01"), cutoff)
        self.assertEqual(partial["query_start"], "2016-04-14T00:00:00+00:00")
        self.assertFalse(partial["complete"])
        self.assertEqual(m._season_availability(2016, ("03-01", "05-01"), cutoff, donor=True)["status"],
                         "incomplete_donor_season")
        self.assertEqual(m._season_availability(2026, ("09-15", "11-01"), cutoff, donor=True)["status"],
                         "incomplete_donor_season")
        self.assertEqual(m._season_availability(2027, ("01-01", "04-01"), cutoff, donor=True)["status"],
                         "not_started_as_of_cutoff")
        self.assertEqual(m._season_availability(2019, ("02-29", "04-01"), cutoff, donor=True)["status"],
                         "invalid_calendar_window")
        self.assertEqual(m._season_availability(2020, ("01-01", "04-01"), cutoff)["nominal_doy"], 46)
        self.assertEqual(m._season_availability(2019, ("01-01", "04-01"), cutoff)["nominal_doy"], 46)
        with self.assertRaises(ValueError): m._resolve_cutoff("2099-01-01")

    def test_platform_orbit_keys_separate_c_and_d(self):
        records = [record(2026, "06", 42, p) for p in ("Sentinel-1C", "Sentinel-1D")]
        groups = m._records(records, m._utc("2026-06-01"), m._utc("2026-07-01"))
        self.assertEqual(len(groups), 2)
        self.assertEqual({g[0]["platform"] for g in groups}, {"Sentinel-1C", "Sentinel-1D"})

    def test_boundary_selection_never_queries_unavailable_years(self):
        shape = (2, 1, 1)
        for target, window, cutoff, expected in (
                (2016, ("01-01", "04-01"), "2018-01-01", 2017),
                (2026, ("09-15", "11-01"), "2026-10-14", 2025),
                (2025, ("09-15", "11-01"), "2026-10-14", 2024)):
            queried = []
            def load(year, **kwargs):
                info = m._season_availability(year, window, m._utc(cutoff), donor=True)
                if info["status"] != "ready":
                    return None, None, info
                queried.append(year)
                return np.ones(shape, "float32"), np.ones(shape, "uint32"), info
            _, years, _, _, _ = m._fill_season(np.full(shape, np.nan, "float32"),
                np.zeros(shape, "uint32"), target, 46, np.ones((1, 1), bool), 2, load)
            self.assertEqual(queried, [expected])
            self.assertTrue((years == expected).all())

    def test_donor_errors_are_not_treated_as_missing_data(self):
        def load(*args, **kwargs):
            raise RuntimeError("source access failed")
        with self.assertRaisesRegex(RuntimeError, "source access failed"):
            m._fill_season(np.full((2, 1, 1), np.nan, "float32"),
                np.zeros((2, 1, 1), "uint32"), 2020, 46, np.ones((1, 1), bool), 1, load)

    def test_cli_forwards_gapfill_settings_without_authentication(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = main.parser().parse_args([
                "--bbox=400000,7199520,400480,7200000", "--bbox-crs=EPSG:32606",
                "--crs=EPSG:32606", "--year=2020", "--identifier=test",
                "--gap-fill=true", "--write-meta=true", "--max-gap-years=2", "--as-of=2026-01-01",
                "--input-dir=" + str(Path(tmp)/"missing"), "--output-dir=" + tmp])
            with patch.object(main, "seasonal_composites", return_value={"cog": "mock"}) as spy:
                main.run_job(args, maap=object())
            self.assertTrue(spy.call_args.kwargs["gap_fill"])
            self.assertTrue(spy.call_args.kwargs["write_meta"])
            self.assertEqual(spy.call_args.kwargs["max_gap_years"], 2)
            self.assertEqual(spy.call_args.kwargs["as_of"], m._utc("2026-01-01"))

    def test_current_target_query_is_clipped_at_cutoff(self):
        maap = FakeMAAP()
        def reader(rec, **kwargs):
            inside = kwargs["inside"]
            return np.ones((2, *inside.shape), "float32"), inside
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, "_read_burst", side_effect=reader), contextlib.redirect_stdout(io.StringIO()):
            # Extend fixture availability to the current-year acquisitions.
            def search(**kwargs):
                maap.searches.append(kwargs)
                return [record(2026, kwargs["temporal"][5:7], 1)]
            maap.searchGranule = search
            result = m.seasonal_composites(maap, "400000,7199520,400480,7200000", 2026, "partial",
                outdir=tmp, crs="EPSG:32606", bbox_crs="EPSG:32606", gap_fill=True,
                max_gap_years=0, as_of="2026-10-14", speckle_filter=False)
            meta = json.loads(Path(result["manifest"]).read_text())
            self.assertFalse(meta["seasons"]["shoulder"]["complete"])
            self.assertEqual(maap.searches[-1]["temporal"].split(",")[1], "2026-10-14T00:00:00Z")
            self.assertEqual(meta["source_temporal_extent"]["end_exclusive"], "2026-10-14T00:00:00+00:00")

    def test_2016_frozen_can_be_filled_without_querying_prearchive_data(self):
        maap = FakeMAAP()
        def search(**kwargs):
            maap.searches.append(kwargs)
            return [record(int(kwargs["temporal"][:4]), kwargs["temporal"][5:7], 1)]
        maap.searchGranule = search
        def reader(rec, **kwargs):
            inside = kwargs["inside"]
            return np.ones((2, *inside.shape), "float32"), inside
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, "_read_burst", side_effect=reader), contextlib.redirect_stdout(io.StringIO()):
            result = m.seasonal_composites(maap, "400000,7199520,400480,7200000", 2016, "boundary",
                outdir=tmp, crs="EPSG:32606", bbox_crs="EPSG:32606", gap_fill=True,
                as_of="2018-01-01", speckle_filter=False, write_meta=True)
            with rasterio.open(result["cog"]) as src:
                self.assertTrue((src.read(1) == 1).all())
                self.assertEqual(src.count, 6)
            with rasterio.open(result["meta_cog"]) as src:
                qa = m.decode_meta(src.read(1), 2016)
                self.assertTrue((qa["vv_frozen"]["source_year"] == 2017).all())
                self.assertTrue(qa["vv_frozen"]["gap_filled"].all())
            queried = [s["temporal"] for s in maap.searches]
            self.assertEqual(len(queried), 3)
            self.assertFalse(any(t.startswith("2015") or t.startswith("2016-01") for t in queried))

    def run_raster(self, folder, **options):
        shape = (16, 16)
        maap = FakeMAAP()
        def reader(rec, **kwargs):
            year = int(rec["acquired"][:4])
            frozen = rec["acquired"][5:7] == "01"
            clear = np.zeros(shape, dtype=bool)
            if not frozen:
                clear[:] = True
            elif year == 2020:
                clear[:, 0] = True
            elif year == 2019:
                clear[:, 1:3] = True
            elif year == 2021:
                clear[:, 2:4] = True
            clear &= kwargs["inside"]
            values = np.full((2, *shape), year - 2000, dtype="float32")
            values[:, ~clear] = np.nan
            return values, clear
        with patch.object(m, "_read_burst", side_effect=reader), contextlib.redirect_stdout(io.StringIO()):
            result = m.seasonal_composites(maap, "400000,7199520,400480,7200000", 2020, "test",
                outdir=folder, crs="EPSG:32606", bbox_crs="EPSG:32606", gap_fill=True,
                as_of="2026-01-01", **options)
        return result, maap

    def test_cog_provenance_filter_placement_and_original_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m, "refined_lee", wraps=m.refined_lee) as filter_spy:
                result, maap = self.run_raster(Path(tmp)/"job", write_meta=True)
            self.assertEqual(filter_spy.call_count, 5)
            for call in filter_spy.call_args_list:
                a = call.args[0]
                self.assertEqual(np.unique(a[np.isfinite(a)]).size, 1)
            with rasterio.open(result["cog"]) as src:
                self.assertEqual(src.count, 6)
                self.assertEqual(src.transform, from_origin(400000, 7200000, 30, 30))
                for b in (1, 2):
                    np.testing.assert_array_equal(src.read(b)[0, :5], [20, 19, 21, 21, -9999])
                self.assertEqual(src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"], "COG")
                affine, crs = src.transform, src.crs
            with rasterio.open(result["meta_cog"]) as src:
                self.assertEqual(src.count, 1)
                self.assertEqual(src.dtypes, ("int64",))
                self.assertEqual(src.nodata, -9999)
                self.assertEqual(src.transform, affine)
                self.assertEqual(src.crs, crs)
                self.assertEqual(src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"], "COG")
                qa_values = src.read(1)
                detailed = m.decode_meta(qa_values, 2020)
                for pol in ("vv", "vh"):
                    np.testing.assert_array_equal(detailed[f"{pol}_frozen"]["source_year"][0, :5], [2020, 2019, 2021, 2021, -9999])
                    np.testing.assert_array_equal(detailed[f"{pol}_frozen"]["source_doy"][0, :5], [46, 46, 46, 46, -9999])
                    np.testing.assert_array_equal(detailed[f"{pol}_frozen"]["source_count"][0, :5], [1, 2, 3, 3, 0])
                    np.testing.assert_array_equal(detailed[f"{pol}_frozen"]["target_count"][0, :5], [1, 0, 0, 0, 0])
            meta = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(meta["seasons"]["frozen"]["source_year_pixels"]["VV"]["2021"], 32)
            self.assertEqual(meta["source_temporal_extent"]["start"], "2019-01-01T00:00:00+00:00")
            self.assertEqual(meta["source_temporal_extent"]["end_exclusive"], "2021-04-01T00:00:00+00:00")
            self.assertEqual(len(maap.searches), 5)
            filtered = Path(tmp)/"filtered.tif"
            filter_geotiff(result["meta_cog"], filtered)
            with rasterio.open(filtered) as src:
                np.testing.assert_array_equal(src.read(1), qa_values)
                self.assertEqual(src.tags(1)["speckle_filter_applied"], "false")

    def test_stac_includes_donor_interval_and_optional_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flag in (False, True):
                result, _ = self.run_raster(Path(tmp)/str(flag), write_stac=True, write_meta=flag, speckle_filter=False)
                item = json.loads(Path(result["stac_item"]).read_text())
                self.assertEqual(item["properties"]["start_datetime"], "2019-01-01T00:00:00Z")
                self.assertEqual(item["properties"]["end_datetime"], "2021-03-31T23:59:59.999999Z")
                self.assertTrue(item["properties"]["s1-composite:gap_fill"]["enabled"])
                asset = item["assets"]["composite"]
                self.assertEqual(len(asset["raster:bands"]), 6)
                self.assertEqual("metadata" in item["assets"], flag)
                self.assertEqual("meta_cog" in result, flag)
                self.assertEqual(len(list((Path(tmp)/str(flag)).glob("*_meta.tif"))), int(flag))
                if flag:
                    self.assertEqual(len(item["assets"]["metadata"]["raster:bands"]), 1)
                    self.assertEqual(item["assets"]["metadata"]["raster:bands"][0]["data_type"], "int64")

    def test_all_source_and_count_codes_are_exact(self):
        counts = np.tile(np.arange(128, dtype="int64"), 7)
        years = np.repeat([2020 + d for d in m.SOURCE_OFFSETS], 128)
        years[counts == 0] = 0
        source = np.stack([years, years[::-1]])[:, None, :]
        counts = np.stack([counts, counts[::-1]])[:, None, :]
        season = m._pack_quality(source, counts, 2020)
        packed = season | (season << 20) | (season << 40)
        decoded = m.decode_meta(packed, 2020)
        for season in m.SEASONS:
            for p, pol in enumerate(("vv", "vh")):
                expected = np.where(source[p] == 0, -9999, source[p])
                np.testing.assert_array_equal(decoded[f"{pol}_{season}"]["source_year"], expected)
                np.testing.assert_array_equal(decoded[f"{pol}_{season}"]["source_count"], counts[p])
        with self.assertRaisesRegex(ValueError, "127"):
            m._pack_quality(np.full((2, 1, 1), 2020), np.full((2, 1, 1), 128), 2020)
        with self.assertRaisesRegex(ValueError, "3-year"):
            m._pack_quality(np.full((2, 1, 1), 2024), np.ones((2, 1, 1)), 2020)
        for dtype in ("float32", "float64"):
            with self.assertRaisesRegex(ValueError, "never Float"):
                m.decode_meta(packed.astype(dtype), 2020)
        with self.assertRaises(ValueError):
            m.decode_meta(np.array([2**60], dtype="int64"), 2020)

    def test_nodata_and_custom_leap_year_nominal_day(self):
        windows = dict(m.DEFAULT_WINDOWS, summer=("06-01", "09-01"))
        years = np.array([[[2019, 2020]], [[2021, 0]]], dtype="int16")
        counts = np.array([[[3, 4]], [[1, 0]]], dtype="uint32")
        season = m._pack_quality(years, counts, 2020)
        # Other seasons also need valid missing fields: code 7, count 0.
        missing = np.int64(7 | (7 << 10))
        values = (season << 20) | missing | (missing << 40)
        detailed = m.decode_meta(np.concatenate([np.array([[-9999]], dtype="int64"), values], axis=1), 2020, windows)
        self.assertEqual(detailed["vv_summer"]["source_doy"][0, 1], 198)
        self.assertEqual(detailed["vv_summer"]["source_doy"][0, 2], 199)
        self.assertEqual(detailed["vv_summer"]["source_count"][0, 0], -9999)
        self.assertEqual(detailed["vh_summer"]["source_count"][0, 2], 0)
        self.assertFalse(detailed["vv_summer"]["valid"][0, 0])
        self.assertFalse(detailed["vh_summer"]["valid"][0, 2])
        masked = m.decode_meta(np.ma.masked_equal(np.array([-9999], dtype="int64"), -9999), 2020)
        self.assertFalse(masked["vv_frozen"]["valid"].any())

    def test_cog_overviews_preserve_bits_above_float64_precision(self):
        from rasterio.shutil import copy as rio_copy
        rng = np.random.default_rng(55)
        base = np.zeros((512, 512), dtype="int64")
        for f in range(6):
            base |= (rng.integers(0, 7, base.shape, dtype="int64") |
                     (rng.integers(1, 128, base.shape, dtype="int64") << 3)) << (10 * f)
        base[0, 0] = -9999
        values = base.repeat(2, axis=0).repeat(2, axis=1)
        self.assertTrue(np.any(values.astype("float64").astype("int64") != values))
        with tempfile.TemporaryDirectory() as tmp:
            raw, cog, copied = [Path(tmp)/name for name in ("raw.tif", "cog.tif", "copy.tif")]
            with rasterio.open(raw, "w", driver="GTiff", width=1024, height=1024,
                               count=1, dtype="int64", crs="EPSG:32606",
                               transform=from_origin(400000, 7200000, 30, 30), nodata=-9999,
                               tiled=True) as dst:
                dst.write(values, 1)
                dst.update_tags(1, units="packed_metadata", role="source_provenance", speckle_filter_applied="false")
            rio_copy(raw, cog, driver="COG", compress="DEFLATE", blocksize=512, overview_resampling="nearest")
            with rasterio.open(cog) as src:
                np.testing.assert_array_equal(src.read(1), values)
                self.assertEqual(src.overviews(1), [2])
                self.assertEqual(src.nodata, -9999)
            with rasterio.open(cog, overview_level=0) as src:
                np.testing.assert_array_equal(src.read(1), base)
            filter_geotiff(cog, copied)
            with rasterio.open(copied) as src:
                self.assertEqual(src.dtypes, ("int64",))
                np.testing.assert_array_equal(src.read(1), values)
            m.decode_meta(base, 2020)

    def test_no_gap_fill_companion_is_optional_and_means_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = lambda rec, **kw: (np.ones((2, *kw["inside"].shape), "float32"), kw["inside"])
            arrays = []
            for flag in (False, True):
                with patch.object(m, "_read_burst", side_effect=reader), contextlib.redirect_stdout(io.StringIO()):
                    result = m.seasonal_composites(FakeMAAP(), "400000,7199520,400480,7200000", 2020, "plain",
                        outdir=Path(tmp)/str(flag), crs="EPSG:32606", bbox_crs="EPSG:32606",
                        as_of="2026-01-01", speckle_filter=False, write_meta=flag)
                with rasterio.open(result["cog"]) as src:
                    self.assertEqual(src.count, 6)
                    arrays.append(src.read())
                self.assertEqual("meta_cog" in result, flag)
                if flag:
                    with rasterio.open(result["meta_cog"]) as src:
                        qa = m.decode_meta(src.read(1), 2020)
                        self.assertTrue((qa["vv_frozen"]["source_year"] == 2020).all())
            np.testing.assert_array_equal(*arrays)

    def test_count_overflow_fails_without_writing_final_outputs(self):
        def composite(groups, shape, reader):
            return np.ones((2, *shape), "float32"), np.full((2, *shape), 128, "uint32")
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, "_composite", side_effect=composite), contextlib.redirect_stdout(io.StringIO()):
            kwargs = dict(maap=FakeMAAP(), bbox="400000,7199520,400480,7200000", year=2020,
                identifier="overflow", crs="EPSG:32606", bbox_crs="EPSG:32606",
                as_of="2026-01-01", speckle_filter=False)
            bad = Path(tmp)/"overflow"
            with self.assertRaisesRegex(ValueError, "127"):
                m.seasonal_composites(**kwargs, outdir=bad, write_meta=True)
            self.assertFalse(list(bad.glob("*.tif")))
            self.assertFalse(list(bad.glob("*_manifest.json")))
            result = m.seasonal_composites(**kwargs, outdir=Path(tmp)/"means_only", write_meta=False)
            with rasterio.open(result["cog"]) as src:
                self.assertEqual(src.count, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
