"""Write a portable, validated DPS STAC catalog and one item beside a COG."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import pystac
from pystac.extensions.raster import DataType, RasterBand, RasterExtension
import rasterio
import rio_stac


def _utc(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Manifest times must include a UTC offset")
    return dt


def _z(value):
    return value.isoformat().replace("+00:00", "Z")


def write_stac(cog_path, manifest_path):
    cog, manifest = Path(cog_path).resolve(), Path(manifest_path).resolve()
    if cog.parent != manifest.parent:
        raise ValueError("COG and manifest must be in the same output directory")
    meta = json.loads(manifest.read_text())
    item_path, catalog_path = cog.with_suffix(".json"), cog.parent / "catalog.json"
    if item_path.exists() or catalog_path.exists():
        raise FileExistsError("STAC outputs already exist; use a fresh DPS job directory")
    windows = meta["seasons"]
    extent = meta.get("source_temporal_extent")
    start = _utc(extent["start"]) if extent else min(_utc(s["start"]) for s in windows.values())
    end_exclusive = (_utc(extent["end_exclusive"]) if extent else
                     max(_utc(s["end_exclusive"]) for s in windows.values()))
    end = end_exclusive - timedelta(microseconds=1)
    identities = {
        k: meta[k] for k in ("year", "identifier", "bbox", "crs", "transform", "width", "height",
                             "resampling", "temporal_weighting", "speckle_filter", "speckle_filter_version",
                             "filter_stage", "nodata", "mask_values_kept")
    }
    identities["seasons"] = {
        name: {"start": s["start"], "end_exclusive": s["end_exclusive"],
               "source_ids": sorted(r["id"] for r in s["records"])} for name, s in windows.items()
    }
    identities["bbox_crs"] = meta.get("bbox_crs", "EPSG:4326")
    identities["gap_fill"] = meta.get("gap_fill", {"enabled": False})
    identities["metadata_raster"] = meta.get("metadata_raster")
    identities["donors"] = {name: [{"year": d["year"],
                                       "source_ids": sorted(r["id"] for r in d["records"])}
                                      for d in s.get("donors", []) if any(d["selected_pixels"].values())]
                             for name, s in windows.items()}
    signature = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()[:16]
    item_id = f"{cog.stem}-{signature}"
    properties = {
        "start_datetime": _z(start), "end_datetime": _z(end),
        "constellation": "sentinel-1", "instruments": ["c-sar"],
        "s1-composite:year": meta["year"], "s1-composite:identifier": meta["identifier"],
        "s1-composite:source_collection": "OPERA_L2_RTC-S1_V1",
        "s1-composite:polarizations": ["VV", "VH"],
        "s1-composite:requested_bbox": meta["bbox"],
        "s1-composite:requested_bbox_crs": meta.get("bbox_crs", "EPSG:4326"),
        "s1-composite:resampling": meta["resampling"],
        "s1-composite:mask_values_kept": meta["mask_values_kept"],
        "s1-composite:temporal_weighting": meta["temporal_weighting"],
        "s1-composite:speckle_filter": meta["speckle_filter"],
        "s1-composite:filter_version": meta["speckle_filter_version"],
        "s1-composite:filter_stage": meta["filter_stage"],
        "s1-composite:gap_fill": meta.get("gap_fill", {"enabled": False}),
        "s1-composite:season_windows": {name: {
            "start": s["start"], "end_exclusive": s["end_exclusive"],
            "retained_bursts": s["retained_bursts"], "passes": s["passes"],
            "donor_years_used": [d["year"] for d in s.get("donors", []) if any(d["selected_pixels"].values())]}
            for name, s in windows.items()},
    }
    # Read headers/georeferencing only: do not scan the raster for statistics.
    item = rio_stac.create_stac_item(
        str(cog), id=item_id, properties=properties, input_datetime=None,
        asset_name="composite", asset_href=str(cog),
        asset_media_type=pystac.MediaType.COG, asset_roles=["data"],
        with_proj=True, with_raster=False, with_eo=False, geom_densify_pts=21)
    raster_bands, common_bands = [], []
    with rasterio.open(cog) as src:
        if src.count != len(meta["bands"]) or src.count not in (6, 12, 30):
            raise ValueError("Expected 6 raster bands (or legacy 12/30) matching the manifest")
        for b in meta["bands"]:
            index, name = b["index"], b["name"]
            if src.descriptions[index - 1] != name:
                raise ValueError("Raster descriptions differ from manifest band names")
            units = b["units"]
            source = b.get("role") == "source_provenance"
            if units == "calendar_year":
                unit, description = "year", "Source calendar year of the selected seasonal mean; -9999 means no observations"
            elif units == "nominal_day_of_year":
                unit, description = "day", "Nominal DOY: UTC calendar day containing midpoint of the source season window, not a measured acquisition date"
            elif units == "valid_pass_count":
                unit = "count"
                description = ("Valid satellite-pass count in the selected source year" if source else
                               "Original target-year valid satellite-pass count, unchanged by gap filling")
                description += "; 0 means no observations inside bbox; -9999 is nodata"
            else:
                unit, description = "1", "Seasonal equal-pass mean of linear gamma-zero backscatter; may use the recorded donor year if gap filling is enabled"
            raster_bands.append(RasterBand.create(data_type=DataType.FLOAT32,
                                                  nodata=meta["nodata"],
                                                  unit=unit))
            common_bands.append({"name": name, "description": description})
    asset = item.assets["composite"]
    asset.title = "Three-season Sentinel-1 VV/VH means and quality"
    asset.description = "Six Float32 seasonal VV/VH means; exact counts and source provenance are in the optional Int64 companion."
    asset.extra_fields["bands"] = common_bands
    RasterExtension.ext(asset, add_if_missing=True).bands = raster_bands
    item.add_asset("manifest", pystac.Asset(str(manifest), media_type=pystac.MediaType.JSON,
                                          roles=["metadata"], title="Source acquisitions and processing manifest"))
    details = meta.get("metadata_raster")
    if details:
        meta_cog = cog.parent / details["filename"]
        with rasterio.open(meta_cog) as src:
            if (src.count != 1 or list(src.descriptions) != [b["name"] for b in details["bands"]]
                    or src.crs != rasterio.crs.CRS.from_wkt(meta["crs"])
                    or list(src.transform)[:6] != meta["transform"]
                    or (src.width, src.height) != (meta["width"], meta["height"])
                    or any(t != "int64" for t in src.dtypes) or src.nodata != meta["nodata"]):
                raise ValueError("Companion metadata raster does not match the manifest/grid")
        extra = pystac.Asset(str(meta_cog), media_type=pystac.MediaType.COG,
                            roles=["metadata"], title="Exact counts and packed source provenance")
        extra.extra_fields["bands"] = [{"name": b["name"], "description": "Packed per-polarization source-year code and exact valid-pass count"}
                                        for b in details["bands"]]
        extra.extra_fields["s1-composite:encoding"] = details
        item.add_asset("metadata", extra)
        RasterExtension.ext(extra, add_if_missing=True).bands = [
            RasterBand.create(data_type=DataType.INT64, nodata=meta["nodata"], unit="1") for _ in details["bands"]]
    # DPS supplies username, algorithm/version, job tag, created and dps-metadata.
    # Do not guess the final S3 prefix or fabricate those provenance properties.
    catalog = pystac.Catalog(id="DPS", description="Seasonal Sentinel-1 RTC composite outputs",
                             catalog_type=pystac.CatalogType.SELF_CONTAINED)
    catalog.set_self_href(str(catalog_path))
    catalog.add_item(item)
    # add_item applies PySTAC's default nested layout. Restore the sidecar
    # location before resolving relative asset links and saving the catalog.
    item.set_self_href(str(item_path))
    catalog.make_all_asset_hrefs_relative()
    item.validate()
    catalog.validate_all()
    # Ensure strict JSON: no numeric NaN/Infinity serialization.
    json.dumps(item.to_dict(), allow_nan=False)
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return {"stac_item": str(item_path), "stac_catalog": str(catalog_path), "stac_item_id": item_id}
