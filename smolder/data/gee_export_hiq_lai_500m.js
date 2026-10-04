// Export HiQ-LAI 500 m (Yan et al. 2024) onto the 1 km SMOLDER grid, 2015-2020.
//
// Run once in the Google Earth Engine Code Editor (https://code.earthengine.google.com):
// paste this file, press Run, then start the 6 export tasks in the Tasks tab
// (one per year). Files land in Google Drive, folder HiQ_LAI_1km.
//
// Source: projects/verselab-398313/assets/HiQ_LAI/wgs_500m_8d, band LAI,
// uint8, valid 0-100 with scale factor 0.1, 249-255 = fill (data description
// of the HiQ-LAI repository). The 5 km Zenodo version used so far is a
// nearest-neighbour subsample of these 500 m data.
//
// Each 1 km cell is the area-weighted mean of the valid 500 m pixels inside
// it (fill values excluded), on exactly the grid of the daily cubes
// (EPSG:4326, origin 112.904998779 / -9.005000114, pixel 0.009997566 x
// 0.009997122 deg, 4110 x 3474). Output per year: one GeoTIFF with one band
// per 8-day composite, named by its start date (YYYYMMDD), values
// LAI x 100 as uint16 (LAI = value / 100), 65535 = no valid 500 m pixel.

var COLLECTION = 'projects/verselab-398313/assets/HiQ_LAI/wgs_500m_8d';
var X0 = 112.904998779, DX = 0.009997566018978103;
var Y0 = -9.005000113999998, DY = -0.009997121616580312;
var W = 4110, H = 3474;
var TRANSFORM = [DX, 0, X0, 0, DY, Y0];
var REGION = ee.Geometry.Rectangle([X0, Y0 + H * DY, X0 + W * DX, Y0], 'EPSG:4326', false);

var col = ee.ImageCollection(COLLECTION);
print('first image', col.first());                       // check band names and dates here
print('band names', col.first().bandNames());

function toKm(img) {
  var lai = img.select('LAI');
  var valid = lai.updateMask(lai.lte(100));               // drop fill values 249-255
  return valid.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 64})
              .multiply(10).round()                        // raw 0-100 (x0.1 LAI) -> LAI x 100
              .unmask(65535).toUint16()
              .rename(ee.Date(img.get('system:time_start')).format('YYYYMMdd'));
}

for (var year = 2015; year <= 2020; year++) {
  var yc = col.filterDate(year + '-01-01', (year + 1) + '-01-01')
              .filterBounds(REGION)
              .sort('system:time_start');
  print(year + ': composites', yc.size(),
        yc.aggregate_array('system:time_start').map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }));
  var stack = ee.ImageCollection(yc.map(toKm)).toBands();
  // toBands prefixes band names with the image index: keep only the date
  stack = stack.rename(stack.bandNames().map(function (b) { return ee.String(b).split('_').get(-1); }));
  Export.image.toDrive({
    image: stack,
    description: 'hiq_lai_1km_' + year,
    folder: 'HiQ_LAI_1km',
    fileNamePrefix: 'hiq_lai_1km_' + year,
    crs: 'EPSG:4326',
    crsTransform: TRANSFORM,
    region: REGION,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
}
