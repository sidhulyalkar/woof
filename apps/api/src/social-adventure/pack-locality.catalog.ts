export const PACK_COARSE_REGIONS = [
  {
    id: 'us-ca-san-francisco',
    displayName: 'San Francisco, CA',
    countryCode: 'US',
    subdivisionCode: 'CA',
    granularity: 'METRO',
  },
  {
    id: 'us-ca-south-bay',
    displayName: 'South Bay, CA',
    countryCode: 'US',
    subdivisionCode: 'CA',
    granularity: 'BROAD_DISTRICT',
  },
  {
    id: 'us-ca-peninsula',
    displayName: 'Peninsula, CA',
    countryCode: 'US',
    subdivisionCode: 'CA',
    granularity: 'BROAD_DISTRICT',
  },
  {
    id: 'us-ca-east-bay',
    displayName: 'East Bay, CA',
    countryCode: 'US',
    subdivisionCode: 'CA',
    granularity: 'BROAD_DISTRICT',
  },
  {
    id: 'us-ca-north-bay',
    displayName: 'North Bay, CA',
    countryCode: 'US',
    subdivisionCode: 'CA',
    granularity: 'BROAD_DISTRICT',
  },
  {
    id: 'us-ca-santa-cruz-county',
    displayName: 'Santa Cruz County, CA',
    countryCode: 'US',
    subdivisionCode: 'CA',
    granularity: 'COUNTY',
  },
] as const;

export type PackCoarseRegion = (typeof PACK_COARSE_REGIONS)[number];
export type PackCoarseRegionId = PackCoarseRegion['id'];

export const PACK_COARSE_REGION_IDS: readonly PackCoarseRegionId[] = PACK_COARSE_REGIONS.map(
  (region) => region.id
);

export const PACK_LOCALITY_CONTRACT =
  'server-approved coarse region only; no device GPS, address, route, or precise venue authority' as const;
