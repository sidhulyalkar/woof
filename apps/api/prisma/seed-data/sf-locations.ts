/**
 * San Francisco locations for seeding
 * Focused on dog-friendly parks, beaches, cafes, and neighborhoods
 */

export const sfNeighborhoods = [
  'Mission District',
  'Castro',
  'Noe Valley',
  'Pacific Heights',
  'Marina District',
  'Hayes Valley',
  'Dogpatch',
  'Potrero Hill',
  'Russian Hill',
  'North Beach',
  'SoMa',
  'Financial District',
  'Presidio Heights',
  'Inner Sunset',
  'Outer Sunset',
  'Richmond District',
  'Haight-Ashbury',
  'Bernal Heights',
];

export const sfDogParks = [
  {
    name: 'Fort Funston',
    address: 'Fort Funston Rd, San Francisco, CA 94132',
    lat: 37.7136,
    lng: -122.5025,
    neighborhood: 'Outer Sunset',
    features: ['off-leash', 'beach', 'trails', 'ocean-views'],
    description: 'Popular off-leash beach and hiking area with stunning ocean views',
  },
  {
    name: 'Crissy Field',
    address: 'Crissy Field, San Francisco, CA 94129',
    lat: 37.8024,
    lng: -122.4567,
    neighborhood: 'Presidio',
    features: ['off-leash', 'beach', 'bay-views', 'golden-gate-views'],
    description: 'Scenic waterfront park with Golden Gate Bridge views',
  },
  {
    name: 'Corona Heights Dog Park',
    address: 'Roosevelt Way & Museum Way, San Francisco, CA 94114',
    lat: 37.7647,
    lng: -122.4389,
    neighborhood: 'Castro',
    features: ['off-leash', 'fenced', 'city-views', 'hills'],
    description: 'Hilltop park with panoramic city views and separate small/large dog areas',
  },
  {
    name: 'Dolores Park',
    address: 'Dolores St & 19th St, San Francisco, CA 94114',
    lat: 37.7596,
    lng: -122.4269,
    neighborhood: 'Mission District',
    features: ['social', 'city-views', 'picnic-areas'],
    description: 'Popular social gathering spot with great views and vibrant community',
  },
  {
    name: 'Alta Plaza Park',
    address: 'Clay St & Steiner St, San Francisco, CA 94115',
    lat: 37.7913,
    lng: -122.4363,
    neighborhood: 'Pacific Heights',
    features: ['off-leash', 'fenced', 'tennis-courts', 'playground'],
    description: 'Well-maintained park in Pacific Heights with designated off-leash hours',
  },
  {
    name: 'Bernal Heights Park',
    address: 'Bernal Heights Blvd, San Francisco, CA 94110',
    lat: 37.7419,
    lng: -122.4193,
    neighborhood: 'Bernal Heights',
    features: ['off-leash', 'hills', '360-views', 'trails'],
    description: 'Hilltop park with 360-degree city views and off-leash areas',
  },
  {
    name: 'Golden Gate Park Dog Training Area',
    address: '37th Ave & Fulton St, San Francisco, CA 94121',
    lat: 37.7717,
    lng: -122.4965,
    neighborhood: 'Outer Richmond',
    features: ['off-leash', 'fenced', 'training-area', 'large-space'],
    description: 'Dedicated fenced area for dog training and socialization',
  },
  {
    name: 'Alamo Square',
    address: 'Hayes St & Steiner St, San Francisco, CA 94117',
    lat: 37.7766,
    lng: -122.4345,
    neighborhood: 'Hayes Valley',
    features: ['iconic-views', 'painted-ladies', 'picnic-areas'],
    description: 'Famous for Painted Ladies views, popular dog walking spot',
  },
  {
    name: 'Buena Vista Park',
    address: 'Buena Vista Ave W, San Francisco, CA 94117',
    lat: 37.7685,
    lng: -122.4401,
    neighborhood: 'Haight-Ashbury',
    features: ['off-leash', 'forest', 'trails', 'city-views'],
    description: 'Wooded park with trails and off-leash areas',
  },
  {
    name: 'Ocean Beach',
    address: 'Great Highway, San Francisco, CA 94122',
    lat: 37.7596,
    lng: -122.5107,
    neighborhood: 'Outer Sunset',
    features: ['beach', 'off-leash', 'ocean', 'long-walks'],
    description: 'Mile-long beach perfect for off-leash running and ocean play',
  },
  {
    name: 'Lafayette Park',
    address: 'Washington St & Laguna St, San Francisco, CA 94109',
    lat: 37.7917,
    lng: -122.4283,
    neighborhood: 'Pacific Heights',
    features: ['off-leash', 'hills', 'city-views', 'tennis-courts'],
    description: 'Beautiful hilltop park with off-leash hours and city views',
  },
  {
    name: 'Duboce Park',
    address: 'Duboce Ave & Steiner St, San Francisco, CA 94117',
    lat: 37.7694,
    lng: -122.4345,
    neighborhood: 'Duboce Triangle',
    features: ['off-leash', 'fenced', 'community', 'central'],
    description: 'Popular fenced dog park with active community of regulars',
  },
];

export const sfDogFriendlyCafes = [
  {
    name: 'Wag Dog Cafe',
    address: '2323 Fillmore St, San Francisco, CA 94115',
    neighborhood: 'Pacific Heights',
    type: 'cafe',
    description: 'Dog-themed cafe with patio seating and dog treats',
  },
  {
    name: 'The Dogpatch Saloon',
    address: '2496 3rd St, San Francisco, CA 94107',
    neighborhood: 'Dogpatch',
    type: 'bar',
    description: 'Dog-friendly bar with outdoor seating and water bowls',
  },
  {
    name: 'Lands End Lookout Cafe',
    address: '680 Point Lobos Ave, San Francisco, CA 94121',
    neighborhood: 'Outer Richmond',
    type: 'cafe',
    description: 'Cafe near trails with outdoor seating for you and your pup',
  },
  {
    name: 'Noe Valley Bakery',
    address: '4073 24th St, San Francisco, CA 94114',
    neighborhood: 'Noe Valley',
    type: 'bakery',
    description: 'Dog-friendly bakery with outdoor seating',
  },
  {
    name: 'The Grove - Fillmore',
    address: '2016 Fillmore St, San Francisco, CA 94115',
    neighborhood: 'Pacific Heights',
    type: 'cafe',
    description: 'Popular cafe with dog-friendly patio',
  },
];

export const sfVetClinics = [
  {
    name: 'SAGE Veterinary Centers',
    address: '907 Dell Ave, Campbell, CA 95008',
    neighborhood: 'Mission Bay',
    specialty: 'Emergency & Specialty Care',
  },
  {
    name: 'San Francisco Veterinary Specialists',
    address: '600 Alabama St, San Francisco, CA 94110',
    neighborhood: 'Mission District',
    specialty: 'Specialty & Emergency',
  },
  {
    name: 'Pets Unlimited',
    address: '2343 Fillmore St, San Francisco, CA 94115',
    neighborhood: 'Pacific Heights',
    specialty: 'General Practice & Wellness',
  },
  {
    name: 'The Bay Area Veterinary Hospital',
    address: '3301 Divisadero St, San Francisco, CA 94123',
    neighborhood: 'Marina District',
    specialty: 'General Practice',
  },
];

export const sfPetServices = [
  {
    name: 'Zoom Room Dog Training',
    neighborhood: 'SoMa',
    serviceType: 'training',
    specialties: ['obedience', 'agility', 'puppy-classes'],
  },
  {
    name: 'Ruff House Dog Grooming',
    neighborhood: 'Noe Valley',
    serviceType: 'grooming',
    specialties: ['full-service', 'breed-specific', 'spa-treatments'],
  },
  {
    name: 'Wag Hotels SF',
    neighborhood: 'SoMa',
    serviceType: 'boarding',
    specialties: ['luxury-boarding', 'daycare', 'grooming'],
  },
  {
    name: 'SF Dog Walker Collective',
    neighborhood: 'Mission District',
    serviceType: 'walking',
    specialties: ['group-walks', 'private-walks', 'adventure-hikes'],
  },
  {
    name: 'Pet Camp',
    neighborhood: 'Bayview',
    serviceType: 'daycare',
    specialties: ['daycare', 'boarding', 'grooming', 'training'],
  },
];

export const sfEventLocations = [
  {
    name: 'SF SPCA',
    address: '201 Alabama St, San Francisco, CA 94103',
    neighborhood: 'Mission District',
    type: 'nonprofit',
  },
  {
    name: 'Presidio Tunnel Tops',
    address: 'Presidio, San Francisco, CA 94129',
    neighborhood: 'Presidio',
    type: 'park',
  },
  {
    name: 'Fort Mason Center',
    address: '2 Marina Blvd, San Francisco, CA 94123',
    neighborhood: 'Marina District',
    type: 'event-space',
  },
];
