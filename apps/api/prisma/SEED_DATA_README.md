# San Francisco Beta Seed Data 🌉

This directory contains realistic seed data specifically designed for Woof's **San Francisco beta launch**.

## Overview

The seed data represents a diverse community of 20 SF dog owners across different neighborhoods, each with unique personalities, occupations, and dog breeds. This data is optimized for testing and demonstrating Woof's core features in a real-world SF context.

## Data Structure

### 📍 SF Locations (`sf-locations.ts`)

**12 Dog Parks & Beaches:**
- Fort Funston (Outer Sunset) - Off-leash beach paradise
- Crissy Field (Presidio) - Golden Gate Bridge views
- Corona Heights (Castro) - Hilltop city views
- Dolores Park (Mission) - Social gathering spot
- Alta Plaza (Pacific Heights) - Upscale neighborhood park
- Bernal Heights Park - 360-degree views
- Golden Gate Park Training Area
- Alamo Square - Painted Ladies views
- Buena Vista Park - Wooded trails
- Ocean Beach - Mile-long beach
- Lafayette Park - Pacific Heights hilltop
- Duboce Park - Fenced community park

**5 Dog-Friendly Cafes:**
- Wag Dog Cafe (Pacific Heights)
- The Dogpatch Saloon (Dogpatch)
- Lands End Lookout Cafe (Outer Richmond)
- Noe Valley Bakery
- The Grove - Fillmore

**4 Veterinary Clinics:**
- SAGE Veterinary Centers (Emergency/Specialty)
- SF Veterinary Specialists (Mission)
- Pets Unlimited (Pacific Heights)
- Bay Area Veterinary Hospital (Marina)

**5 Pet Services:**
- Zoom Room Dog Training (SoMa)
- Ruff House Dog Grooming (Noe Valley)
- Wag Hotels SF (SoMa)
- SF Dog Walker Collective (Mission)
- Pet Camp (Bayview)

### 👥 SF Users (`sf-users.ts`)

**20 Diverse Personas** representing SF's tech-forward, diverse community:

#### Tech Professionals
- `techie_sarah` - Software Engineer, Mission District, Golden Retriever owner
- `soma_jessica` - UX Designer, SoMa, Toy Poodle Mix owner
- `dogpatch_sam` - Product Manager, Dogpatch, Siberian Husky owner
- `richmond_kevin` - Data Scientist, Richmond, Beagle owner

#### Creative Professionals
- `artist_jen` - Graphic Designer, Mission, Corgi owner
- `haight_maya` - Musician, Haight-Ashbury, German Shepherd Mix owner

#### Service Industry
- `chef_marcus` - Chef, Mission, Pit Bull owner
- `north_beach_maria` - Restaurant Owner, North Beach, 2 Chihuahuas

#### Healthcare
- `castro_tony` - Nurse at UCSF, Castro, Jack Russell Terrier owner

#### Business Professionals
- `pacific_heights_alex` - Real Estate Agent, Pacific Heights, 2 French Bulldogs
- `financial_jason` - Investment Banker, Financial District, English Bulldog owner

#### Wellness & Lifestyle
- `yoga_rachel` - Yoga Instructor, Outer Sunset, Australian Shepherd owner
- `sunset_lisa` - Surf Instructor, Outer Sunset, Portuguese Water Dog owner

#### And more diverse occupations: Architect, Fashion Buyer, Coffee Shop Owner, Environmental Scientist, Teacher, Remote Worker

### 🐕 SF Pets (`sf-pets.ts`)

**20 Dogs** with realistic SF breed distribution:

#### Popular SF Breeds:
- Golden Retrievers (athletic, beach-loving)
- Labradoodles (family-friendly)
- French Bulldogs (apartment-suitable)
- Corgis (trendy, compact)
- Australian Shepherds (active lifestyle)
- Rescue Mixes (common in SF)

#### Size Distribution:
- Small: 7 dogs (Corgi, Frenchie, Chihuahua, Jack Russell, Poodle Mix, Cavapoo, Boston Terrier)
- Medium: 7 dogs (Aussie, Pit Bull, Border Collie, Husky, Portuguese Water Dog, Beagle, Bulldog)
- Large: 6 dogs (Golden, Lab Mix, Labradoodle, Shepherd Mix, Great Dane, Lab)

#### Energy Levels:
- High: 10 dogs (perfect for testing activity features)
- Medium: 8 dogs (balanced lifestyle)
- Low: 2 dogs (senior/chill dogs)

Each pet includes:
- Realistic temperament traits
- Play style preferences
- Compatibility indicators
- Detailed bio

## Generated Data

The seed script creates:

### Events
5 upcoming SF dog events:
- Crissy Field Morning Meetup (3 days out)
- Fort Funston Beach Day (1 week out)
- Dolores Park Puppy Social (5 days out)
- SF SPCA Adoption Fair (2 weeks out)
- Presidio Trail Hike (10 days out)

### Activities
- 50 realistic activity logs
- Distributed across 20 users
- Various types: walk, run, play, training
- Located at real SF dog parks
- Past 30 days of data

### Posts
- 30 social media posts
- Featuring real SF locations
- Realistic engagement patterns
- Location-tagged at popular parks

### Services
- 5 SF pet service businesses
- Types: training, grooming, boarding, walking, daycare
- Verified status (70% verified rate)
- Ratings: 4.0-5.0 stars

## Running the Seed Script

### Prerequisites
```bash
# Make sure database is set up
pnpm prisma generate
pnpm prisma migrate dev
```

### Run Seed
```bash
# From apps/api directory
pnpm db:seed

# Or from root
pnpm --filter @woof/api db:seed
```

### Output
```
🌉 Starting San Francisco Beta Seed Data Generation...

🧹 Clearing existing data...

👥 Creating SF users and their pets...
  ✅ Created techie_sarah with Sunny (Golden Retriever)
  ✅ Created marina_mike with Duke (Rescue Mix)
  ...

🎉 Creating SF dog events...
  ✅ Created event: Crissy Field Morning Meetup
  ...

🏪 Creating SF pet services...
  ✅ Created service: Zoom Room Dog Training
  ...

🏃 Creating sample activities...
  ✅ Created 50 sample activities

📱 Creating sample posts...
  ✅ Created 30 sample posts

✨ San Francisco seed data complete!

📊 Summary:
   - 20 users created
   - 20 pets created
   - 5 events created
   - 5 services created
   - 50 activities created
   - 30 posts created

🌉 Ready for SF beta launch!
```

## Test Credentials

All users have the same password for testing: `password123`

Sample logins:
- Email: `sarah@example.com` / Password: `password123`
- Email: `mike@example.com` / Password: `password123`
- Email: `jen@example.com` / Password: `password123`

## Why San Francisco?

### Strategic Advantages:
1. **High Pet Ownership**: SF has one of the highest dog ownership rates in the US
2. **Tech-Savvy Population**: Early adopters, app-friendly demographic
3. **Dog-Friendly Culture**: Abundant dog parks, cafes, and services
4. **Concentrated Geography**: 7x7 mile city makes meetups feasible
5. **Weather**: Year-round outdoor activity opportunities
6. **Demographics**: Affluent, willing to pay for premium pet services
7. **Community**: Strong neighborhood identities and local communities

### SF Dog Park Culture:
- Off-leash areas are social hubs
- Regular "park friends" and communities
- Neighborhood-specific vibes (e.g., Pacific Heights vs Mission)
- High engagement with local pet businesses

### Key SF Neighborhoods Represented:
1. Mission District (tech, arts, diverse)
2. Marina District (fitness-focused, young professionals)
3. Pacific Heights (upscale, well-maintained parks)
4. Noe Valley (family-oriented, community-focused)
5. Castro (LGBTQ+ community, active lifestyle)
6. Outer Sunset (beach culture, surfers, laid-back)
7. SoMa (startup scene, office dogs)
8. Dogpatch (emerging neighborhood, namesake!)
9. Haight-Ashbury (artistic, bohemian)
10. And more...

## Data Quality

### Realistic Personas:
- ✅ Authentic SF occupations and lifestyles
- ✅ Neighborhood-specific characteristics
- ✅ Common SF dog breeds
- ✅ Real locations with accurate coordinates
- ✅ Appropriate energy levels for SF lifestyle
- ✅ Diverse age ranges and demographics

### Testing Coverage:
- ✅ Various dog sizes (small apartment dogs to Great Danes)
- ✅ Different energy levels (couch potatoes to marathon runners)
- ✅ Multiple activity types
- ✅ Diverse neighborhoods
- ✅ Various play styles and temperaments
- ✅ Service discovery scenarios
- ✅ Event attendance patterns

## Future Expansion

Once SF beta succeeds, expand to:
1. **Oakland/East Bay** (10-15 users)
2. **Peninsula** (Palo Alto, Mountain View - 5-10 users)
3. **Marin County** (Sausalito, Mill Valley - 5-10 users)
4. **Berkeley** (5-10 users)

Then scale to other major metro areas.

---

**Generated for Woof SF Beta Launch 🐾**
*Bringing SF's dog community together, one park at a time.*
