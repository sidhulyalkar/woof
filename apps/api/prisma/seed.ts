import { PrismaClient } from '@prisma/client';
import * as bcrypt from 'bcrypt';
import { sfUsers } from './seed-data/sf-users';
import { sfPets } from './seed-data/sf-pets';
import { sfDogParks, sfDogFriendlyCafes, sfPetServices, sfEventLocations } from './seed-data/sf-locations';

const prisma = new PrismaClient();

async function main() {
  console.log('🌉 Starting San Francisco Beta Seed Data Generation...\n');

  // Clear existing data
  console.log('🧹 Clearing existing data...');
  await prisma.post.deleteMany();
  await prisma.activity.deleteMany();
  await prisma.meetup.deleteMany();
  await prisma.event.deleteMany();
  await prisma.service.deleteMany();
  await prisma.pet.deleteMany();
  await prisma.user.deleteMany();

  // Create Users and Pets
  console.log('\n👥 Creating SF users and their pets...');
  const createdUsers = [];

  for (const userData of sfUsers) {
    const hashedPassword = await bcrypt.hash(userData.password, 10);

    const user = await prisma.user.create({
      data: {
        handle: userData.handle,
        email: userData.email,
        passwordHash: hashedPassword,
        bio: userData.bio,
        location: userData.neighborhood,
        profileComplete: true,
      },
    });

    // Create pet for this user
    const petData = sfPets[userData.handle];
    if (petData) {
      await prisma.pet.create({
        data: {
          userId: user.id,
          name: petData.name,
          species: petData.species,
          breed: petData.breed,
          age: petData.age,
          size: petData.size,
          energyLevel: petData.energyLevel,
          temperament: petData.temperament,
          bio: petData.bio,
          preferences: petData.preferences,
        },
      });
      console.log(`  ✅ Created ${user.handle} with ${petData.name} (${petData.breed})`);
    }

    createdUsers.push(user);
  }

  // Create Events at SF locations
  console.log('\n🎉 Creating SF dog events...');
  const now = new Date();
  const events = [
    {
      title: 'Crissy Field Morning Meetup',
      description: 'Start your Saturday with coffee and dog playtime at Crissy Field! Bring your pup for off-leash fun with Golden Gate Bridge views.',
      location: 'Crissy Field, San Francisco',
      latitude: 37.8024,
      longitude: -122.4567,
      datetime: new Date(now.getTime() + 3 * 24 * 60 * 60 * 1000), // 3 days from now
      capacity: 30,
      category: 'social',
    },
    {
      title: 'Fort Funston Beach Day',
      description: 'Monthly beach adventure! Off-leash beach running, ocean play, and sunset photos.',
      location: 'Fort Funston, San Francisco',
      latitude: 37.7136,
      longitude: -122.5025,
      datetime: new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000), // 1 week from now
      capacity: 50,
      category: 'social',
    },
    {
      title: 'Dolores Park Puppy Social',
      description: 'Puppies under 1 year! Socialization and play in the heart of the Mission.',
      location: 'Dolores Park, San Francisco',
      latitude: 37.7596,
      longitude: -122.4269,
      datetime: new Date(now.getTime() + 5 * 24 * 60 * 60 * 1000),
      capacity: 20,
      category: 'training',
    },
    {
      title: 'SF SPCA Adoption Fair',
      description: 'Meet adoptable dogs and support rescue! Food trucks, demos, and community vibes.',
      location: 'SF SPCA, 201 Alabama St',
      latitude: 37.7651,
      longitude: -122.4103,
      datetime: new Date(now.getTime() + 14 * 24 * 60 * 60 * 1000), // 2 weeks from now
      capacity: 100,
      category: 'community',
    },
    {
      title: 'Presidio Trail Hike',
      description: 'Explore the Presidio trails with fellow dog lovers. Moderate difficulty, 5 miles.',
      location: 'Presidio Tunnel Tops',
      latitude: 37.7989,
      longitude: -122.4662,
      datetime: new Date(now.getTime() + 10 * 24 * 60 * 60 * 1000),
      capacity: 25,
      category: 'fitness',
    },
  ];

  for (const eventData of events) {
    const organizer = createdUsers[Math.floor(Math.random() * createdUsers.length)];
    const event = await prisma.event.create({
      data: {
        ...eventData,
        organizerId: organizer.id,
      },
    });
    console.log(`  ✅ Created event: ${event.title}`);
  }

  // Create Pet Services
  console.log('\n🏪 Creating SF pet services...');
  for (const service of sfPetServices) {
    await prisma.service.create({
      data: {
        name: service.name,
        type: service.serviceType,
        description: `${service.specialties.join(', ')} in ${service.neighborhood}`,
        location: service.neighborhood,
        verified: Math.random() > 0.3, // 70% verified
        rating: 4 + Math.random(), // 4.0 - 5.0 rating
      },
    });
    console.log(`  ✅ Created service: ${service.name}`);
  }

  // Create Sample Activities
  console.log('\n🏃 Creating sample activities...');
  const activityTypes = ['walk', 'run', 'play', 'training'];
  const activityCount = 50;

  for (let i = 0; i < activityCount; i++) {
    const user = createdUsers[Math.floor(Math.random() * createdUsers.length)];
    const pets = await prisma.pet.findMany({ where: { userId: user.id } });

    if (pets.length > 0) {
      const park = sfDogParks[Math.floor(Math.random() * sfDogParks.length)];
      const daysAgo = Math.floor(Math.random() * 30);

      await prisma.activity.create({
        data: {
          userId: user.id,
          petId: pets[0].id,
          type: activityTypes[Math.floor(Math.random() * activityTypes.length)],
          location: park.name,
          latitude: park.lat,
          longitude: park.lng,
          distance: Math.random() * 5 + 1, // 1-6 km
          duration: Math.floor(Math.random() * 90) + 15, // 15-105 minutes
          calories: Math.floor(Math.random() * 300) + 50, // 50-350 calories
          date: new Date(now.getTime() - daysAgo * 24 * 60 * 60 * 1000),
        },
      });
    }
  }
  console.log(`  ✅ Created ${activityCount} sample activities`);

  // Create Sample Posts
  console.log('\n📱 Creating sample posts...');
  const postTemplates = [
    'Amazing morning at {park}! {pet} made so many new friends 🐕',
    'Perfect weather for a walk at {park} today! Who else was there?',
    'Just discovered {park} and {pet} is in love! New favorite spot ❤️',
    '{pet} living their best life at {park}! 📸',
    'Coffee and dog playtime at {park} - the perfect Saturday!',
  ];

  const postCount = 30;
  for (let i = 0; i < postCount; i++) {
    const user = createdUsers[Math.floor(Math.random() * createdUsers.length)];
    const pets = await prisma.pet.findMany({ where: { userId: user.id } });
    const park = sfDogParks[Math.floor(Math.random() * sfDogParks.length)];
    const template = postTemplates[Math.floor(Math.random() * postTemplates.length)];

    const content = template
      .replace('{park}', park.name)
      .replace('{pet}', pets[0]?.name || 'my pup');

    await prisma.post.create({
      data: {
        userId: user.id,
        content,
        petId: pets[0]?.id,
        location: park.name,
        latitude: park.lat,
        longitude: park.lng,
      },
    });
  }
  console.log(`  ✅ Created ${postCount} sample posts`);

  console.log('\n✨ San Francisco seed data complete!\n');
  console.log('📊 Summary:');
  console.log(`   - ${sfUsers.length} users created`);
  console.log(`   - ${Object.keys(sfPets).length} pets created`);
  console.log(`   - ${events.length} events created`);
  console.log(`   - ${sfPetServices.length} services created`);
  console.log(`   - ${activityCount} activities created`);
  console.log(`   - ${postCount} posts created`);
  console.log('\n🌉 Ready for SF beta launch!\n');
}

main()
  .catch((e) => {
    console.error('❌ Seed failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
