import { PrismaClient } from '@prisma/client';
import { hash } from 'bcrypt';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Seeding database...');

  // Clear existing data (development only!)
  await prisma.telemetry.deleteMany();
  await prisma.comment.deleteMany();
  await prisma.like.deleteMany();
  await prisma.post.deleteMany();
  await prisma.meetupInvite.deleteMany();
  await prisma.meetup.deleteMany();
  await prisma.petEdge.deleteMany();
  await prisma.mutualGoal.deleteMany();
  await prisma.activity.deleteMany();
  await prisma.device.deleteMany();
  await prisma.pet.deleteMany();
  await prisma.notification.deleteMany();
  await prisma.integrationToken.deleteMany();
  await prisma.reward.deleteMany();
  await prisma.user.deleteMany();
  await prisma.place.deleteMany();

  // Create demo users
  const passwordHash = await hash('password123', 10);

  const user1 = await prisma.user.create({
    data: {
      handle: 'petlover_nyc',
      email: 'demo@woof.com',
      passwordHash,
      bio: 'NYC dog dad 🐕 | Golden Retriever parent | Love hiking trails!',
      avatarUrl: 'https://i.pravatar.cc/300?img=12',
      homeLocation: { lat: 40.7128, lng: -74.006 }, // NYC
      points: 1250,
    },
  });

  const user2 = await prisma.user.create({
    data: {
      handle: 'dogmom_sf',
      email: 'sarah@woof.com',
      passwordHash,
      bio: 'SF based 🌉 | Husky mom | Running enthusiast',
      avatarUrl: 'https://i.pravatar.cc/300?img=25',
      homeLocation: { lat: 37.7749, lng: -122.4194 }, // SF
      points: 890,
    },
  });

  const user3 = await prisma.user.create({
    data: {
      handle: 'adventures_with_max',
      email: 'john@woof.com',
      passwordHash,
      bio: 'Adventure seeker 🏔️ | Labrador dad | Trail runner',
      avatarUrl: 'https://i.pravatar.cc/300?img=33',
      homeLocation: { lat: 40.7128, lng: -74.006 },
      points: 2100,
    },
  });

  // Create pets
  const buddy = await prisma.pet.create({
    data: {
      ownerId: user1.id,
      name: 'Buddy',
      species: 'DOG',
      breed: 'Golden Retriever',
      sex: 'MALE',
      birthdate: new Date('2020-03-15'),
      temperament: { friendly: 5, energetic: 4, playful: 5, trainable: 4 },
      avatarUrl: 'https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg',
    },
  });

  const luna = await prisma.pet.create({
    data: {
      ownerId: user2.id,
      name: 'Luna',
      species: 'DOG',
      breed: 'Siberian Husky',
      sex: 'FEMALE',
      birthdate: new Date('2019-08-22'),
      temperament: { friendly: 4, energetic: 5, playful: 5, independent: 4 },
      avatarUrl: 'https://images.dog.ceo/breeds/husky/n02110185_10047.jpg',
    },
  });

  const max = await prisma.pet.create({
    data: {
      ownerId: user3.id,
      name: 'Max',
      species: 'DOG',
      breed: 'Labrador Retriever',
      sex: 'MALE',
      birthdate: new Date('2018-12-10'),
      temperament: { friendly: 5, energetic: 5, playful: 4, loyal: 5 },
      avatarUrl: 'https://images.dog.ceo/breeds/labrador/n02099712_3503.jpg',
    },
  });

  // Create pet friendships
  await prisma.petEdge.create({
    data: {
      petAId: buddy.id,
      petBId: max.id,
      compatibilityScore: 0.92,
      status: 'CONFIRMED',
      lastInteractionAt: new Date(),
    },
  });

  await prisma.petEdge.create({
    data: {
      petAId: buddy.id,
      petBId: luna.id,
      compatibilityScore: 0.78,
      status: 'PROPOSED',
    },
  });

  // Create activities
  const activity1 = await prisma.activity.create({
    data: {
      userId: user1.id,
      petId: buddy.id,
      type: 'WALK',
      startedAt: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
      endedAt: new Date(Date.now() - 1 * 60 * 60 * 1000), // 1 hour ago
      route: {
        type: 'LineString',
        coordinates: [
          [-74.006, 40.7128],
          [-74.005, 40.7138],
          [-74.004, 40.7148],
        ],
      },
      humanMetrics: { steps: 3245, calories: 185, hr_avg: 98 },
      petMetrics: { distance: 2.3, active_time: 45 },
    },
  });

  // Create posts
  const post1 = await prisma.post.create({
    data: {
      authorUserId: user1.id,
      petId: buddy.id,
      text: 'Beautiful morning walk in Central Park! 🌳 Buddy made 3 new friends today!',
      mediaUrls: ['https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'],
      activityId: activity1.id,
    },
  });

  const post2 = await prisma.post.create({
    data: {
      authorUserId: user2.id,
      petId: luna.id,
      text: 'Luna absolutely loves the snow! ❄️ Can\'t wait for more winter adventures!',
      mediaUrls: ['https://images.dog.ceo/breeds/husky/n02110185_10047.jpg'],
    },
  });

  // Create likes and comments
  await prisma.like.create({
    data: {
      postId: post1.id,
      userId: user2.id,
    },
  });

  await prisma.like.create({
    data: {
      postId: post1.id,
      userId: user3.id,
    },
  });

  await prisma.comment.create({
    data: {
      postId: post1.id,
      userId: user2.id,
      text: 'So adorable! Luna would love to meet Buddy!',
    },
  });

  // Create a meetup
  const meetup = await prisma.meetup.create({
    data: {
      title: 'Central Park Dog Meetup',
      location: { type: 'Point', coordinates: [-73.9654, 40.7829] },
      radiusM: 500,
      startsAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // Tomorrow
      creatorUserId: user1.id,
    },
  });

  await prisma.meetupInvite.create({
    data: {
      meetupId: meetup.id,
      userId: user3.id,
      petId: max.id,
      rsvp: 'YES',
    },
  });

  // Create mutual goals
  await prisma.mutualGoal.create({
    data: {
      userId: user1.id,
      petId: buddy.id,
      goalType: 'DISTANCE',
      period: 'WEEK',
      targetNumber: 25.0, // 25 km per week
      progress: 12.5,
    },
  });

  // Create places
  await prisma.place.create({
    data: {
      name: 'Central Park Dog Run',
      kind: 'DOG_PARK',
      location: { type: 'Point', coordinates: [-73.9654, 40.7829] },
      score: 4.8,
      hours: { monday: '6:00-21:00', tuesday: '6:00-21:00', wednesday: '6:00-21:00' },
      petPolicies: { dogs_allowed: true, off_leash: true, water_available: true },
    },
  });

  console.log('✅ Database seeded successfully!');
  console.log(`
    Created:
    - 3 users
    - 3 pets
    - 2 pet friendships
    - 1 activity
    - 2 posts
    - 2 likes
    - 1 comment
    - 1 meetup
    - 1 mutual goal
    - 1 place
  `);
}

main()
  .catch((e) => {
    console.error('❌ Seeding failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
