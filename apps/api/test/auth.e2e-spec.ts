import { INestApplication, ValidationPipe } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import * as request from 'supertest';
import { AppModule } from '../src/app.module';
import { PrismaService } from '../src/prisma/prisma.service';

describe('Auth (e2e)', () => {
  let app: INestApplication;
  let prisma: PrismaService;

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.setGlobalPrefix('api/v1');
    app.useGlobalPipes(
      new ValidationPipe({
        whitelist: true,
        forbidNonWhitelisted: true,
        transform: true,
      })
    );

    prisma = app.get<PrismaService>(PrismaService);
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  beforeEach(async () => {
    await prisma.user.deleteMany();
  });

  describe('/auth/register (POST)', () => {
    it('should register a new user', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({
          email: 'test@example.com',
          handle: 'testuser',
          password: 'password123',
        })
        .expect(201)
        .expect((res) => {
          expect(res.body).toHaveProperty('access_token');
          expect(res.body).toHaveProperty('user');
          expect(res.body.user.email).toBe('test@example.com');
          expect(res.body.user.handle).toBe('testuser');
          expect(res.body.user).not.toHaveProperty('passwordHash');
        });
    });

    it('canonicalizes new email and handle identity', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({
          email: 'Canonical.User@Example.COM',
          handle: 'TrailPaws',
          password: 'password123',
        })
        .expect(201)
        .expect((res) => {
          expect(res.body.user.email).toBe('canonical.user@example.com');
          expect(res.body.user.handle).toBe('trailpaws');
        });
    });

    it('recovers an exact replay as the same account with a fresh session', async () => {
      const payload = {
        email: 'replay@example.com',
        handle: 'replayuser',
        password: 'password123',
        bio: 'weekend hikes',
        registrationKey: '7efc01f2-0f7e-45e1-b923-748d6f727ef0',
      };

      const first = await request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send(payload)
        .expect(201);
      const second = await request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send(payload)
        .expect(201);

      expect(second.body.user.id).toBe(first.body.user.id);
      expect(second.body.access_token).not.toBe(first.body.access_token);
      await expect(prisma.user.count({ where: { email: payload.email } })).resolves.toBe(1);
    });

    it('fails closed when the original replay key is reused with changed account fields', async () => {
      const payload = {
        email: 'divergent@example.com',
        handle: 'originaluser',
        password: 'password123',
        registrationKey: '7efc01f2-0f7e-45e1-b923-748d6f727ef0',
      };

      await request(app.getHttpServer()).post('/api/v1/auth/register').send(payload).expect(201);

      await request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({ ...payload, handle: 'changeduser' })
        .expect(409);
      await expect(prisma.user.count({ where: { email: payload.email } })).resolves.toBe(1);
    });

    it('does not authorize replay recovery for the same email under a different key', async () => {
      const payload = {
        email: 'wrong-key@example.com',
        handle: 'originaluser',
        password: 'password123',
        registrationKey: '7efc01f2-0f7e-45e1-b923-748d6f727ef0',
      };

      await request(app.getHttpServer()).post('/api/v1/auth/register').send(payload).expect(201);

      await request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({ ...payload, registrationKey: 'a4fddf1f-1a06-4ea7-b9f1-54ca772ef5dc' })
        .expect(409);
    });

    it('rejects malformed registration replay keys at validation', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({
          email: 'bad-key@example.com',
          handle: 'badkeyuser',
          password: 'password123',
          registrationKey: 'not-a-uuid',
        })
        .expect(400);
    });

    it('should fail with invalid email', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({
          email: 'invalid-email',
          handle: 'testuser',
          password: 'password123',
        })
        .expect(400);
    });

    it('should fail with short password', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({
          email: 'test@example.com',
          handle: 'testuser',
          password: '123',
        })
        .expect(400);
    });

    it('should fail with duplicate email', async () => {
      await request(app.getHttpServer()).post('/api/v1/auth/register').send({
        email: 'duplicate@example.com',
        handle: 'user1',
        password: 'password123',
      });

      return request(app.getHttpServer())
        .post('/api/v1/auth/register')
        .send({
          email: 'duplicate@example.com',
          handle: 'user2',
          password: 'password123',
        })
        .expect(409);
    });
  });

  describe('/auth/login (POST)', () => {
    beforeEach(async () => {
      await request(app.getHttpServer()).post('/api/v1/auth/register').send({
        email: 'login@example.com',
        handle: 'loginuser',
        password: 'password123',
      });
    });

    it('should login with valid credentials', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/login')
        .send({
          email: 'login@example.com',
          password: 'password123',
        })
        .expect(200)
        .expect((res) => {
          expect(res.body).toHaveProperty('access_token');
          expect(res.body).toHaveProperty('user');
          expect(res.body.user.email).toBe('login@example.com');
        });
    });

    it('should fail with wrong password', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/login')
        .send({
          email: 'login@example.com',
          password: 'wrongpassword',
        })
        .expect(401);
    });

    it('should fail with non-existent user', () => {
      return request(app.getHttpServer())
        .post('/api/v1/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: 'password123',
        })
        .expect(401);
    });
  });

  describe('/auth/me (GET)', () => {
    let authToken: string;

    beforeEach(async () => {
      const response = await request(app.getHttpServer()).post('/api/v1/auth/register').send({
        email: 'me@example.com',
        handle: 'meuser',
        password: 'password123',
      });
      authToken = response.body.access_token;
    });

    it('should return current user with valid token', () => {
      return request(app.getHttpServer())
        .get('/api/v1/auth/me')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
        .expect((res) => {
          expect(res.body.email).toBe('me@example.com');
          expect(res.body.handle).toBe('meuser');
        });
    });

    it('should fail without token', () => {
      return request(app.getHttpServer()).get('/api/v1/auth/me').expect(401);
    });

    it('should fail with invalid token', () => {
      return request(app.getHttpServer())
        .get('/api/v1/auth/me')
        .set('Authorization', 'Bearer invalid-token')
        .expect(401);
    });
  });
});
