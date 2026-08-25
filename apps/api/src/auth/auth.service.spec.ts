import { UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { Test, TestingModule } from '@nestjs/testing';
import { hash } from 'bcrypt';
import { UsersService } from '../users/users.service';
import { AuthService } from './auth.service';
import { SessionAuthorityService } from './session-authority.service';

describe('AuthService', () => {
  let service: AuthService;
  let jwtService: { sign: jest.Mock; decode: jest.Mock };
  let usersService: {
    findByEmail: jest.Mock;
    create: jest.Mock;
    findSelfById: jest.Mock;
  };
  let sessionAuthority: {
    createSession: jest.Mock;
    revokeSession: jest.Mock;
    revokeAllSessions: jest.Mock;
  };

  beforeEach(async () => {
    usersService = {
      findByEmail: jest.fn(),
      create: jest.fn(),
      findSelfById: jest.fn(),
    };
    jwtService = {
      sign: jest.fn(() => 'mock-jwt-token'),
      decode: jest.fn(() => ({ exp: Math.floor(Date.now() / 1_000) + 3_600 })),
    };
    sessionAuthority = {
      createSession: jest.fn().mockResolvedValue(undefined),
      revokeSession: jest.fn().mockResolvedValue({ revoked: true }),
      revokeAllSessions: jest.fn().mockResolvedValue({ revokedCount: 2 }),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AuthService,
        { provide: UsersService, useValue: usersService },
        { provide: JwtService, useValue: jwtService },
        { provide: SessionAuthorityService, useValue: sessionAuthority },
      ],
    }).compile();

    service = module.get<AuthService>(AuthService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('validateUser', () => {
    it('returns a user without passwordHash when credentials are valid', async () => {
      usersService.findByEmail.mockResolvedValue({
        id: '123',
        email: 'test@example.com',
        passwordHash: await hash('password123', 4),
        handle: 'testuser',
        bio: 'Test bio',
      });

      const result = await service.validateUser('test@example.com', 'password123');

      expect(result).toEqual({
        id: '123',
        email: 'test@example.com',
        handle: 'testuser',
        bio: 'Test bio',
      });
      expect(result).not.toHaveProperty('passwordHash');
    });

    it('rejects an unknown email without revealing which credential failed', async () => {
      usersService.findByEmail.mockResolvedValue(null);

      await expect(service.validateUser('wrong@example.com', 'password123')).rejects.toThrow(
        new UnauthorizedException('Invalid credentials')
      );
    });

    it('rejects an incorrect password', async () => {
      usersService.findByEmail.mockResolvedValue({
        id: '123',
        email: 'test@example.com',
        passwordHash: await hash('password123', 4),
        handle: 'testuser',
      });

      await expect(service.validateUser('test@example.com', 'wrongpassword')).rejects.toThrow(
        new UnauthorizedException('Invalid credentials')
      );
    });
  });

  describe('login', () => {
    it('persists a finite server-owned session before returning the canonical token', async () => {
      usersService.findByEmail.mockResolvedValue({
        id: '123',
        email: 'test@example.com',
        passwordHash: await hash('password123', 4),
        handle: 'testuser',
        bio: 'Test bio',
        avatarUrl: null,
        points: 9,
      });

      const result = await service.login({
        email: 'test@example.com',
        password: 'password123',
      });

      expect(result).toEqual({
        access_token: 'mock-jwt-token',
        user: {
          id: '123',
          email: 'test@example.com',
          handle: 'testuser',
          bio: 'Test bio',
          avatarUrl: null,
          points: 9,
        },
      });

      const payload = jwtService.sign.mock.calls[0][0] as { sid: string };
      expect(payload).toEqual({
        sub: '123',
        email: 'test@example.com',
        handle: 'testuser',
        sid: expect.any(String),
      });
      expect(sessionAuthority.createSession).toHaveBeenCalledWith({
        id: payload.sid,
        userId: '123',
        expiresAt: expect.any(Date),
      });
    });
  });

  describe('register', () => {
    it('delegates uniqueness and persists a server-owned session', async () => {
      usersService.create.mockResolvedValue({
        id: '456',
        email: 'new@example.com',
        handle: 'newuser',
        passwordHash: 'hashed-password',
        bio: null,
        authProvider: 'EMAIL',
      });

      const result = await service.register({
        email: 'new@example.com',
        handle: 'newuser',
        password: 'password123',
      });

      expect(usersService.create).toHaveBeenCalledWith(
        expect.objectContaining({
          email: 'new@example.com',
          handle: 'newuser',
          authProvider: 'EMAIL',
          passwordHash: expect.any(String),
        })
      );
      const payload = jwtService.sign.mock.calls[0][0] as { sid: string };
      expect(sessionAuthority.createSession).toHaveBeenCalledWith({
        id: payload.sid,
        userId: '456',
        expiresAt: expect.any(Date),
      });
      expect(result.access_token).toBe('mock-jwt-token');
      expect(result.user).not.toHaveProperty('passwordHash');
    });

    it('propagates user-creation conflicts', async () => {
      usersService.create.mockRejectedValue(new Error('duplicate account'));

      await expect(
        service.register({
          email: 'existing@example.com',
          handle: 'existinguser',
          password: 'password123',
        })
      ).rejects.toThrow('duplicate account');
      expect(sessionAuthority.createSession).not.toHaveBeenCalled();
    });
  });

  it('revokes the current server session on logout', async () => {
    await expect(service.logout('user-1', 'session-1')).resolves.toEqual({ success: true });
    expect(sessionAuthority.revokeSession).toHaveBeenCalledWith('user-1', 'session-1', 'LOGOUT');
  });

  it('revokes every active server session on logout-all', async () => {
    await expect(service.logoutAll('user-1')).resolves.toEqual({ success: true, revokedCount: 2 });
    expect(sessionAuthority.revokeAllSessions).toHaveBeenCalledWith('user-1', 'LOGOUT_ALL');
  });

  it('fails closed if the signed token does not contain a finite expiry', async () => {
    usersService.findByEmail.mockResolvedValue({
      id: '123',
      email: 'test@example.com',
      passwordHash: await hash('password123', 4),
      handle: 'testuser',
    });
    jwtService.decode.mockReturnValueOnce({});

    await expect(
      service.login({ email: 'test@example.com', password: 'password123' })
    ).rejects.toThrow(new UnauthorizedException('Session is unavailable'));
    expect(sessionAuthority.createSession).not.toHaveBeenCalled();
  });
});
