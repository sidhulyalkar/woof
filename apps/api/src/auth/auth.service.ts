import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import type { User } from '@woof/database';
import { compare, hash } from 'bcrypt';
import { randomUUID } from 'node:crypto';
import { UsersService } from '../users/users.service';
import { LoginDto } from './dto/login.dto';
import { RegisterDto } from './dto/register.dto';
import { SessionAuthorityService } from './session-authority.service';

type SafeUser = Omit<User, 'passwordHash'>;

type AccessTokenClaims = {
  sub: string;
  email: string;
  handle: string;
  sid: string;
};

function withoutPassword(user: User): SafeUser {
  const { passwordHash, ...safeUser } = user;
  void passwordHash;
  return safeUser;
}

@Injectable()
export class AuthService {
  constructor(
    private readonly usersService: UsersService,
    private readonly jwtService: JwtService,
    private readonly sessionAuthority: SessionAuthorityService
  ) {}

  async validateUser(email: string, password: string): Promise<SafeUser> {
    const user = await this.usersService.findByEmail(email);

    if (!user || !user.passwordHash) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isPasswordValid = await compare(password, user.passwordHash);

    if (!isPasswordValid) {
      throw new UnauthorizedException('Invalid credentials');
    }

    return withoutPassword(user);
  }

  async login(loginDto: LoginDto) {
    const user = await this.validateUser(loginDto.email, loginDto.password);
    const accessToken = await this.issueAccessToken(user);

    return {
      access_token: accessToken,
      user: {
        id: user.id,
        email: user.email,
        handle: user.handle,
        bio: user.bio,
        avatarUrl: user.avatarUrl,
        points: user.points,
      },
    };
  }

  async register(registerDto: RegisterDto) {
    const hashedPassword = await hash(registerDto.password, 10);

    const user = await this.usersService.create({
      email: registerDto.email,
      handle: registerDto.handle,
      passwordHash: hashedPassword,
      bio: registerDto.bio,
      authProvider: 'EMAIL',
    });

    const userWithoutPassword = withoutPassword(user);
    const accessToken = await this.issueAccessToken(userWithoutPassword);

    return {
      access_token: accessToken,
      user: userWithoutPassword,
    };
  }

  async logout(userId: string, sessionId: string) {
    await this.sessionAuthority.revokeSession(userId, sessionId, 'LOGOUT');
    return { success: true };
  }

  async logoutAll(userId: string) {
    const result = await this.sessionAuthority.revokeAllSessions(userId, 'LOGOUT_ALL');
    return { success: true, revokedCount: result.revokedCount };
  }

  async getProfile(userId: string) {
    try {
      return await this.usersService.findSelfById(userId);
    } catch {
      throw new UnauthorizedException('User not found');
    }
  }

  private async issueAccessToken(user: SafeUser) {
    const sessionId = randomUUID();
    const payload: AccessTokenClaims = {
      sub: user.id,
      email: user.email,
      handle: user.handle,
      sid: sessionId,
    };
    const token = this.jwtService.sign(payload);
    const decoded = this.jwtService.decode(token) as { exp?: unknown } | null;
    const exp = decoded?.exp;

    if (typeof exp !== 'number' || !Number.isSafeInteger(exp) || exp <= 0) {
      throw new UnauthorizedException('Session is unavailable');
    }

    const expiresAt = new Date(exp * 1_000);
    if (!Number.isFinite(expiresAt.getTime()) || expiresAt.getTime() <= Date.now()) {
      throw new UnauthorizedException('Session is unavailable');
    }

    await this.sessionAuthority.createSession({
      id: sessionId,
      userId: user.id,
      expiresAt,
    });

    return token;
  }
}
