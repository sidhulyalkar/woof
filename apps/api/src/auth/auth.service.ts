import { ConflictException, Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import type { User } from '@woof/database';
import { compare, hash } from 'bcrypt';
import { createHash, randomUUID } from 'node:crypto';
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

type CanonicalRegistration = {
  email: string;
  handle: string;
  password: string;
  bio: string | null;
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
    const submittedEmail = email.trim();
    const canonicalEmail = submittedEmail.toLowerCase();
    let user = await this.usersService.findByEmail(canonicalEmail);

    // New email registrations are canonicalized. Preserve sign-in compatibility
    // for any older account that was stored with mixed-case email before that rule.
    if (!user && submittedEmail !== canonicalEmail) {
      user = await this.usersService.findByEmail(submittedEmail);
    }

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
    const canonical = this.canonicalRegistration(registerDto);
    const replaySafeId = registerDto.registrationKey
      ? this.replaySafeRegistrationId(canonical.email, registerDto.registrationKey)
      : null;

    if (replaySafeId) {
      const existing = await this.usersService.findByEmail(canonical.email);
      if (existing) {
        return this.resumeRegistrationReplay(existing, replaySafeId, canonical);
      }
    }

    const hashedPassword = await hash(canonical.password, 10);

    try {
      const user = await this.usersService.create({
        ...(replaySafeId ? { id: replaySafeId } : {}),
        email: canonical.email,
        handle: canonical.handle,
        passwordHash: hashedPassword,
        bio: canonical.bio,
        authProvider: 'EMAIL',
      });

      return this.finishRegistration(user);
    } catch (error) {
      if (!replaySafeId) throw error;

      // A concurrent/exact retry can race the initial unique insert. Re-read the
      // canonical email and only recover if the deterministic transaction identity
      // and every supplied account field prove this is the same registration.
      const existing = await this.usersService.findByEmail(canonical.email);
      if (!existing) throw error;
      return this.resumeRegistrationReplay(existing, replaySafeId, canonical);
    }
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

  private canonicalRegistration(registerDto: RegisterDto): CanonicalRegistration {
    return {
      email: registerDto.email.trim().toLowerCase(),
      handle: registerDto.handle.trim().toLowerCase(),
      password: registerDto.password,
      bio: registerDto.bio?.trim() || null,
    };
  }

  private async resumeRegistrationReplay(
    existing: User,
    replaySafeId: string,
    canonical: CanonicalRegistration
  ) {
    if (existing.id !== replaySafeId) {
      throw new ConflictException('User with this email already exists');
    }

    const passwordMatches =
      existing.authProvider === 'EMAIL' &&
      Boolean(existing.passwordHash) &&
      (await compare(canonical.password, existing.passwordHash as string));
    const fieldsMatch =
      existing.email.trim().toLowerCase() === canonical.email &&
      existing.handle === canonical.handle &&
      (existing.bio?.trim() || null) === canonical.bio;

    if (!passwordMatches || !fieldsMatch) {
      throw new ConflictException('Registration key was replayed with divergent account fields');
    }

    return this.finishRegistration(existing);
  }

  private replaySafeRegistrationId(email: string, registrationKey: string): string {
    const digest = createHash('sha256')
      .update(`woof-email-registration-v1:${email}:${registrationKey.trim()}`)
      .digest('hex');
    const uuid = digest.slice(0, 32).split('');

    // UUIDv8 reserves the version nibble for application-defined deterministic IDs.
    uuid[12] = '8';
    uuid[16] = ((Number.parseInt(uuid[16] ?? '0', 16) & 0x3) | 0x8).toString(16);
    const value = uuid.join('');
    return `${value.slice(0, 8)}-${value.slice(8, 12)}-${value.slice(12, 16)}-${value.slice(16, 20)}-${value.slice(20, 32)}`;
  }

  private async finishRegistration(user: User) {
    const userWithoutPassword = withoutPassword(user);
    const accessToken = await this.issueAccessToken(userWithoutPassword);

    return {
      access_token: accessToken,
      user: userWithoutPassword,
    };
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
