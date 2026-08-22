import { ConflictException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { UpdateProfileDto } from './dto/update-profile.dto';

function prismaErrorCode(error: unknown): string | undefined {
  if (!error || typeof error !== 'object' || !('code' in error)) return undefined;
  const code = (error as { code?: unknown }).code;
  return typeof code === 'string' ? code : undefined;
}

@Injectable()
export class UsersService {
  constructor(private prisma: PrismaService) {}

  async create(data: Prisma.UserCreateInput) {
    const existingUser = await this.prisma.user.findUnique({
      where: { email: data.email },
    });

    if (existingUser) {
      throw new ConflictException('User with this email already exists');
    }

    if (data.handle) {
      const existingHandle = await this.prisma.user.findUnique({
        where: { handle: data.handle },
      });

      if (existingHandle) {
        throw new ConflictException('This handle is already taken');
      }
    }

    return this.prisma.user.create({ data });
  }

  async findAll(skip = 0, take = 20) {
    const safeSkip = Math.max(0, Number(skip) || 0);
    const safeTake = Math.max(1, Math.min(Number(take) || 20, 100));

    const [users, total] = await Promise.all([
      this.prisma.user.findMany({
        skip: safeSkip,
        take: safeTake,
        select: {
          id: true,
          handle: true,
          bio: true,
          avatarUrl: true,
          points: true,
          isVerified: true,
          createdAt: true,
        },
      }),
      this.prisma.user.count(),
    ]);

    return { users, total, skip: safeSkip, take: safeTake };
  }

  /** Public/member-facing profile. Email and authentication fields never leave this projection. */
  async findById(id: string) {
    const user = await this.prisma.user.findUnique({
      where: { id },
      select: {
        id: true,
        handle: true,
        bio: true,
        avatarUrl: true,
        visibility: true,
        points: true,
        isVerified: true,
        createdAt: true,
        pets: {
          select: {
            id: true,
            name: true,
            species: true,
            breed: true,
            sex: true,
            birthdate: true,
            temperament: true,
            avatarUrl: true,
          },
        },
        _count: {
          select: {
            posts: true,
            activities: true,
          },
        },
      },
    });

    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }

    return user;
  }

  /** Authenticated self profile, including the account email but never passwordHash. */
  async findSelfById(id: string) {
    const user = await this.prisma.user.findUnique({
      where: { id },
      select: {
        id: true,
        handle: true,
        email: true,
        bio: true,
        avatarUrl: true,
        visibility: true,
        points: true,
        totalPoints: true,
        isVerified: true,
        createdAt: true,
        updatedAt: true,
        pets: {
          select: {
            id: true,
            name: true,
            species: true,
            breed: true,
            sex: true,
            birthdate: true,
            temperament: true,
            avatarUrl: true,
          },
        },
        _count: {
          select: {
            posts: true,
            activities: true,
          },
        },
      },
    });

    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }

    return user;
  }

  /** Authentication-only lookup. Keep private to server-side auth flows. */
  async findByEmail(email: string) {
    return this.prisma.user.findUnique({
      where: { email },
    });
  }

  async updateProfile(id: string, dto: UpdateProfileDto) {
    try {
      return await this.prisma.user.update({
        where: { id },
        data: {
          ...(dto.handle !== undefined ? { handle: dto.handle.trim().toLowerCase() } : {}),
          ...(dto.bio !== undefined ? { bio: dto.bio.trim() || null } : {}),
          ...(dto.avatarUrl !== undefined ? { avatarUrl: dto.avatarUrl } : {}),
          ...(dto.visibility !== undefined ? { visibility: dto.visibility } : {}),
        },
        select: {
          id: true,
          handle: true,
          email: true,
          bio: true,
          avatarUrl: true,
          visibility: true,
          points: true,
          totalPoints: true,
          isVerified: true,
          createdAt: true,
          pets: {
            select: {
              id: true,
              name: true,
              species: true,
              breed: true,
              sex: true,
              birthdate: true,
              temperament: true,
              avatarUrl: true,
            },
          },
          _count: {
            select: {
              posts: true,
              activities: true,
            },
          },
        },
      });
    } catch (error: unknown) {
      const code = prismaErrorCode(error);
      if (code === 'P2025') {
        throw new NotFoundException(`User with ID ${id} not found`);
      }
      if (code === 'P2002') {
        throw new ConflictException('This handle is already taken');
      }
      throw error;
    }
  }

  async update(id: string, data: Prisma.UserUpdateInput) {
    try {
      return await this.prisma.user.update({
        where: { id },
        data,
      });
    } catch (error: unknown) {
      const code = prismaErrorCode(error);
      if (code === 'P2025') {
        throw new NotFoundException(`User with ID ${id} not found`);
      }
      if (code === 'P2002') {
        throw new ConflictException('A unique user field is already in use');
      }
      throw error;
    }
  }

  async delete(id: string) {
    try {
      return await this.prisma.user.delete({ where: { id } });
    } catch (error: unknown) {
      if (prismaErrorCode(error) === 'P2025') {
        throw new NotFoundException(`User with ID ${id} not found`);
      }
      throw error;
    }
  }
}
