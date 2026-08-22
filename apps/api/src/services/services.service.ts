import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { CreateBusinessDto } from './dto/create-business.dto';
import { UpdateBusinessDto } from './dto/update-business.dto';
import { TrackServiceIntentDto, ServiceIntentFollowupDto } from './dto/track-service-intent.dto';

@Injectable()
export class ServicesService {
  constructor(private prisma: PrismaService) {}

  async createBusiness(dto: CreateBusinessDto) {
    return this.prisma.business.create({
      data: {
        name: dto.name,
        type: dto.type,
        address: dto.address,
        lat: dto.lat ?? 0,
        lng: dto.lng ?? 0,
        phone: dto.phone,
        website: dto.website,
        hours: dto.hours || {},
        photos: dto.photos || [],
        amenities: dto.services || [],
      },
    });
  }

  async findAllBusinesses(type?: string, lat?: number, lng?: number, radiusKm?: number) {
    const where: Prisma.BusinessWhereInput = {};

    if (type) {
      where.type = type;
    }

    void lat;
    void lng;
    void radiusKm;

    return this.prisma.business.findMany({
      where,
      orderBy: { createdAt: 'desc' },
    });
  }

  async findOneBusiness(id: string) {
    const business = await this.prisma.business.findUnique({
      where: { id },
    });

    if (!business) {
      throw new NotFoundException(`Business ${id} not found`);
    }

    return business;
  }

  async updateBusiness(id: string, dto: UpdateBusinessDto) {
    await this.findOneBusiness(id);

    return this.prisma.business.update({
      where: { id },
      data: {
        ...dto,
      },
    });
  }

  async removeBusiness(id: string) {
    await this.findOneBusiness(id);

    return this.prisma.business.delete({
      where: { id },
    });
  }

  async trackIntent(userId: string, dto: TrackServiceIntentDto) {
    await this.findOneBusiness(dto.businessId);

    return this.prisma.serviceIntent.create({
      data: {
        userId,
        businessId: dto.businessId,
        action: dto.action,
      },
    });
  }

  async getUserIntents(userId: string) {
    return this.prisma.serviceIntent.findMany({
      where: { userId },
      include: {
        business: true,
      },
      orderBy: { createdAt: 'desc' },
    });
  }

  async getIntentsNeedingFollowup() {
    const twentyFourHoursAgo = new Date();
    twentyFourHoursAgo.setHours(twentyFourHoursAgo.getHours() - 24);

    return this.prisma.serviceIntent.findMany({
      where: {
        action: 'tap_book',
        conversionFollowup: null,
        createdAt: {
          lte: twentyFourHoursAgo,
        },
      },
      include: {
        business: true,
      },
    });
  }

  async recordFollowup(intentId: string, dto: ServiceIntentFollowupDto) {
    return this.prisma.serviceIntent.update({
      where: { id: intentId },
      data: {
        conversionFollowup: dto.converted,
        followupAskedAt: new Date(),
        followupResponse: dto.notes,
      },
    });
  }

  async getConversionStats(businessId?: string) {
    const where: Prisma.ServiceIntentWhereInput = {};
    if (businessId) {
      where.businessId = businessId;
    }

    const intents = await this.prisma.serviceIntent.findMany({
      where,
    });

    const tapBookIntents = intents.filter((intent) => intent.action === 'tap_book');
    const conversions = tapBookIntents.filter((intent) => intent.conversionFollowup === true);

    return {
      totalIntents: intents.length,
      tapBookCount: tapBookIntents.length,
      conversions: conversions.length,
      conversionRate:
        tapBookIntents.length > 0 ? (conversions.length / tapBookIntents.length) * 100 : 0,
    };
  }
}
