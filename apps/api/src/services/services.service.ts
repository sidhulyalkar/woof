import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateBusinessDto } from './dto/create-business.dto';
import { UpdateBusinessDto } from './dto/update-business.dto';
import { TrackServiceIntentDto, ServiceIntentFollowupDto } from './dto/track-service-intent.dto';

@Injectable()
export class ServicesService {
  constructor(private prisma: PrismaService) {}

  /**
   * Create a new business listing
   */
  async createBusiness(dto: CreateBusinessDto) {
    return this.prisma.business.create({
      data: {
        name: dto.name,
        type: dto.type,
        description: dto.description,
        address: dto.address,
        lat: dto.lat,
        lng: dto.lng,
        phone: dto.phone,
        website: dto.website,
        hours: dto.hours || {},
        services: dto.services || [],
        photos: dto.photos || [],
      },
    });
  }

  /**
   * Get all businesses with optional filters
   */
  async findAllBusinesses(type?: string, lat?: number, lng?: number, radiusKm?: number) {
    const where: any = {};

    if (type) {
      where.type = type;
    }

    // TODO: Add geospatial filtering when lat/lng/radiusKm are provided
    // For now, just return all matching type

    return this.prisma.business.findMany({
      where,
      orderBy: { createdAt: 'desc' },
    });
  }

  /**
   * Get a specific business
   */
  async findOneBusiness(id: string) {
    const business = await this.prisma.business.findUnique({
      where: { id },
    });

    if (!business) {
      throw new NotFoundException(`Business ${id} not found`);
    }

    return business;
  }

  /**
   * Update a business
   */
  async updateBusiness(id: string, dto: UpdateBusinessDto) {
    await this.findOneBusiness(id); // Ensure exists

    return this.prisma.business.update({
      where: { id },
      data: {
        ...dto,
      },
    });
  }

  /**
   * Delete a business
   */
  async removeBusiness(id: string) {
    await this.findOneBusiness(id); // Ensure exists

    return this.prisma.business.delete({
      where: { id },
    });
  }

  /**
   * Track service intent (view, tap call, tap directions, etc.)
   */
  async trackIntent(userId: string, dto: TrackServiceIntentDto) {
    // Ensure business exists
    await this.findOneBusiness(dto.businessId);

    return this.prisma.serviceIntent.create({
      data: {
        userId,
        businessId: dto.businessId,
        action: dto.action,
      },
    });
  }

  /**
   * Get all service intents for a user
   */
  async getUserIntents(userId: string) {
    return this.prisma.serviceIntent.findMany({
      where: { userId },
      include: {
        business: true,
      },
      orderBy: { createdAt: 'desc' },
    });
  }

  /**
   * Get intents that need follow-up (24h after tap_book action, no followup yet)
   */
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

  /**
   * Record followup response for a service intent
   */
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

  /**
   * Get conversion stats for businesses
   */
  async getConversionStats(businessId?: string) {
    const where: any = {};
    if (businessId) {
      where.businessId = businessId;
    }

    const intents = await this.prisma.serviceIntent.findMany({
      where,
    });

    const tapBookIntents = intents.filter((i: any) => i.action === 'tap_book');
    const conversions = tapBookIntents.filter((i: any) => i.conversionFollowup === true);

    return {
      totalIntents: intents.length,
      tapBookCount: tapBookIntents.length,
      conversions: conversions.length,
      conversionRate:
        tapBookIntents.length > 0 ? (conversions.length / tapBookIntents.length) * 100 : 0,
    };
  }
}
