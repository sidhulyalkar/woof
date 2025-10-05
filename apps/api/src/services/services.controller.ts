import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  UseGuards,
  Request,
  Query,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiQuery } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { ServicesService } from './services.service';
import { CreateBusinessDto } from './dto/create-business.dto';
import { UpdateBusinessDto } from './dto/update-business.dto';
import { TrackServiceIntentDto, ServiceIntentFollowupDto } from './dto/track-service-intent.dto';

@ApiTags('services')
@Controller('services')
export class ServicesController {
  constructor(private readonly servicesService: ServicesService) {}

  // Business CRUD endpoints

  @Post('businesses')
  @ApiOperation({ summary: 'Create a new business listing' })
  @ApiResponse({ status: 201, description: 'Business created successfully' })
  async createBusiness(@Body() createBusinessDto: CreateBusinessDto) {
    return this.servicesService.createBusiness(createBusinessDto);
  }

  @Get('businesses')
  @ApiOperation({ summary: 'Get all businesses with optional filters' })
  @ApiQuery({ name: 'type', required: false, description: 'Filter by business type' })
  @ApiQuery({ name: 'lat', required: false, description: 'Latitude for proximity search' })
  @ApiQuery({ name: 'lng', required: false, description: 'Longitude for proximity search' })
  @ApiQuery({ name: 'radiusKm', required: false, description: 'Search radius in kilometers' })
  @ApiResponse({ status: 200, description: 'List of businesses' })
  async findAllBusinesses(
    @Query('type') type?: string,
    @Query('lat') lat?: string,
    @Query('lng') lng?: string,
    @Query('radiusKm') radiusKm?: string,
  ) {
    const latNum = lat ? parseFloat(lat) : undefined;
    const lngNum = lng ? parseFloat(lng) : undefined;
    const radiusNum = radiusKm ? parseFloat(radiusKm) : undefined;

    return this.servicesService.findAllBusinesses(type, latNum, lngNum, radiusNum);
  }

  @Get('businesses/:id')
  @ApiOperation({ summary: 'Get a specific business' })
  @ApiResponse({ status: 200, description: 'Business details' })
  @ApiResponse({ status: 404, description: 'Business not found' })
  async findOneBusiness(@Param('id') id: string) {
    return this.servicesService.findOneBusiness(id);
  }

  @Patch('businesses/:id')
  @ApiOperation({ summary: 'Update a business' })
  @ApiResponse({ status: 200, description: 'Business updated successfully' })
  @ApiResponse({ status: 404, description: 'Business not found' })
  async updateBusiness(@Param('id') id: string, @Body() updateBusinessDto: UpdateBusinessDto) {
    return this.servicesService.updateBusiness(id, updateBusinessDto);
  }

  @Delete('businesses/:id')
  @ApiOperation({ summary: 'Delete a business' })
  @ApiResponse({ status: 200, description: 'Business deleted successfully' })
  @ApiResponse({ status: 404, description: 'Business not found' })
  async removeBusiness(@Param('id') id: string) {
    return this.servicesService.removeBusiness(id);
  }

  // Service Intent endpoints

  @Post('intents')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Track user intent to use a service' })
  @ApiResponse({ status: 201, description: 'Intent tracked successfully' })
  async trackIntent(@Request() req: any, @Body() trackServiceIntentDto: TrackServiceIntentDto) {
    return this.servicesService.trackIntent(req.user.id, trackServiceIntentDto);
  }

  @Get('intents/me')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get all service intents for the current user' })
  @ApiResponse({ status: 200, description: 'List of user service intents' })
  async getUserIntents(@Request() req: any) {
    return this.servicesService.getUserIntents(req.user.id);
  }

  @Get('intents/followup-needed')
  @ApiOperation({ summary: 'Get intents that need 24h follow-up' })
  @ApiResponse({ status: 200, description: 'List of intents needing follow-up' })
  async getIntentsNeedingFollowup() {
    return this.servicesService.getIntentsNeedingFollowup();
  }

  @Patch('intents/:id/followup')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Record follow-up response for a service intent' })
  @ApiResponse({ status: 200, description: 'Followup recorded successfully' })
  async recordFollowup(
    @Param('id') id: string,
    @Body() serviceIntentFollowupDto: ServiceIntentFollowupDto,
  ) {
    return this.servicesService.recordFollowup(id, serviceIntentFollowupDto);
  }

  @Get('stats/conversion')
  @ApiOperation({ summary: 'Get conversion statistics' })
  @ApiQuery({ name: 'businessId', required: false, description: 'Filter by specific business' })
  @ApiResponse({ status: 200, description: 'Conversion statistics' })
  async getConversionStats(@Query('businessId') businessId?: string) {
    return this.servicesService.getConversionStats(businessId);
  }
}
